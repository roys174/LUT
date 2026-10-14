"""A small web chat interface for the trained LM checkpoints.

    python -m lut_torch.chat_server --port 8000

Then open http://localhost:8000 (through an SSH tunnel when the server runs on a
compute node -- ``run_files/chat_server.sh`` prints the exact command).

Stdlib only, on purpose. This has to start on a cluster node with no internet and no
right to install anything, so the server is :mod:`http.server` and the frontend is three
static files. The only imports beyond the standard library are torch and this package.

Layout:

* :mod:`lut_torch.run_index` finds the runs without opening a checkpoint.
* :mod:`lut_torch.inference` loads one and samples from it.
* this module is transport: JSON for the listing, newline-delimited JSON for the token
  stream, and static files for the page.

NDJSON rather than server-sent events because the generate request is a POST (it carries
the whole transcript), and ``EventSource`` can only issue GETs. ``fetch`` + a streaming
reader handles both equally well, and NDJSON needs no framing beyond ``\\n``.
"""

import argparse
import json
import mimetypes
import os
import socket
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import run_index
from .run_index import TASK_LM, discover_runs

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_ui')
MAX_BODY_BYTES = 4 << 20


class ChatService:
    """Everything the handler needs that is not HTTP.

    One instance is shared by every request thread. The model cache has its own lock, and
    each loaded model has one too; ``generation_lock`` is coarser and serialises whole
    generations, so two browser tabs take turns rather than interleaving forward passes
    through the same module.
    """

    def __init__(self, results_root, device='auto', cache_size=2, task=TASK_LM):
        self.results_root = os.path.abspath(results_root)
        self.task = task
        self.generation_lock = threading.Lock()
        self._runs = {}
        self._tokenizers = {}
        self._tokenizer_lock = threading.Lock()
        # Imported here so that --help and a bad --results-root fail before torch is
        # paid for (it is a multi-second import on a cold filesystem).
        from .inference import ModelCache, make_tokenizer_adapter, resolve_device
        self._make_adapter = make_tokenizer_adapter
        self.device = resolve_device(device)
        self.cache = ModelCache(device=self.device, capacity=cache_size)

    # -- runs --------------------------------------------------------------

    def refresh(self):
        runs = discover_runs(self.results_root, task=self.task)
        self._runs = {run.run_id: run for run in runs}
        return runs

    def run(self, run_id):
        if run_id not in self._runs:
            self.refresh()
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"unknown run {run_id!r}")
        return run

    def checkpoint_for(self, run_id, kind):
        """The requested checkpoint, falling back to the other kind when absent.

        ``best`` is the default because for LM runs it is selected on validation loss,
        and these runs peak long before they end. It can also be the copy that is
        mid-write, which is why the fallback exists.
        """
        run = self.run(run_id)
        chosen = run.checkpoint(kind) or (run.checkpoints[0] if run.checkpoints else None)
        if chosen is None:
            raise FileNotFoundError(f"run {run_id!r} has no checkpoint on disk yet")
        return run, chosen

    # -- models ------------------------------------------------------------

    def model(self, run_id, kind='best'):
        run, ckpt = self.checkpoint_for(run_id, kind)
        try:
            return run, ckpt, self.cache.get(ckpt.path)
        except Exception:
            # A run whose job is still going rewrites `model_best.pt` on every validation
            # improvement; if we lost that race, the other copy is a complete file.
            other = 'last' if ckpt.kind == 'best' else 'best'
            fallback = run.checkpoint(other)
            if fallback is None:
                raise
            return run, fallback, self.cache.get(fallback.path)

    def tokenizer(self, run_id):
        """The run's tokenizer adapter alone -- no checkpoint read.

        Tokenizing the input box on every keystroke must not depend on a gigabyte of
        tables being resident, so this goes through the two fields ``config.txt``
        records rather than through a loaded model. The key keeps the vocabulary file in
        it, so a word run still reads exactly its own vocabulary while a BPE run (whose
        ``vocab_file`` is empty) reads none at all.

        An adapter and not a raw encoder: tiktoken has no ``tokens_from_ids`` and no
        ``<unk>``, so every caller below would otherwise have to know which tokenizer it
        is holding.
        """
        run = self.run(run_id)
        key = (run.tokenizer or 'word', run.vocab_file)
        with self._tokenizer_lock:
            if key not in self._tokenizers:
                self._tokenizers[key] = self._make_adapter(*key)
            return self._tokenizers[key]

    def tokenize(self, run_id, text):
        # The same entry point generation uses, so the strip under the input box shows
        # exactly the tokens that will be sent -- trailing <eol> included or not.
        adapter = self.tokenizer(run_id)
        ids = adapter.encode_prompt(text)
        # Only a tokenizer that *has* an out-of-vocabulary id can produce one. Comparing
        # against a hard-coded 0 here is what made the strip wrong under BPE, where id 0
        # is "!" -- an ordinary token that would have been flagged as missing vocabulary
        # (and in the frontend, counted into "N became <unk>").
        unk = ([int(i) == adapter.unk_id for i in ids] if adapter.has_unk
               else [False] * len(ids))
        run = self.run(run_id)
        context = run.context_size or 32
        return {
            'ids': ids,
            'tokens': adapter.pieces(ids),
            'unk': unk,
            'count': len(ids),
            'unk_count': sum(unk),
            # The UI drops the <unk> wording entirely rather than reporting zero of them,
            # so it needs the capability and not just the count.
            'has_unk': adapter.has_unk,
            'tokenizer': adapter.name,
            'context_size': context,
            # How many leading tokens the window throws away. Only the last `context`
            # reach the model; the UI strikes the rest through, because "the model
            # ignored the start of my message" is otherwise indistinguishable from a bug.
            'dropped': max(0, len(ids) - context),
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class ChatHandler(BaseHTTPRequestHandler):
    service = None                       # set by serve()
    server_version = 'lut-chat/1.0'
    # HTTP/1.0 keeps the streaming response framed by connection close, which is all the
    # token stream needs and avoids hand-rolling chunked encoding.
    protocol_version = 'HTTP/1.0'

    def log_message(self, fmt, *args):          # quieter than the default one-line-per-file
        # getattr, because a malformed request line is logged before `path` is assigned.
        if getattr(self, 'path', '').startswith('/api/'):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing ----------------------------------------------------------

    def _send(self, code, body, content_type='application/json; charset=utf-8',
              extra_headers=()):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for key, value in extra_headers:
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, payload, code=200):
        self._send(code, json.dumps(payload))

    def _error(self, code, message):
        self._json({'error': message}, code=code)

    def _body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError('request body too large')
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode('utf-8'))

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        try:
            if path in ('/', '/index.html'):
                return self._static('index.html')
            if path.startswith('/static/'):
                return self._static(path[len('/static/'):])
            if path == '/api/runs':
                return self._runs()
            if path == '/api/health':
                return self._json({'ok': True, 'device': str(self.service.device)})
            return self._error(404, f"no route for {path}")
        except Exception as exc:                      # noqa: BLE001
            traceback.print_exc()
            return self._error(500, str(exc))

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        try:
            if path == '/api/tokenize':
                return self._tokenize()
            if path == '/api/model':
                return self._model()
            if path == '/api/generate':
                return self._generate()
            return self._error(404, f"no route for {path}")
        except (BrokenPipeError, ConnectionResetError):
            return                                   # the browser went away mid-reply
        except KeyError as exc:
            return self._error(404, str(exc))
        except FileNotFoundError as exc:
            return self._error(409, str(exc))
        except Exception as exc:                      # noqa: BLE001
            traceback.print_exc()
            return self._error(500, str(exc))

    def _static(self, relative):
        # Refuse anything that escapes the UI directory.
        full = os.path.normpath(os.path.join(STATIC_DIR, relative))
        if not full.startswith(STATIC_DIR + os.sep) and full != STATIC_DIR:
            return self._error(403, 'forbidden')
        if not os.path.isfile(full):
            return self._error(404, f"no such file {relative}")
        content_type = mimetypes.guess_type(full)[0] or 'application/octet-stream'
        if content_type.startswith('text/') or content_type.endswith('javascript'):
            content_type += '; charset=utf-8'
        with open(full, 'rb') as handle:
            self._send(200, handle.read(), content_type=content_type)

    def _runs(self):
        runs = self.service.refresh()
        self._json({
            'runs': [run.to_json() for run in runs],
            'device': str(self.service.device),
            'results_root': self.service.results_root,
            'loaded': self.service.cache.loaded(),
        })

    def _tokenize(self):
        payload = self._body()
        run_id = payload.get('run_id') or ''
        self._json(self.service.tokenize(run_id, payload.get('text') or ''))

    def _model(self):
        payload = self._body()
        run, ckpt, model = self.service.model(payload.get('run_id') or '',
                                              payload.get('kind') or 'best')
        self._json({'run_id': run.run_id, 'checkpoint_kind': ckpt.kind,
                    'model': model.describe()})

    # -- generation --------------------------------------------------------

    def _generate(self):
        payload = self._body()
        run, ckpt, model = self.service.model(payload.get('run_id') or '',
                                              payload.get('kind') or 'best')
        prompt = payload.get('prompt') or ''
        prompt_ids = model.encode(prompt)
        if not prompt_ids:
            return self._error(400, 'the prompt is empty after tokenization')

        options = {
            'max_new_tokens': int(payload.get('max_new_tokens') or 48),
            'temperature': float(payload.get('temperature', 0.9)),
            'top_k': int(payload.get('top_k') or 0),
            'top_p': float(payload.get('top_p', 1.0)),
            'suppress_unk': bool(payload.get('suppress_unk', True)),
            'stop_at_eol': bool(payload.get('stop_at_eol', False)),
        }
        if payload.get('seed') not in (None, ''):
            options['seed'] = int(payload['seed'])
        options['max_new_tokens'] = max(1, min(options['max_new_tokens'], 512))

        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Connection', 'close')
        self.end_headers()

        broken = threading.Event()

        def emit(event):
            self.wfile.write((json.dumps(event) + '\n').encode('utf-8'))
            self.wfile.flush()

        context_ids = [int(i) for i in prompt_ids][-model.context_size:]
        # Whether a context id is <unk> is the tokenizer's own answer. The inspector used
        # to decide it with `id === 0`, which under GPT-2 BPE marks every "!" as missing
        # vocabulary. Ask the adapter rather than restating the word vocabulary's ids
        # here: this module should not know which id that is, only that the adapter does.
        adapter = model.tokenizer
        has_unk = bool(adapter.has_unk)
        with self.service.generation_lock:
            try:
                emit({
                    'type': 'start',
                    'run_id': run.run_id,
                    'checkpoint_kind': ckpt.kind,
                    'prompt_tokens': len(prompt_ids),
                    'has_unk': has_unk,
                    'context': [{'id': i, 'token': t,
                                 'unk': has_unk and i == adapter.unk_id}
                                for i, t in zip(context_ids,
                                                model.token_strings(context_ids))],
                    'top_next': model.top_next(prompt_ids, k=8,
                                               exclude_unk=options['suppress_unk']),
                })
                produced = 0
                for event in model.generate(prompt_ids, should_stop=broken.is_set,
                                            **options):
                    emit(dict(event, type='token'))
                    produced += 1
                emit({'type': 'done', 'tokens': produced})
            except (BrokenPipeError, ConnectionResetError):
                # Stop sampling the moment the reader disappears -- otherwise an
                # abandoned tab keeps a GPU busy for the rest of its token budget.
                broken.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m lut_torch.chat_server',
        description='Web chat interface for trained LUT language models.')
    p.add_argument('--host', default='0.0.0.0',
                   help='Bind address. The default is reachable through an SSH tunnel '
                        'to a compute node; use 127.0.0.1 to keep it on one host.')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--results-root', default=None,
                   help='Directory holding <sweep>/<run>/config.txt (default: the '
                        'repository\'s results/)')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--cache-size', type=int, default=2,
                   help='How many checkpoints stay resident (the baseline LM arm is '
                        '1.1 GB each)')
    p.add_argument('--task', default=TASK_LM, choices=[TASK_LM, run_index.TASK_POS],
                   help='Which runs to offer. The chat only makes sense for LM.')
    return p


def serve(argv=None):
    args = build_parser().parse_args(argv)
    from .inference import REPO_ROOT

    results_root = args.results_root or os.path.join(REPO_ROOT, 'results')
    if not os.path.isdir(results_root):
        sys.exit(f"results root not found: {results_root}")

    service = ChatService(results_root, device=args.device, cache_size=args.cache_size,
                          task=args.task)
    runs = service.refresh()
    ChatHandler.service = service

    httpd = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    httpd.daemon_threads = True
    host = socket.gethostname()
    print('=' * 68)
    print(f"  LUT chat server")
    print(f"  results root : {results_root}")
    print(f"  {args.task} runs     : {len(runs)} with a checkpoint on disk")
    for run in runs[:12]:
        kinds = '/'.join(c.kind for c in run.checkpoints)
        loss = run.metrics.best_loss
        print(f"     {run.sweep}/{run.name}  [{kinds}]"
              + (f"  best val loss {loss:.4f}" if loss is not None else ''))
    print(f"  device       : {service.device}")
    if args.device == 'auto' and service.device.type == 'cpu':
        # A silent CPU fallback on an allocated GPU node costs roughly 20x on generation
        # and looks like nothing at all. `gg:g0:1` allocations here do sometimes come up
        # without a usable CUDA runtime, which is why run_files/gpu_check.sh exists.
        allocated = os.environ.get('CUDA_VISIBLE_DEVICES')
        print(f"  !! --device auto resolved to CPU: torch.cuda.is_available() is False")
        print(f"     CUDA_VISIBLE_DEVICES={allocated!r}"
              + ('  (SLURM did allocate a GPU, so this is the node or the runtime)'
                 if allocated else '  (no GPU allocated to this job)'))
        print(f"     Generation will be seconds per token. Diagnose with:"
              f"  srun -n1 --gres=gg:g0:1 --mem=4g -c1 --time=0-0:05 "
              f"run_files/gpu_check.sh")
    print(f"  listening on : http://{args.host}:{args.port}")
    if args.host == '0.0.0.0':
        print(f"  from a login node: ssh -N -L {args.port}:{host}:{args.port} "
              f"<user>@<login-host>   then open http://localhost:{args.port}")
    print('=' * 68)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping')
    finally:
        httpd.server_close()


if __name__ == '__main__':
    serve()
