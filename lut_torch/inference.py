"""Loading a trained checkpoint and sampling text from it.

This is the torch half of the chat interface; :mod:`lut_torch.run_index` is the half that
needs no torch. Nothing here is used by training.

Three things about this model shape the whole file:

* **The context is fixed at ``cfg.context_size`` and there is no KV cache.** The model
  maps ``(B, CS)`` tokens to ``(B, CS, V)`` logits in one shot, so generating a token is
  a full forward over a window rather than an incremental step. Generation is therefore
  a sliding window: append the sampled token, re-run the last ``CS``.

* **Attention is causal** (``comparisons.causal_pair_indices`` emits only ``j < i``), so a
  prompt shorter than the window goes at positions ``0..n-1`` and the prediction is read
  from position ``n-1``. Whatever fills positions ``n..CS-1`` cannot reach it --
  ``test_torch_chat.py`` asserts that rather than trusting it. Left-padding instead would
  be wrong twice over: the positional encoding is absolute, and position 0 receives no
  attention output at all.

* **Dropout is alive in ``train()`` mode** (``LutLayer._residual_dropout``), so the model
  must be put in ``eval()`` or every reply is sampled from a different perturbed model.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass

import torch

from .config import LutConfig
from .modules import LutModel
from .run_index import ENGINE_LUT, ENGINE_NANOGPT
from .stacking import looks_unstacked, stack_state_dict

# Kept in sync with data.py / word_tokenizer.py.
UNK_TOKEN_ID = 0
EOL_TOKEN_ID = 1
EOL = '<eol>'

# Written by gpt_baseline_train.save_checkpoint. LUT checkpoints carry no 'kind' at all,
# so its absence is what identifies them.
CHECKPOINT_KIND_NANOGPT = 'nanogpt_lm'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Checkpoint -> model
# ---------------------------------------------------------------------------

def config_from_checkpoint(ckpt):
    """Rebuild the :class:`LutConfig` a checkpoint was trained with.

    ``LutConfig.to_dict`` is ``asdict``, so it carries the four ``init=False`` derived
    fields (``mha_head_dim`` and friends) that ``__post_init__`` recomputes. Feeding
    those back to the constructor is a ``TypeError``, so filter to the init fields --
    the same thing ``LutConfig.from_namespace`` does to the trainer's argparse namespace.
    """
    stored = dict(ckpt['config'])
    init_fields = {name for name, spec in LutConfig.__dataclass_fields__.items()
                   if spec.init}
    unknown = sorted(set(stored) - init_fields - {
        f for f, spec in LutConfig.__dataclass_fields__.items() if not spec.init})
    if unknown:
        # A checkpoint from a newer config than this working tree. Say so rather than
        # silently building a differently shaped model.
        raise ValueError(f"checkpoint config has fields this LutConfig does not know: "
                         f"{', '.join(unknown)}")
    return LutConfig(**{k: v for k, v in stored.items() if k in init_fields})


def load_checkpoint_file(path, map_location='cpu', retries=3, retry_delay=2.0,
                         mmap=True):
    """``torch.load`` with retries, because these files are written under us.

    The sweeps write ``model_best.pt`` on every validation improvement, so selecting a
    run whose job is still going can land mid-write and raise from the zip reader. That
    is transient: retry, and let the caller fall back to the other copy.

    ``mmap=True`` maps the tensor storages instead of reading them into anonymous memory,
    which is most of what makes loading these checkpoints affordable -- see
    :func:`load_model`. It needs the zipfile serialisation ``torch.save`` has produced by
    default since 1.6, so it is attempted and not required.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return torch.load(path, map_location=map_location, weights_only=False,
                              mmap=mmap)
        except Exception as exc:                      # noqa: BLE001 - torch raises broadly
            last_error = exc
            if mmap:
                # An old-format or otherwise unmappable file: fall back rather than
                # spending the retries on something that cannot work.
                return load_checkpoint_file(path, map_location, retries, retry_delay,
                                            mmap=False)
            if attempt + 1 < retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"could not read {path}: {last_error}")


def resolve_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but torch.cuda.is_available() "
                           f"is False")
    return device


def resolve_repo_path(path):
    """Checkpoints record data paths as the run script passed them -- relative to the
    repository root, which is not necessarily the server's working directory."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_tokenizer(tokenizer_name, vocab_file):
    """The encoder the run was trained with. ``word`` needs its exact vocabulary file."""
    if tokenizer_name == 'word':
        from word_tokenizer import WordTokenizer
        resolved = resolve_repo_path(vocab_file)
        if not resolved or not os.path.exists(resolved):
            raise FileNotFoundError(
                f"vocabulary file {vocab_file!r} not found (looked at {resolved!r}); the "
                f"model cannot be tokenized for without it")
        return WordTokenizer(resolved)
    import tiktoken
    return tiktoken.get_encoding(tokenizer_name)


@dataclass
class GptSpec:
    """The handful of fields :class:`LutChatModel` needs from a nanoGPT checkpoint.

    Deliberately not a ``GPTConfig``: this stands in for ``LutConfig`` on the chat side,
    so it carries the two names that code reads (``context_size``, ``vocab_size_*``) plus
    the shape, and nothing else. Keeping it separate means the chat client never has to
    know which engine it is holding except where it genuinely differs -- the description
    panel.
    """

    context_size: int
    vocab_size_input: int
    vocab_size_output: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True


def load_nanogpt_model(ckpt, device='cpu'):
    """Rebuild the GPT baseline from its checkpoint.

    Reuses ``gpt_baseline_train``'s own ``load_nanogpt`` and ``FullLogits`` rather than
    re-deriving them. ``FullLogits`` matters: ``GPT.forward`` keeps only the last
    position when it is given no targets, and this chat client reads position ``n-1`` of
    a padded window, so the sampling shortcut would hand back the wrong row.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from gpt_baseline_train import FullLogits, load_nanogpt

    margs = dict(ckpt['model_args'])
    nanogpt = load_nanogpt()
    gpt = nanogpt.GPT(nanogpt.GPTConfig(**margs))

    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        # A checkpoint saved from the torch.compile wrapper rather than the module.
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    # Plain copy, not assign=True: nanoGPT ties lm_head.weight to wte.weight, and the
    # state dict holds both keys. Copying preserves the shared tensor; assigning would
    # quietly split it into two.
    gpt.load_state_dict(state)

    model = FullLogits(gpt)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    cfg = GptSpec(context_size=margs['block_size'],
                  vocab_size_input=margs['vocab_size'],
                  vocab_size_output=margs['vocab_size'],
                  n_layer=margs['n_layer'], n_head=margs['n_head'],
                  n_embd=margs['n_embd'], dropout=margs.get('dropout', 0.0),
                  bias=margs.get('bias', True))
    return model, cfg, dict(ckpt.get('args') or {}), int(ckpt.get('step', 0))


def load_model(path, device='cpu'):
    """``(model, cfg, args, step)`` for a checkpoint, in ``eval`` mode on ``device``.

    Dispatches on the checkpoint's own ``kind``. The GPT baseline stamps
    ``nanogpt_lm``; LUT checkpoints predate the field and have none, so absence means
    LUT. Both come back with the same contract -- a module mapping ``(B, T)`` ids to
    ``(B, T, V)`` logits, in eval mode, with grad off.

    Loading is written for peak memory, not elegance, because the uncoded LM arm is a
    1.1 GB checkpoint whose output table alone is 944 MB. Done naively -- read the whole
    file, construct a model that allocates its own tables, copy one into the other --
    peak is about three times the model, which is enough to get the process killed on a
    modest host. Two changes remove two of those three copies:

    * ``mmap`` in :func:`load_checkpoint_file`, so the tensors are file-backed pages
      rather than a second heap allocation, and
    * ``assign=True`` here, which hands those tensors to the modules instead of copying
      into the freshly built ones, letting the constructor's zero tables be freed.

    Every buffer the model needs -- comparison indices, tree maps, the frozen token
    embedding -- is registered, so the checkpoint really does carry all of the state and
    nothing survives from construction except the shapes.
    """
    ckpt = load_checkpoint_file(path, map_location='cpu')
    if ckpt.get('kind') == CHECKPOINT_KIND_NANOGPT:
        return load_nanogpt_model(ckpt, device=device)
    cfg = config_from_checkpoint(ckpt)
    model = LutModel(cfg)
    state = ckpt['model']
    if looks_unstacked(state, model):
        # Pre-``--stack-heads`` checkpoints hold one table per head; fold them into this
        # model's slabs, exactly as train.load_checkpoint does when resuming.
        state = stack_state_dict(state, model)
    model.load_state_dict(state, assign=True)
    args, step = dict(ckpt.get('args') or {}), int(ckpt.get('step', 0))
    # Drop the optimizer and RNG state (and, for a non-mmap fallback, the tensors
    # themselves) before moving anything to the device.
    del ckpt, state
    model.to(device)
    # Dropout is live in train() mode, so this is not a formality: without it every
    # reply would be sampled from a differently perturbed model, silently.
    model.eval()
    model.requires_grad_(False)
    return model, cfg, args, step


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def encode_prompt(tokenizer, text):
    """Encode ``text`` as a prefix to be continued, not as a finished document.

    ``WordTokenizer._split`` appends ``<eol>`` after every non-empty line *including the
    last*, because it was written to encode whole corpora, where the final newline is
    real. For a chat prompt it is not: it tells the model the sentence is over and the
    next token starts a fresh line, and in WikiText-2 a fresh line is very often a
    heading. Measured on ``learnedD14_bias``, the prompt "The first game in the series
    was released in" put ``<eol>`` at 20% -- the single most likely continuation -- and
    greedy decoding answered with ``= = =``.

    So the trailing ``<eol>`` is kept only when the user actually ended with a newline.
    The encoder emits exactly one either way (``"abc"`` and ``"abc\\n"`` both give
    ``[abc, <eol>]``), so the raw text is the only thing that can tell them apart.
    """
    ids = [int(i) for i in tokenizer.encode(text)]
    if ids and ids[-1] == EOL_TOKEN_ID and not text.endswith('\n'):
        ids.pop()
    return ids


class IncrementalDecoder:
    """Stream ``WordTokenizer.decode`` one token at a time.

    Reproduces that method's spacing rule exactly -- ``<eol>`` becomes a newline, a
    single non-alphanumeric character attaches to what precedes it, everything else takes
    a leading space unless it opens the text or follows a newline -- so that concatenating
    the streamed pieces equals decoding the whole sequence in one call.
    ``test_incremental_decode_matches_whole`` pins that equality.
    """

    def __init__(self, tokenizer, previous_token=None):
        self.tokenizer = tokenizer
        # None means "nothing has been emitted yet", which is what suppresses the leading
        # space on the very first token. A continuation seeds this from the prompt.
        self.previous = previous_token

    @classmethod
    def continuing(cls, tokenizer, prompt_ids):
        """A decoder whose next piece is spaced as if it followed ``prompt_ids``."""
        if not len(prompt_ids):
            return cls(tokenizer)
        last = tokenizer.tokens_from_ids([int(prompt_ids[-1])])[0]
        return cls(tokenizer, '\n' if last == EOL else last)

    def feed(self, token_id):
        token = self.tokenizer.tokens_from_ids([int(token_id)])[0]
        if token == EOL:
            piece = '\n'
        elif self.previous is None or self.previous == '\n':
            piece = token
        elif len(token) == 1 and not token.isalnum():
            piece = token
        else:
            piece = ' ' + token
        self.previous = piece
        return piece


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0, banned_ids=()):
    """Apply the usual sampling restrictions to a 1-D logit vector.

    Order matters: bans first (a banned token must not consume top-k budget), then
    temperature, then top-k, then nucleus. ``temperature <= 0`` means greedy and is
    handled by the caller, not here.
    """
    logits = logits.clone().float()
    for token_id in banned_ids:
        logits[token_id] = float('-inf')
    if temperature and temperature != 1.0:
        logits = logits / temperature
    if top_k and 0 < top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float('-inf')
    if 0.0 < top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Keep everything up to and including the token that crosses top_p, so the
        # nucleus is never empty even when one token already exceeds the mass.
        drop = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        drop[0] = False
        logits[order[drop]] = float('-inf')
    return logits


class LutChatModel:
    """One loaded checkpoint plus its tokenizer, wrapped for interactive use."""

    def __init__(self, model, cfg, args, step, tokenizer, device, checkpoint_path,
                 engine=ENGINE_LUT):
        self.model = model
        self.cfg = cfg
        # Which network this is. Everything between tokenization and sampling is
        # identical for both -- same vocabulary, same fixed window, same (B,T,V) logits
        # -- so this only steers `describe`.
        self.engine = engine
        self.args = args
        self.step = step
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.context_size = cfg.context_size
        # Two tabs sharing one model would interleave into the same module; every entry
        # point that touches `model` takes this.
        self.lock = threading.RLock()

    # -- tokenization ------------------------------------------------------

    def encode(self, text):
        return encode_prompt(self.tokenizer, text)

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids])

    def token_strings(self, ids):
        return self.tokenizer.tokens_from_ids([int(i) for i in ids])

    # -- forward -----------------------------------------------------------

    def _window(self, ids):
        """``(tokens, read_position)``: the last ``CS`` ids placed at positions 0.., and
        the position whose logits predict the next token."""
        window = [int(i) for i in ids][-self.context_size:]
        if not window:
            raise ValueError('need at least one token to predict from')
        padded = window + [UNK_TOKEN_ID] * (self.context_size - len(window))
        tokens = torch.tensor([padded], dtype=torch.int64, device=self.device)
        return tokens, len(window) - 1

    @torch.no_grad()
    def next_logits(self, ids):
        """Logits over the next token given ``ids`` (only the last ``CS`` matter)."""
        tokens, read_at = self._window(ids)
        logits = self.model(tokens)
        return logits[0, read_at].float()

    @torch.no_grad()
    def top_next(self, ids, k=10, exclude_unk=False):
        logits = self.next_logits(ids)
        if exclude_unk:
            logits = logits.clone()
            logits[UNK_TOKEN_ID] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, min(k, probs.numel()))
        tokens = self.token_strings(indices.tolist())
        return [{'id': int(i), 'token': t, 'prob': float(p)}
                for i, t, p in zip(indices.tolist(), tokens, values.tolist())]

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=48, temperature=0.9, top_k=40,
                 top_p=0.95, suppress_unk=True, stop_at_eol=False, seed=None,
                 should_stop=None):
        """Yield one dict per sampled token: ``id``, ``token``, ``text``, ``prob``.

        ``text`` is the incremental rendering, so a client can append it directly.
        ``should_stop`` is polled between tokens, which is how the server cancels a
        generation whose browser has gone away.
        """
        ids = [int(i) for i in prompt_ids]
        if not ids:
            return
        generator = None
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
        banned = (UNK_TOKEN_ID,) if suppress_unk else ()
        decoder = IncrementalDecoder.continuing(self.tokenizer, ids)

        for _ in range(max_new_tokens):
            if should_stop is not None and should_stop():
                return
            logits = self.next_logits(ids)
            filtered = filter_logits(logits, temperature=temperature, top_k=top_k,
                                     top_p=top_p, banned_ids=banned)
            probs = torch.softmax(filtered, dim=-1)
            if not temperature or temperature <= 0:
                next_id = int(torch.argmax(filtered))
            else:
                # Sampling happens on the CPU so a --seed reproduces a reply regardless
                # of which device the model is on.
                next_id = int(torch.multinomial(probs.cpu(), 1,
                                                generator=generator).item())
            probability = float(probs[next_id])
            ids.append(next_id)
            token = self.token_strings([next_id])[0]
            yield {'id': next_id, 'token': token, 'text': decoder.feed(next_id),
                   'prob': probability}
            if stop_at_eol and next_id == EOL_TOKEN_ID:
                return

    # -- description -------------------------------------------------------

    def describe(self):
        """What the UI shows about the loaded model.

        The common half is whatever both engines have; the rest is per-engine, because
        "64 trees x 2^8 rows with a learned:448 code" and "512-wide, 8 heads" are not the
        same quantity wearing different labels and should not be forced into one schema.
        """
        cfg = self.cfg
        common = {
            'engine': self.engine,
            'checkpoint': self.checkpoint_path,
            'step': self.step,
            'device': str(self.device),
            'context_size': cfg.context_size,
            'vocab_size': cfg.vocab_size_output,
            'num_layers': cfg.n_layer if self.engine == ENGINE_NANOGPT else cfg.num_layers,
            'num_heads': cfg.n_head if self.engine == ENGINE_NANOGPT else cfg.num_heads,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'tokenizer': self.args.get('tokenizer', 'word'),
            'vocab_file': self.args.get('vocab_file', ''),
        }
        if self.engine == ENGINE_NANOGPT:
            common.update({
                'n_embd': cfg.n_embd,
                'dropout': cfg.dropout,
                'n_t': None, 'n_c': None, 'output_code': None, 'table_entries': None,
            })
        else:
            common.update({
                'n_t': cfg.n_t,
                'n_c': cfg.n_c,
                'output_code': (None if cfg.output_code_dim is None
                                else f"{cfg.output_code}:{cfg.output_code_dim}"
                                     f"{'+bias' if cfg.output_code_bias else ''}"),
                'table_entries': sum(p.numel()
                                     for p in self.model.table_parameters()),
                'n_embd': None, 'dropout': None,
            })
        return common


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class ModelCache:
    """Keeps the few most recently used checkpoints resident.

    The baseline LM arm is 1.1 GB on disk and about the same in memory, so the default
    of two is a deliberate ceiling rather than an arbitrary one: switching back and forth
    between two arms stays instant, a third eviction costs a reload instead of the host
    swapping.
    """

    def __init__(self, device='cpu', capacity=2):
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.capacity = max(1, capacity)
        self._entries = {}                   # path -> LutChatModel
        self._order = []                     # least recent first
        self._lock = threading.Lock()

    def get(self, path, tokenizer_name=None, vocab_file=None):
        path = os.path.abspath(path)
        with self._lock:
            if path in self._entries:
                self._order.remove(path)
                self._order.append(path)
                return self._entries[path]

        model, cfg, args, step = load_model(path, device=self.device)
        enc = load_tokenizer(tokenizer_name or args.get('tokenizer', 'word'),
                             vocab_file or args.get('vocab_file', ''))
        vocab_n = getattr(enc, 'n_vocab', None)
        if vocab_n is not None and vocab_n != cfg.vocab_size_input:
            raise ValueError(
                f"vocabulary has {vocab_n} entries but the model was trained with "
                f"vocab_size_input={cfg.vocab_size_input}; wrong --vocab-file")
        engine = ENGINE_NANOGPT if isinstance(cfg, GptSpec) else ENGINE_LUT
        wrapper = LutChatModel(model, cfg, args, step, enc, self.device, path,
                               engine=engine)

        with self._lock:
            self._entries[path] = wrapper
            self._order.append(path)
            while len(self._order) > self.capacity:
                evicted = self._order.pop(0)
                self._entries.pop(evicted, None)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        return wrapper

    def loaded(self):
        with self._lock:
            return list(self._order)
