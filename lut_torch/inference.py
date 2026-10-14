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

A fourth thing shapes it since the WikiText-103 runs: **there are two tokenizers**, and
nothing about the word one generalises -- not id 0, not id 1, not ``tokens_from_ids``,
not the spacing rules. :class:`TokenizerAdapter` is the single place that difference
lives; everything below it asks the adapter rather than assuming.
"""

import codecs
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

# **Word-tokenizer ids**, kept in sync with data.py / word_tokenizer.py -- they are the
# first two entries of a vocabulary file, not universal facts. Under GPT-2 BPE id 0 is
# ``!`` and id 1 is ``"``, both ordinary tokens, so acting on these under BPE bans an
# exclamation mark and truncates any prompt ending in a quotation mark. Nothing outside
# the word tokenizer's own adapter may use them; they stay module-level because other
# modules import ``UNK_TOKEN_ID``.
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
    """The raw encoder the run was trained with.

    ``word`` needs its exact vocabulary file; a BPE name (``gpt2``) carries its own
    vocabulary, so ``vocab_file`` is empty for those runs and is ignored here rather than
    demanded. Returns the encoder itself -- :func:`make_tokenizer_adapter` is what wraps
    it in the uniform interface the chat path uses.
    """
    if not tokenizer_name:
        tokenizer_name = 'word'
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

def encode_bpe_prompt(enc, text):
    """Encode ``text`` with a tiktoken encoder, verbatim.

    No trimming: see :func:`encode_prompt`. BPE has no end-of-line token to strip, and a
    newline is carried inside ordinary pieces.

    ``disallowed_special=()`` turns off tiktoken's default, which *raises* when the text
    contains a special-token spelling such as ``<|endoftext|>``. A chat box is arbitrary
    user text, so that default would turn typing ``<|endoftext|>`` into a server error;
    encoding it as the ordinary bytes it looks like is the only sensible reading here.
    """
    return [int(i) for i in enc.encode(text, disallowed_special=())]


def encode_prompt(tokenizer, text):
    """Encode ``text`` as a prefix to be continued, not as a finished document.

    **Word tokenizer only** -- the trim below is wrong for BPE, see the end of this
    docstring. ``WordTokenizer._split`` appends ``<eol>`` after every non-empty line
    *including the last*, because it was written to encode whole corpora, where the final
    newline is real. For a chat prompt it is not: it tells the model the sentence is over
    and the next token starts a fresh line, and in WikiText-2 a fresh line is very often a
    heading. Measured on ``learnedD14_bias``, the prompt "The first game in the series
    was released in" put ``<eol>`` at 20% -- the single most likely continuation -- and
    greedy decoding answered with ``= = =``.

    So the trailing ``<eol>`` is kept only when the user actually ended with a newline.
    The encoder emits exactly one either way (``"abc"`` and ``"abc\\n"`` both give
    ``[abc, <eol>]``), so the raw text is the only thing that can tell them apart.

    This takes an adapter, a ``WordTokenizer`` or a tiktoken encoder, and dispatches,
    because callers (and ``test_torch_chat``) hand it a raw encoder. Under BPE the trim
    must not happen: id 1 there is ``"``, so trimming would delete the closing quotation
    mark of every prompt that ends in one.
    """
    if isinstance(tokenizer, TokenizerAdapter):
        return tokenizer.encode_prompt(text)
    if not hasattr(tokenizer, 'tokens_from_ids'):
        # Duck-typed as "not the word tokenizer": tiktoken has no such method.
        return encode_bpe_prompt(tokenizer, text)
    ids = [int(i) for i in tokenizer.encode(text)]
    if ids and ids[-1] == EOL_TOKEN_ID and not text.endswith('\n'):
        ids.pop()
    return ids


class IncrementalDecoder:
    """Stream ``WordTokenizer.decode`` one token at a time. **Word tokenizer only** --
    BPE streams through :class:`ByteStreamDecoder`, which concatenates instead.

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


def token_bytes(enc, token_id):
    """The raw bytes of one BPE id, or ``None`` for an id the encoder does not assign.

    A model may be built with a padded ``vocab_size_input`` above the encoder's
    ``n_vocab`` (GPT-2's is an awkward 50,257, and training pads it). Nothing bans the
    extra ids, so sampling can emit one, and tiktoken raises ``KeyError`` for it rather
    than returning empty bytes. Every display path has to survive that.
    """
    try:
        return enc.decode_single_token_bytes(int(token_id))
    except KeyError:
        return None


class ByteStreamDecoder:
    """Stream a BPE sequence one token at a time, buffering by bytes.

    Two differences from :class:`IncrementalDecoder`, both from the same fact -- a BPE
    piece is a byte string that already carries its own leading space:

    * **No spacing rules.** Rendering is plain concatenation; the word tokenizer's
      "prepend a space unless it is punctuation" would insert spaces that are not there.
    * **No prompt seeding.** A continuation is spaced correctly without knowing what
      preceded it, which is exactly what ``IncrementalDecoder.continuing`` exists for.

    A single UTF-8 character can span two tokens, so decoding each token's bytes on their
    own would turn one character into two replacement characters. ``codecs``' incremental
    decoder holds an incomplete sequence back until the byte completing it arrives, which
    is what makes the concatenated stream equal ``decode()`` of the whole sequence. A
    sequence left incomplete at the end stays buffered and emits nothing -- correct, and
    also what a reply cut off mid-character should look like.
    """

    def __init__(self, enc):
        self.enc = enc
        self._decoder = codecs.getincrementaldecoder('utf-8')('replace')

    def feed(self, token_id):
        return self._decoder.decode(token_bytes(self.enc, token_id) or b'')


# ---------------------------------------------------------------------------
# Tokenizer adapters
# ---------------------------------------------------------------------------

class TokenizerAdapter:
    """The tokenizer facts the chat path needs, for either tokenizer.

    The chat stack was written against ``word_tokenizer.WordTokenizer`` and assumed four
    of its properties at every call site: that id 0 is ``<unk>``, that id 1 is ``<eol>``,
    that ``tokens_from_ids`` exists, and that displaying a token means re-deriving the
    spacing the tokenizer discarded. Not one of the four survives GPT-2 BPE -- id 0 is
    ``!``, id 1 is ``"``, tiktoken has no ``tokens_from_ids``, and a BPE piece carries its
    own leading space. Two of those fail *silently* (banning ``!`` from every reply;
    deleting the final character of any prompt that ends in a quotation mark), which is
    why they are behind this interface rather than behind an ``if tokenizer == 'word'``
    repeated at each site.

    Implementations provide:

    ``name``
        the ``--tokenizer`` value the run was trained with.
    ``n_vocab``
        how many ids the encoder can produce. A model may have been built *wider* than
        this (see :meth:`ModelCache.get`); it may not have been built narrower.
    ``has_unk`` / ``unk_id``
        whether an out-of-vocabulary id exists at all, and which one. False/``None`` for
        BPE, which encodes anything -- so ``exclude_unk`` and ``suppress_unk`` have
        nothing to act on there and must become no-ops rather than banning id 0.
    ``pad_id``
        the id that fills positions ``n..CS-1`` of a short window. **Any id is safe**:
        attention is causal, so nothing after position ``n-1`` can reach the position the
        prediction is read from, and ``test_short_prompt_ignores_padding`` checks that
        against the model rather than trusting it.
    """

    name = ''
    has_unk = False
    unk_id = None
    pad_id = 0

    @property
    def n_vocab(self):
        raise NotImplementedError

    def encode_prompt(self, text):
        """Ids for ``text`` treated as a prefix to be continued."""
        raise NotImplementedError

    def pieces(self, ids):
        """One display string per id, for the token strip and the top-k panel."""
        raise NotImplementedError

    def decode(self, ids):
        """The text of a whole sequence."""
        raise NotImplementedError

    def is_newline(self, token_id):
        """Does this token end a line? (``--stop-at-eol`` and nothing else.)"""
        raise NotImplementedError

    def stream(self, prompt_ids=()):
        """An incremental decoder with ``.feed(token_id) -> str``, seeded so that its
        first piece continues ``prompt_ids`` correctly."""
        raise NotImplementedError


class WordTokenizerAdapter(TokenizerAdapter):
    """``word_tokenizer.WordTokenizer``: the behaviour this file had before adapters."""

    name = 'word'
    has_unk = True
    unk_id = UNK_TOKEN_ID
    # Padding with <unk> is what this file always did; it is arbitrary (see pad_id above)
    # and kept only so word runs are byte-for-byte unchanged.
    pad_id = UNK_TOKEN_ID

    def __init__(self, enc):
        self.enc = enc

    @property
    def n_vocab(self):
        return self.enc.n_vocab

    def encode_prompt(self, text):
        return encode_prompt(self.enc, text)

    def pieces(self, ids):
        return self.enc.tokens_from_ids([int(i) for i in ids])

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])

    def is_newline(self, token_id):
        return int(token_id) == EOL_TOKEN_ID

    def stream(self, prompt_ids=()):
        return IncrementalDecoder.continuing(self.enc, list(prompt_ids))


class BpeTokenizerAdapter(TokenizerAdapter):
    """A tiktoken encoder (``gpt2``), whose pieces are bytes rather than words."""

    # Any id would do; 0 is "!" here and is as invisible as anything else behind causal
    # attention.
    pad_id = 0

    def __init__(self, enc, name='gpt2'):
        self.enc = enc
        self.name = name

    @property
    def n_vocab(self):
        return self.enc.n_vocab

    def encode_prompt(self, text):
        return encode_bpe_prompt(self.enc, text)

    def pieces(self, ids):
        """Per-token strings. A character split across two tokens shows as replacement
        characters in *this* view and not in the stream, which is inherent to showing a
        byte-level tokenizer's tokens individually, not a decoding bug."""
        out = []
        for i in ids:
            raw = token_bytes(self.enc, i)
            # An id past the encoder's table has no bytes at all; show the number so the
            # strip stays aligned with `ids` instead of silently losing an entry.
            out.append(raw.decode('utf-8', 'replace') if raw is not None
                       else f'<{int(i)}>')
        return out

    def decode(self, ids):
        # Byte-join rather than enc.decode, which raises on a padded-vocabulary id. The
        # result is identical otherwise: tiktoken's decode is this with errors='replace'.
        raw = b''.join(token_bytes(self.enc, i) or b'' for i in ids)
        return raw.decode('utf-8', 'replace')

    def is_newline(self, token_id):
        # There is no <eol>: a newline is a byte inside an ordinary piece, and plenty of
        # GPT-2 tokens carry one (e.g. b'.\n').
        raw = token_bytes(self.enc, token_id)
        return raw is not None and b'\n' in raw

    def stream(self, prompt_ids=()):
        return ByteStreamDecoder(self.enc)


def make_tokenizer_adapter(tokenizer_name, vocab_file=''):
    """The adapter for a run, from the two fields its ``config.txt`` records.

    ``vocab_file`` is empty for BPE runs, which is why it is not required there.
    """
    name = tokenizer_name or 'word'
    enc = load_tokenizer(name, vocab_file)
    if name == 'word':
        return WordTokenizerAdapter(enc)
    return BpeTokenizerAdapter(enc, name)


def as_tokenizer_adapter(tokenizer, name=None):
    """Wrap a raw encoder, or pass an adapter through.

    Callers that already hold an encoder (the server's tokenizer cache, the tests' stub)
    should not have to reload one. ``tokens_from_ids`` is the discriminator: it is the
    word tokenizer's, and tiktoken has nothing like it.
    """
    if isinstance(tokenizer, TokenizerAdapter):
        return tokenizer
    if hasattr(tokenizer, 'tokens_from_ids'):
        return WordTokenizerAdapter(tokenizer)
    if name in (None, '', 'word'):
        # The object and the recorded name disagree; the object is the one that will do
        # the decoding, so take its own name (tiktoken encoders carry one).
        name = getattr(tokenizer, 'name', 'gpt2')
    return BpeTokenizerAdapter(tokenizer, name)


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
    """One loaded checkpoint plus its tokenizer adapter, wrapped for interactive use."""

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
        # Always an adapter: every id below (padding, the <unk> ban, the end-of-line
        # stop) means something different under BPE, and nothing here may decide that
        # for itself. A raw encoder is accepted and wrapped for callers that hold one.
        self.tokenizer = as_tokenizer_adapter(tokenizer, args.get('tokenizer'))
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.context_size = cfg.context_size
        # Two tabs sharing one model would interleave into the same module; every entry
        # point that touches `model` takes this.
        self.lock = threading.RLock()

    # -- tokenization ------------------------------------------------------

    def encode(self, text):
        return self.tokenizer.encode_prompt(text)

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids])

    def token_strings(self, ids):
        return self.tokenizer.pieces([int(i) for i in ids])

    # -- forward -----------------------------------------------------------

    def _window(self, ids):
        """``(tokens, read_position)``: the last ``CS`` ids placed at positions 0.., and
        the position whose logits predict the next token."""
        window = [int(i) for i in ids][-self.context_size:]
        if not window:
            raise ValueError('need at least one token to predict from')
        padded = window + [self.tokenizer.pad_id] * (self.context_size - len(window))
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
        # No-op without an <unk>: under BPE there is no out-of-vocabulary id to exclude,
        # and acting on the request anyway would blank out "!".
        if exclude_unk and self.tokenizer.has_unk:
            logits = logits.clone()
            logits[self.tokenizer.unk_id] = float('-inf')
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
        # Same no-op as in top_next: `suppress_unk` is on by default because <unk> is
        # wiki2's most frequent token, but BPE has no <unk> and id 0 there is "!".
        banned = ((self.tokenizer.unk_id,)
                  if suppress_unk and self.tokenizer.has_unk else ())
        decoder = self.tokenizer.stream(ids)

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
            if stop_at_eol and self.tokenizer.is_newline(next_id):
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
            # From the adapter, not from `args`: the adapter is what the decoding path
            # actually used, and the UI changes behaviour on `has_unk` (no <unk> toggle,
            # no <unk> highlighting in the token strip when there is no such token).
            'tokenizer': self.tokenizer.name,
            'has_unk': self.tokenizer.has_unk,
            # Only when it is the file that was actually read. A BPE run's `args` still
            # carries argparse's default ('word_vocab.txt' -- checked on the WT103
            # checkpoints), which nothing opened and which the UI would otherwise show
            # beside 'gpt2' as if it had.
            'vocab_file': (self.args.get('vocab_file', '')
                           if isinstance(self.tokenizer, WordTokenizerAdapter) else ''),
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
        adapter = make_tokenizer_adapter(tokenizer_name or args.get('tokenizer', 'word'),
                                         vocab_file or args.get('vocab_file', ''))
        # Only *too small* is an error. A wider model is legitimate: GPT-2's awkward
        # 50,257 is routinely padded up for alignment (upstream nanoGPT uses 50,304;
        # these trainers pass enc.n_vocab straight through, but a checkpoint from
        # anywhere else may not), and the extra rows are simply never a target. Too small
        # is the real mismatch -- the tokenizer can emit an id the model has no row for
        # -- and the likeliest cause is the wrong vocabulary file, so say that.
        vocab_n = adapter.n_vocab
        if vocab_n > cfg.vocab_size_input:
            raise ValueError(
                f"tokenizer {adapter.name!r} has {vocab_n} entries but the model was "
                f"trained with vocab_size_input={cfg.vocab_size_input}; it cannot score "
                f"every token this tokenizer produces (wrong --vocab-file?)")
        engine = ENGINE_NANOGPT if isinstance(cfg, GptSpec) else ENGINE_LUT
        wrapper = LutChatModel(model, cfg, args, step, adapter, self.device, path,
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
