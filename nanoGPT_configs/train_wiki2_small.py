# nanoGPT config — small model matching LUT default params
dataset = 'wiki2'
out_dir = 'out-wiki2-small'

# model — match LUT defaults where sensible
n_layer    = 6
n_head     = 4
n_embd     = 128      # larger than LUT's 32 (transformers need more width to work)
block_size = 32       # same context_size as LUT
dropout    = 0.1
bias       = False

# vocab — wiki2.bin is GPT-2 tokenized, no meta.pkl
vocab_size      = 50257
meta_vocab_size = None

# training
batch_size     = 16
max_iters      = 100000
learning_rate  = 3e-4
min_lr         = 3e-5
warmup_iters   = 4000
lr_decay_iters = 100000
eval_interval  = 1000
eval_iters     = 200
log_interval   = 100

# system
device  = 'mps'   # Apple Silicon GPU; use 'cpu' or 'cuda' if needed
compile = False   # set True if PyTorch 2.0+ and want speed
