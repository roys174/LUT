"""SNN transformer architecture in pure Python/NumPy."""
"""Based on Spiking Manifesto (Izhikevich 2025)"""
"""The code strives for simplicity, not efficiency."""
"""Eugene Izhikevich, October 2025"""
"""Python port by Claude, February 2026"""

import argparse
import math
import random
import sys
import numpy as np
import tiktoken
from tqdm import tqdm


# Global learning rate (updated each step)
learning_rate = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def softmax(x, size, temperature):
    max_val = np.max(x[:size])
    x[:size] = np.exp((x[:size] - max_val) / temperature)
    s = np.sum(x[:size])
    x[:size] /= s


def vector_multiply(vector1, vector2, size):
    return np.dot(vector1[:size], vector2[:size])


def random_vector(size, scale):
    return scale * 2.0 * (np.random.rand(size).astype(np.float32) - 0.5)


def sample(probabilities, n):
    coin = random.random()
    cdf = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if coin < cdf:
            return i
    return n - 1


def fill_vector_with_random_integers(N, max_value):
    return np.array([random.randint(0, max_value - 1) for _ in range(N)], dtype=np.int32)


def fill_vector_with_random_integers_different_from_vector2(vector2, N, max_value):
    vector = np.zeros(N, dtype=np.int32)
    for i in range(N):
        while True:
            vector[i] = random.randint(0, max_value - 1)
            if vector[i] != vector2[i]:
                break
    return vector


def Up(x):
    """Surrogate gradient of the Heaviside step function."""
    sign_x = 1 if x > 0 else -1  # zero has "minus" sign
    return -0.5 * sign_x / (1 + abs(x)) / (1 + abs(x))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class LUT:
    __slots__ = ['y_dim', 'S', 'a_arr', 'b_arr', 'bitmasks']
    def __init__(self, n_t):
        self.y_dim = 0
        self.S = None
        self.a_arr = None
        self.b_arr = None
        self.bitmasks = None


class LUTcache:
    __slots__ = ['r_min', 'u_min', 'j']
    def __init__(self, n_t):
        self.r_min = np.zeros(n_t, dtype=np.int32)
        self.u_min = np.zeros(n_t, dtype=np.float32)
        self.j = np.zeros(n_t, dtype=np.int32)


class AttentionHead:
    __slots__ = ['V', 'V_cache', 'Positional_encoding', 'PE_cache']
    def __init__(self, args):
        N_T = args.n_t
        CONTEXT_SIZE = args.context_size
        POSITIONAL_DIM = args.positional_dim
        self.V = LUT(N_T)
        self.V_cache = [LUTcache(N_T) for _ in range(CONTEXT_SIZE)]
        self.Positional_encoding = np.zeros((CONTEXT_SIZE, N_T, POSITIONAL_DIM), dtype=np.float32)
        self.PE_cache = [LUTcache(N_T) for _ in range(CONTEXT_SIZE)]


class StandardOutputHead:
    __slots__ = ['unembedder', 'unembedder_vars', 'output', 'tokens']
    def __init__(self, args):
        N_T = args.n_t
        CONTEXT_SIZE = args.context_size
        VOCAB_SIZE = args.vocab_size
        self.unembedder = LUT(N_T)
        self.unembedder_vars = [LUTcache(N_T) for _ in range(CONTEXT_SIZE)]
        self.output = np.zeros((CONTEXT_SIZE, VOCAB_SIZE), dtype=np.float32)
        self.tokens = None  # set by load_targets

    def build(self, n_c, args):
        build_LUT(self.unembedder, n_c, args.vocab_size, args)

    def load_targets(self, tokens, context_size):
        self.tokens = tokens

    def forward(self, z, args):
        CONTEXT_SIZE = args.context_size
        self.output[:] = 0.0
        for pos in range(CONTEXT_SIZE):
            cache_index(self.unembedder, self.unembedder_vars[pos], z[pos], args)
            LUT_forward(self.unembedder, self.unembedder_vars[pos], self.output[pos], args)

    def backward(self, x_grad, args):
        CONTEXT_SIZE = args.context_size
        for pos in range(CONTEXT_SIZE):
            LUT_backward(self.unembedder, self.unembedder_vars[pos], x_grad[pos], self.output[pos], args)

    def compute_gradients(self, args):
        CONTEXT_SIZE = args.context_size
        VOCAB_SIZE = args.vocab_size
        for pos in range(CONTEXT_SIZE):
            softmax(self.output[pos], VOCAB_SIZE, 1.0)
            self.output[pos][self.tokens[pos + 1]] -= 1.0

    def sample_token(self, args):
        CONTEXT_SIZE = args.context_size
        VOCAB_SIZE = args.vocab_size
        softmax(self.output[CONTEXT_SIZE - 1], VOCAB_SIZE, args.temperature)
        return sample(self.output[CONTEXT_SIZE - 1], VOCAB_SIZE)

    def validation_loss(self, args):
        last = args.context_size - 1
        softmax(self.output[last], args.vocab_size, 1.0)
        target = self.tokens[args.context_size]
        prob = max(self.output[last][target], 1e-30)
        return -math.log(prob)


class FactoredOutputHead:
    __slots__ = ['unembedder_hi', 'unembedder_hi_vars', 'output_hi',
                 'unembedder_lo', 'unembedder_lo_vars', 'output_lo',
                 'tokens_hi', 'tokens_lo']
    def __init__(self, args):
        N_T = args.n_t
        CONTEXT_SIZE = args.context_size
        VOCAB_HI = args.vocab_hi
        self.unembedder_hi = LUT(N_T)
        self.unembedder_hi_vars = [LUTcache(N_T) for _ in range(CONTEXT_SIZE)]
        self.output_hi = np.zeros((CONTEXT_SIZE, VOCAB_HI), dtype=np.float32)
        self.unembedder_lo = LUT(N_T)
        self.unembedder_lo_vars = [LUTcache(N_T) for _ in range(CONTEXT_SIZE)]
        self.output_lo = np.zeros((CONTEXT_SIZE, 256), dtype=np.float32)
        self.tokens_hi = np.zeros(CONTEXT_SIZE + 1, dtype=np.int32)
        self.tokens_lo = np.zeros(CONTEXT_SIZE + 1, dtype=np.int32)

    def build(self, n_c, args):
        build_LUT(self.unembedder_hi, n_c, args.vocab_hi, args)
        build_LUT(self.unembedder_lo, n_c, 256, args)

    def load_targets(self, tokens, context_size):
        for pos in range(context_size + 1):
            self.tokens_hi[pos] = tokens[pos] // 256
            self.tokens_lo[pos] = tokens[pos] % 256

    def forward(self, z, args):
        CONTEXT_SIZE = args.context_size
        self.output_hi[:] = 0.0
        self.output_lo[:] = 0.0
        for pos in range(CONTEXT_SIZE):
            cache_index(self.unembedder_hi, self.unembedder_hi_vars[pos], z[pos], args)
            LUT_forward(self.unembedder_hi, self.unembedder_hi_vars[pos], self.output_hi[pos], args)
            cache_index(self.unembedder_lo, self.unembedder_lo_vars[pos], z[pos], args)
            LUT_forward(self.unembedder_lo, self.unembedder_lo_vars[pos], self.output_lo[pos], args)

    def backward(self, x_grad, args):
        CONTEXT_SIZE = args.context_size
        for pos in range(CONTEXT_SIZE):
            LUT_backward(self.unembedder_hi, self.unembedder_hi_vars[pos], x_grad[pos], self.output_hi[pos], args)
            LUT_backward(self.unembedder_lo, self.unembedder_lo_vars[pos], x_grad[pos], self.output_lo[pos], args)

    def compute_gradients(self, args):
        CONTEXT_SIZE = args.context_size
        VOCAB_HI = args.vocab_hi
        for pos in range(CONTEXT_SIZE):
            softmax(self.output_hi[pos], VOCAB_HI, 1.0)
            self.output_hi[pos][self.tokens_hi[pos + 1]] -= 1.0
            softmax(self.output_lo[pos], 256, 1.0)
            self.output_lo[pos][self.tokens_lo[pos + 1]] -= 1.0

    def sample_token(self, args):
        CONTEXT_SIZE = args.context_size
        VOCAB_SIZE = args.vocab_size
        VOCAB_HI = args.vocab_hi
        softmax(self.output_hi[CONTEXT_SIZE - 1], VOCAB_HI, args.temperature)
        hi = sample(self.output_hi[CONTEXT_SIZE - 1], VOCAB_HI)
        softmax(self.output_lo[CONTEXT_SIZE - 1], 256, args.temperature)
        lo = sample(self.output_lo[CONTEXT_SIZE - 1], 256)
        token_id = hi * 256 + lo
        if token_id >= VOCAB_SIZE:
            token_id = hi * 256
        return token_id

    def validation_loss(self, args):
        last = args.context_size - 1
        softmax(self.output_hi[last], args.vocab_hi, 1.0)
        softmax(self.output_lo[last], 256, 1.0)
        target_hi = self.tokens_hi[args.context_size]
        target_lo = self.tokens_lo[args.context_size]
        prob_hi = max(self.output_hi[last][target_hi], 1e-30)
        prob_lo = max(self.output_lo[last][target_lo], 1e-30)
        return -math.log(prob_hi) + -math.log(prob_lo)


class Model:
    __slots__ = ['Token_embedder', 'tokens', 'z', 'FFN', 'FFN_cache', 'head', 'output_head']
    def __init__(self, args):
        VOCAB_SIZE = args.vocab_size
        EMBEDDING_DIM = args.embedding_dim
        CONTEXT_SIZE = args.context_size
        NUM_LAYERS = args.num_layers
        NUM_HEADS = args.num_heads
        N_T = args.n_t

        self.Token_embedder = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
        self.tokens = np.zeros(CONTEXT_SIZE + 1, dtype=np.int32)
        self.z = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM), dtype=np.float32)

        self.FFN = [LUT(N_T) for _ in range(NUM_LAYERS)]
        self.FFN_cache = [[LUTcache(N_T) for _ in range(CONTEXT_SIZE)] for _ in range(NUM_LAYERS)]

        self.head = [[AttentionHead(args) for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]

        if args.factored_output:
            self.output_head = FactoredOutputHead(args)
        else:
            self.output_head = StandardOutputHead(args)


class TrainingData:
    __slots__ = ['data', 'length', 'val_data', 'val_length', 'testing_input_data']
    def __init__(self):
        self.data = None
        self.length = 0
        self.val_data = None
        self.val_length = 0
        self.testing_input_data = None


# ---------------------------------------------------------------------------
# LUT operations
# ---------------------------------------------------------------------------

def build_LUT(lut, total_n_c, y_dim, args):
    N_T = args.n_t
    N_C = args.n_c
    EMBEDDING_DIM = args.embedding_dim

    lut.y_dim = y_dim
    num_bins = 1 << total_n_c
    lut.S = np.zeros((N_T, num_bins, y_dim), dtype=np.float32)
    lut.a_arr = np.zeros((N_T, N_C), dtype=np.int32)
    lut.b_arr = np.zeros((N_T, N_C), dtype=np.int32)
    lut.bitmasks = (1 << np.arange(N_C)).astype(np.int32)
    for i in range(N_T):
        lut.a_arr[i] = fill_vector_with_random_integers(N_C, EMBEDDING_DIM)
        lut.b_arr[i] = fill_vector_with_random_integers_different_from_vector2(
            lut.a_arr[i], N_C, EMBEDDING_DIM)


def cache_index(lut, cache, x, args):
    N_T = args.n_t
    u = x[lut.a_arr] - x[lut.b_arr]                        # (N_T, N_C)
    cache.j[:N_T] = ((u > 0).astype(np.int32) * lut.bitmasks).sum(axis=1)
    abs_u = np.abs(u)
    cache.r_min[:N_T] = abs_u.argmin(axis=1)
    cache.u_min[:N_T] = u[np.arange(N_T), cache.r_min[:N_T]]


def cache_PE_index(cache, u, args):
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim
    bitmasks = (1 << np.arange(POSITIONAL_DIM)).astype(np.int32)
    cache.j[:N_T] = ((u[:N_T] > 0).astype(np.int32) * bitmasks).sum(axis=1)
    abs_u = np.abs(u[:N_T])
    cache.r_min[:N_T] = abs_u.argmin(axis=1)
    cache.u_min[:N_T] = u[np.arange(N_T), cache.r_min[:N_T]]


def LUT_forward(lut, cache, y, args):
    N_T = args.n_t
    y[:lut.y_dim] += lut.S[np.arange(N_T), cache.j[:N_T].astype(int)].sum(axis=0)


def backward_update(lut, cache, i, j_bin, jbar_bin, y_gradient, gradient):
    """Equivalent of BACKWARD_UPDATE macro. Uses bin indices (not flat offsets)."""
    gi = np.dot(y_gradient[:lut.y_dim], lut.S[i][jbar_bin] - lut.S[i][j_bin])
    v = gi * Up(cache.u_min[i])
    gradient[lut.a_arr[i, cache.r_min[i]]] += v
    gradient[lut.b_arr[i, cache.r_min[i]]] -= v


def LUT_backward(lut, cache, x_gradient, y_gradient, args):
    global learning_rate
    N_T = args.n_t
    trees = np.arange(N_T)
    j_bins = cache.j[:N_T].astype(int)
    jbar_bins = (cache.j[:N_T] ^ (1 << cache.r_min[:N_T])).astype(int)

    S_j = lut.S[trees, j_bins]            # (N_T, y_dim)
    S_jbar = lut.S[trees, jbar_bins]      # (N_T, y_dim)

    gi = ((S_jbar - S_j) * y_gradient[:lut.y_dim]).sum(axis=1)
    u_min = cache.u_min[:N_T]
    sign_u = np.where(u_min > 0, 1.0, -1.0)
    v = gi * (-0.5 * sign_u / (1 + np.abs(u_min))**2)

    np.add.at(x_gradient, lut.a_arr[trees, cache.r_min[:N_T]], v)
    np.add.at(x_gradient, lut.b_arr[trees, cache.r_min[:N_T]], -v)

    lut.S[trees, j_bins] -= learning_rate * y_gradient[:lut.y_dim]


def CONCATENATE(Q, P, PE, args):
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    return int((Q << (N_C + POSITIONAL_DIM)) | (P << POSITIONAL_DIM) | PE)


def concatenated_LUT_forward(lut, cacheQ, cacheK, cachePE, y, args):
    N_T = args.n_t

    for i in range(N_T):
        j_bin = CONCATENATE(int(cacheQ.j[i]), int(cacheK.j[i]), int(cachePE.j[i]), args)
        y[:lut.y_dim] += lut.S[i, j_bin]


def concatenated_LUT_backward(lut, cacheQ, cacheK, cachePE,
                                x_gradientQ, x_gradientK, PE_grad, y_gradient, args):
    global learning_rate
    N_T = args.n_t

    for i in range(N_T):
        j_bin = CONCATENATE(int(cacheQ.j[i]), int(cacheK.j[i]), int(cachePE.j[i]), args)

        if abs(cacheQ.u_min[i]) < abs(cacheK.u_min[i]):
            jbar_bin = CONCATENATE(
                int(cacheQ.j[i] ^ (1 << cacheQ.r_min[i])),
                int(cacheK.j[i]),
                int(cachePE.j[i]),
                args)
            backward_update(lut, cacheQ, i, j_bin, jbar_bin, y_gradient, x_gradientQ)
        else:
            jbar_bin = CONCATENATE(
                int(cacheQ.j[i]),
                int(cacheK.j[i] ^ (1 << cacheK.r_min[i])),
                int(cachePE.j[i]),
                args)
            backward_update(lut, cacheK, i, j_bin, jbar_bin, y_gradient, x_gradientK)

        if (abs(cachePE.u_min[i]) < abs(cacheQ.u_min[i]) and
                abs(cachePE.u_min[i]) < abs(cacheK.u_min[i])):
            jbarPE_bin = CONCATENATE(
                int(cacheQ.j[i]),
                int(cacheK.j[i]),
                int(cachePE.j[i] ^ (1 << cachePE.r_min[i])),
                args)
            giPE = np.dot(y_gradient[:lut.y_dim], lut.S[i][jbarPE_bin] - lut.S[i][j_bin])
            deltaPE = giPE * Up(cachePE.u_min[i])
            PE_grad[i][cachePE.r_min[i]] += deltaPE

        lut.S[i, j_bin] -= learning_rate * y_gradient[:lut.y_dim]


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------

def build_Model(m, args):
    VOCAB_SIZE = args.vocab_size
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    CONTEXT_SIZE = args.context_size
    N_T = args.n_t

    m.Token_embedder[:] = random_vector(VOCAB_SIZE * EMBEDDING_DIM, 1.0).reshape(VOCAB_SIZE, EMBEDDING_DIM)

    for l in range(NUM_LAYERS):
        build_LUT(m.FFN[l], N_C, EMBEDDING_DIM, args)
        for h in range(NUM_HEADS):
            m.head[l][h].Positional_encoding[:] = random_vector(
                CONTEXT_SIZE * N_T * POSITIONAL_DIM, 1.0
            ).reshape(CONTEXT_SIZE, N_T, POSITIONAL_DIM)
            build_LUT(m.head[l][h].V, N_C + N_C + POSITIONAL_DIM, EMBEDDING_DIM, args)

    m.output_head.build(N_C, args)


def attention_forward(head, x, y, args):
    CONTEXT_SIZE = args.context_size

    for pos in range(CONTEXT_SIZE):
        cache_index(head.V, head.V_cache[pos], x[pos], args)
        cache_PE_index(head.PE_cache[pos], head.Positional_encoding[pos], args)

    for pos in range(1, CONTEXT_SIZE):
        for pos1 in range(pos):
            concatenated_LUT_forward(
                head.V, head.V_cache[pos], head.V_cache[pos1],
                head.PE_cache[pos - pos1], y[pos], args)


def attention_backward(head, x_grad, y_grad, args):
    global learning_rate
    CONTEXT_SIZE = args.context_size
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim

    pos_grad = np.zeros((CONTEXT_SIZE, N_T, POSITIONAL_DIM), dtype=np.float32)

    for pos in range(1, CONTEXT_SIZE):
        for pos1 in range(pos):
            concatenated_LUT_backward(
                head.V, head.V_cache[pos], head.V_cache[pos1],
                head.PE_cache[pos - pos1],
                x_grad[pos], x_grad[pos1], pos_grad[pos - pos1],
                y_grad[pos], args)

    head.Positional_encoding -= learning_rate * pos_grad


def model_forward(m, args):
    CONTEXT_SIZE = args.context_size
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    for l in range(NUM_LAYERS):
        x = m.z.copy()
        for h in range(NUM_HEADS):
            attention_forward(m.head[l][h], x, m.z, args)

        for pos in range(CONTEXT_SIZE):
            cache_index(m.FFN[l], m.FFN_cache[l][pos], m.z[pos], args)
            LUT_forward(m.FFN[l], m.FFN_cache[l][pos], m.z[pos], args)

    m.output_head.forward(m.z, args)


def model_backward(m, args):
    CONTEXT_SIZE = args.context_size
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    x_grad = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM), dtype=np.float32)

    m.output_head.backward(x_grad, args)

    for l in range(NUM_LAYERS - 1, -1, -1):
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        for pos in range(CONTEXT_SIZE):
            LUT_backward(m.FFN[l], m.FFN_cache[l][pos], x_grad[pos], y_grad[pos], args)

        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        for h in range(NUM_HEADS):
            attention_backward(m.head[l][h], x_grad, y_grad, args)

    # no need to compute gradients for the embedder; just update the synaptic values
    # (disabled in the C version too)
    # for pos in range(CONTEXT_SIZE):
    #     for k in range(EMBEDDING_DIM):
    #         m.Token_embedder[m.tokens[pos]][k] -= learning_rate * x_grad[pos][k]


def model_training_step(m, args):
    model_forward(m, args)
    m.output_head.compute_gradients(args)
    model_backward(m, args)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_training_data(training, args):
    CONTEXT_SIZE = args.context_size
    TESTING_LENGTH = args.testing_length
    enc = args.enc

    # Load and tokenize training data
    try:
        with open(args.training_data, 'r', encoding='utf-8') as f:
            text = f.read()
    except IOError:
        print(f"Error opening training datafile {args.training_data}")
        sys.exit(1)

    tokens = enc.encode(text)
    training.data = np.array(tokens, dtype=np.int32)
    training.length = len(training.data) - CONTEXT_SIZE - 1
    print(f"Training data: {len(training.data)} tokens from {args.training_data}")

    # Load or split validation data
    if args.validation_data:
        try:
            with open(args.validation_data, 'r', encoding='utf-8') as f:
                val_text = f.read()
        except IOError:
            print(f"Error opening validation datafile {args.validation_data}")
            sys.exit(1)
        val_tokens = enc.encode(val_text)
        training.val_data = np.array(val_tokens, dtype=np.int32)
        training.val_length = len(training.val_data) - CONTEXT_SIZE - 1
        print(f"Validation data: {len(training.val_data)} tokens from {args.validation_data}")
    else:
        # No separate validation file: use random snippets from training data
        training.val_data = training.data
        training.val_length = training.length
        print("Validation data: sampled from training data")

    # Sample validation snippet indices from val data
    training.testing_input_data = np.zeros(TESTING_LENGTH, dtype=np.int32)
    for i in range(TESTING_LENGTH):
        training.testing_input_data[i] = random.randint(0, training.val_length - 1)

    print(f"Tokenizer: {args.tokenizer}, vocab_size: {args.vocab_size}")


def get_random_training_index(training):
    return random.randint(0, training.length - 1)


def embed_token(m, data, pos, output):
    EMBEDDING_DIM = output.shape[0] if hasattr(output, 'shape') else len(output)
    output[:EMBEDDING_DIM] = m.Token_embedder[data[pos]]


def load_snippet(m, data_array, char_start, args):
    CONTEXT_SIZE = args.context_size

    for pos in range(CONTEXT_SIZE):
        embed_token(m, data_array, char_start + pos, m.z[pos])
        m.tokens[pos] = int(data_array[char_start + pos])
    m.tokens[CONTEXT_SIZE] = int(data_array[char_start + CONTEXT_SIZE])

    m.output_head.load_targets(m.tokens, CONTEXT_SIZE)


def model_inference(m, args):
    model_forward(m, args)
    return m.output_head.sample_token(args)


def model_prompt_response(m, prompt_text, response_length, args):
    CONTEXT_SIZE = args.context_size
    EMBEDDING_DIM = args.embedding_dim
    enc = args.enc

    # Encode prompt to token IDs
    prompt_tokens = enc.encode(prompt_text)
    # Truncate or pad to CONTEXT_SIZE
    if len(prompt_tokens) > CONTEXT_SIZE:
        prompt_tokens = prompt_tokens[:CONTEXT_SIZE]
    else:
        prompt_tokens = [0] * (CONTEXT_SIZE - len(prompt_tokens)) + prompt_tokens
    prompt_tokens = list(prompt_tokens)

    # Print the prompt
    sys.stdout.write(prompt_text[:80])

    for i in range(response_length):
        for pos in range(CONTEXT_SIZE):
            m.z[pos][:EMBEDDING_DIM] = m.Token_embedder[prompt_tokens[pos]]
        response = model_inference(m, args)
        sys.stdout.write(enc.decode([response]))

        # Shift context window and append new token
        prompt_tokens = prompt_tokens[1:] + [response]

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global learning_rate

    parser = argparse.ArgumentParser(description="SNN Transformer (Python port)")
    parser.add_argument('training_data', type=str, help='Path to training data file')
    parser.add_argument('--validation-data', type=str, default=None,
                        help='Path to validation data file (if not provided, samples from training data)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='tiktoken encoding name (default: gpt2)')
    parser.add_argument('--context-size', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='(auto-detected from tokenizer, this default is overridden)')
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--positional-dim', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--n-t', type=int, default=16)
    parser.add_argument('--n-c', type=int, default=6)
    parser.add_argument('--testing-length', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=100000000)
    parser.add_argument('--validation-interval', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--loss-file', type=str, default='loss.csv')
    parser.add_argument('--factored-output', action='store_true',
                        help='Use factored base-256 decomposition for unembedder (faster for large vocabs)')
    args = parser.parse_args()

    # Initialize tokenizer and auto-set vocab size
    enc = tiktoken.get_encoding(args.tokenizer)
    args.vocab_size = enc.n_vocab
    args.enc = enc

    # Factored output dimensions
    args.vocab_hi = (args.vocab_size + 255) // 256
    args.vocab_lo = 256

    # Initialize loss file with header
    with open(args.loss_file, 'w') as f:
        f.write("step, loss, perplexity\n")

    training = TrainingData()
    load_training_data(training, args)

    m = Model(args)
    build_Model(m, args)

    pbar = tqdm(range(args.max_steps), desc="Training", unit="step")
    last_ppl = None
    last_loss = None

    for t in pbar:
        load_snippet(m, training.data, get_random_training_index(training), args)

        # Adam learning rate scheduler
        learning_rate = min(1.0 / math.sqrt(1 + t), t / 4000.0 / math.sqrt(4000))

        model_training_step(m, args)

        if t % args.validation_interval == 0:
            pbar.set_description("Validating")

            validation_loss = 0.0
            for i in tqdm(range(args.testing_length), desc="  Val", leave=False, unit="snip"):
                load_snippet(m, training.val_data, int(training.testing_input_data[i]), args)
                model_forward(m, args)
                validation_loss += m.output_head.validation_loss(args)
            validation_loss /= args.testing_length

            perplexity = math.exp(validation_loss)
            last_ppl = perplexity
            last_loss = validation_loss

            with open(args.loss_file, 'a') as f:
                f.write(f"{t}, {validation_loss:.6f}, {perplexity:.2f}\n")

            # Print validation result and sample generation
            tqdm.write(f"\n--- step {t:,} | ppl={perplexity:.2f} | loss={validation_loss:.3f} ---")
            val_idx = random.randint(0, training.val_length - 1)
            prompt_tokens = training.val_data[val_idx:val_idx + args.context_size].tolist()
            prompt_text = enc.decode(prompt_tokens)
            # Capture generation output
            import io
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            model_prompt_response(m, prompt_text, 80, args)
            sys.stdout = old_stdout
            tqdm.write(buf.getvalue())
            tqdm.write("")

            pbar.set_description("Training")

        if last_ppl is not None:
            pbar.set_postfix(ppl=f"{last_ppl:.2f}", loss=f"{last_loss:.3f}", lr=f"{learning_rate:.4f}")


if __name__ == '__main__':
    main()
