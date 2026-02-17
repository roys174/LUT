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

class Anchors:
    __slots__ = ['a', 'b']
    def __init__(self, n_c):
        self.a = np.zeros(n_c, dtype=np.int32)
        self.b = np.zeros(n_c, dtype=np.int32)


class LUT:
    __slots__ = ['y_dim', 'S', 'anchors']
    def __init__(self, n_t):
        self.y_dim = 0
        self.S = [None] * n_t
        self.anchors = [None] * n_t


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


class Model:
    __slots__ = [
        'Token_embedder', 'tokens', 'z',
        'FFN', 'FFN_cache',
        'head',
        'unembedder', 'unembedder_vars',
        'output',
    ]
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

        self.unembedder = LUT(N_T)
        self.unembedder_vars = [LUTcache(N_T) for _ in range(CONTEXT_SIZE)]

        self.output = np.zeros((CONTEXT_SIZE, VOCAB_SIZE), dtype=np.float32)


class TrainingData:
    __slots__ = ['data', 'length', 'reserved_for_testing', 'testing_input_data']
    def __init__(self):
        self.data = None
        self.length = 0
        self.reserved_for_testing = None
        self.testing_input_data = None


# ---------------------------------------------------------------------------
# LUT operations
# ---------------------------------------------------------------------------

def build_LUT(lut, total_n_c, y_dim, args):
    N_T = args.n_t
    N_C = args.n_c
    EMBEDDING_DIM = args.embedding_dim

    lut.y_dim = y_dim
    for i in range(N_T):
        lut.anchors[i] = Anchors(N_C)
        lut.anchors[i].a = fill_vector_with_random_integers(N_C, EMBEDDING_DIM)
        lut.anchors[i].b = fill_vector_with_random_integers_different_from_vector2(
            lut.anchors[i].a, N_C, EMBEDDING_DIM)
        lut.S[i] = np.zeros((1 << total_n_c) * y_dim, dtype=np.float32)


def cache_index(lut, cache, x, args):
    N_T = args.n_t
    N_C = args.n_c

    for i in range(N_T):
        cache.j[i] = 0
        cache.u_min[i] = float('inf')

        for r in range(N_C):
            u = x[lut.anchors[i].a[r]] - x[lut.anchors[i].b[r]]
            if u > 0:
                cache.j[i] |= (1 << r)
            if abs(u) < abs(cache.u_min[i]):
                cache.r_min[i] = r
                cache.u_min[i] = u


def cache_PE_index(cache, u, args):
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim

    for i in range(N_T):
        cache.j[i] = 0
        cache.u_min[i] = float('inf')

        for r in range(POSITIONAL_DIM):
            if u[i][r] > 0:
                cache.j[i] |= (1 << r)
            if abs(u[i][r]) < abs(cache.u_min[i]):
                cache.r_min[i] = r
                cache.u_min[i] = u[i][r]


def LUT_forward(lut, cache, y, args):
    N_T = args.n_t

    for i in range(N_T):
        offset = int(cache.j[i]) * lut.y_dim
        y[:lut.y_dim] += lut.S[i][offset:offset + lut.y_dim]


def backward_update(lut, cache, i, j, jbar, y_gradient, gradient):
    """Equivalent of BACKWARD_UPDATE macro."""
    gi = 0.0
    for k in range(lut.y_dim):
        gi += y_gradient[k] * (lut.S[i][jbar + k] - lut.S[i][j + k])
    v = gi * Up(cache.u_min[i])
    gradient[lut.anchors[i].a[cache.r_min[i]]] += v
    gradient[lut.anchors[i].b[cache.r_min[i]]] -= v


def LUT_backward(lut, cache, x_gradient, y_gradient, args):
    global learning_rate
    N_T = args.n_t

    for i in range(N_T):
        j = int(cache.j[i]) * lut.y_dim
        jbar = int(cache.j[i] ^ (1 << cache.r_min[i])) * lut.y_dim

        backward_update(lut, cache, i, j, jbar, y_gradient, x_gradient)

        lut.S[i][j:j + lut.y_dim] -= learning_rate * y_gradient[:lut.y_dim]


def CONCATENATE(Q, P, PE, y_dim, args):
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    return int(((Q << (N_C + POSITIONAL_DIM)) | (P << POSITIONAL_DIM) | PE) * y_dim)


def concatenated_LUT_forward(lut, cacheQ, cacheK, cachePE, y, args):
    N_T = args.n_t

    for i in range(N_T):
        j = CONCATENATE(int(cacheQ.j[i]), int(cacheK.j[i]), int(cachePE.j[i]), lut.y_dim, args)
        y[:lut.y_dim] += lut.S[i][j:j + lut.y_dim]


def concatenated_LUT_backward(lut, cacheQ, cacheK, cachePE,
                                x_gradientQ, x_gradientK, PE_grad, y_gradient, args):
    global learning_rate
    N_T = args.n_t

    for i in range(N_T):
        j = CONCATENATE(int(cacheQ.j[i]), int(cacheK.j[i]), int(cachePE.j[i]), lut.y_dim, args)

        if abs(cacheQ.u_min[i]) < abs(cacheK.u_min[i]):
            jbar = CONCATENATE(
                int(cacheQ.j[i] ^ (1 << cacheQ.r_min[i])),
                int(cacheK.j[i]),
                int(cachePE.j[i]),
                lut.y_dim, args)
            backward_update(lut, cacheQ, i, j, jbar, y_gradient, x_gradientQ)
        else:
            jbar = CONCATENATE(
                int(cacheQ.j[i]),
                int(cacheK.j[i] ^ (1 << cacheK.r_min[i])),
                int(cachePE.j[i]),
                lut.y_dim, args)
            backward_update(lut, cacheK, i, j, jbar, y_gradient, x_gradientK)

        if (abs(cachePE.u_min[i]) < abs(cacheQ.u_min[i]) and
                abs(cachePE.u_min[i]) < abs(cacheK.u_min[i])):
            jbarPE = CONCATENATE(
                int(cacheQ.j[i]),
                int(cacheK.j[i]),
                int(cachePE.j[i] ^ (1 << cachePE.r_min[i])),
                lut.y_dim, args)
            giPE = 0.0
            for k in range(lut.y_dim):
                giPE += y_gradient[k] * (lut.S[i][jbarPE + k] - lut.S[i][j + k])
            deltaPE = giPE * Up(cachePE.u_min[i])
            PE_grad[i][cachePE.r_min[i]] += deltaPE

        lut.S[i][j:j + lut.y_dim] -= learning_rate * y_gradient[:lut.y_dim]


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

    build_LUT(m.unembedder, N_C, VOCAB_SIZE, args)


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
    VOCAB_SIZE = args.vocab_size

    for l in range(NUM_LAYERS):
        x = m.z.copy()
        for h in range(NUM_HEADS):
            attention_forward(m.head[l][h], x, m.z, args)

        for pos in range(CONTEXT_SIZE):
            cache_index(m.FFN[l], m.FFN_cache[l][pos], m.z[pos], args)
            LUT_forward(m.FFN[l], m.FFN_cache[l][pos], m.z[pos], args)

    m.output[:] = 0.0
    for pos in range(CONTEXT_SIZE):
        cache_index(m.unembedder, m.unembedder_vars[pos], m.z[pos], args)
        LUT_forward(m.unembedder, m.unembedder_vars[pos], m.output[pos], args)


def model_backward(m, args):
    CONTEXT_SIZE = args.context_size
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    x_grad = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM), dtype=np.float32)

    for pos in range(CONTEXT_SIZE):
        LUT_backward(m.unembedder, m.unembedder_vars[pos], x_grad[pos], m.output[pos], args)

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
    CONTEXT_SIZE = args.context_size
    VOCAB_SIZE = args.vocab_size

    model_forward(m, args)
    for pos in range(CONTEXT_SIZE):
        softmax(m.output[pos], VOCAB_SIZE, 1.0)
        m.output[pos][m.tokens[pos + 1]] -= 1.0  # output becomes a gradient
    model_backward(m, args)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_training_data(training, fname, args):
    CONTEXT_SIZE = args.context_size
    TESTING_LENGTH = args.testing_length

    try:
        with open(fname, 'rb') as f:
            data = f.read()
    except IOError:
        print(f"Error opening training datafile {fname}")
        sys.exit(1)

    print(f"Successfully opened training data file {fname}")

    training.data = np.frombuffer(data, dtype=np.uint8).copy()
    training.length = len(training.data) - CONTEXT_SIZE - 1

    training.reserved_for_testing = np.zeros(training.length, dtype=np.uint8)

    training.testing_input_data = np.zeros(TESTING_LENGTH, dtype=np.int32)
    for i in range(TESTING_LENGTH):
        training.testing_input_data[i] = random.randint(0, training.length - 1)
        for j in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
            idx = max(0, training.testing_input_data[i] + j)
            if idx < training.length:
                training.reserved_for_testing[idx] = 1

    print("Successfully loaded training data")


def get_random_training_index(training):
    while True:
        symbol_index = random.randint(0, training.length - 1)
        if training.reserved_for_testing[symbol_index] != 1:
            return symbol_index


def embed_token(m, data, pos, output):
    EMBEDDING_DIM = output.shape[0] if hasattr(output, 'shape') else len(output)
    output[:EMBEDDING_DIM] = m.Token_embedder[data[pos]]


def load_snippet(m, training, char_start, args):
    CONTEXT_SIZE = args.context_size

    for pos in range(CONTEXT_SIZE):
        embed_token(m, training.data, char_start + pos, m.z[pos])
        m.tokens[pos] = int(training.data[char_start + pos])
    m.tokens[CONTEXT_SIZE] = int(training.data[char_start + CONTEXT_SIZE])


def model_inference(m, args):
    CONTEXT_SIZE = args.context_size
    VOCAB_SIZE = args.vocab_size

    model_forward(m, args)
    softmax(m.output[CONTEXT_SIZE - 1], VOCAB_SIZE, args.temperature)
    sampled_index = sample(m.output[CONTEXT_SIZE - 1], VOCAB_SIZE)
    return sampled_index


def model_prompt_response(m, prompt, response_length, args):
    CONTEXT_SIZE = args.context_size
    EMBEDDING_DIM = args.embedding_dim

    prompt_copy = bytearray(prompt[:CONTEXT_SIZE].ljust(CONTEXT_SIZE, b'\x00'))
    sys.stdout.write(prompt_copy.decode('ascii', errors='replace'))

    for i in range(response_length):
        for pos in range(CONTEXT_SIZE):
            m.z[pos][:EMBEDDING_DIM] = m.Token_embedder[prompt_copy[pos]]
        response = model_inference(m, args)
        sys.stdout.write(chr(response) if 32 <= response < 127 else '?')

        # shift the prompt by one character and insert response as the last character
        prompt_copy[:-1] = prompt_copy[1:]
        prompt_copy[CONTEXT_SIZE - 1] = response

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global learning_rate

    parser = argparse.ArgumentParser(description="SNN Transformer (Python port)")
    parser.add_argument('training_data', type=str, help='Path to training data file')
    parser.add_argument('--context-size', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=256)
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
    args = parser.parse_args()

    # Initialize loss file
    with open(args.loss_file, 'w') as f:
        pass

    training = TrainingData()
    load_training_data(training, args.training_data, args)

    m = Model(args)
    build_Model(m, args)

    for t in range(args.max_steps):
        load_snippet(m, training, get_random_training_index(training), args)

        # Adam learning rate scheduler
        learning_rate = min(1.0 / math.sqrt(1 + t), t / 4000.0 / math.sqrt(4000))

        model_training_step(m, args)

        if t % args.validation_interval == 0:
            sys.stdout.write("...validating... ")
            sys.stdout.flush()

            validation_loss = 0.0
            for i in range(args.testing_length):
                load_snippet(m, training, int(training.testing_input_data[i]), args)
                model_forward(m, args)
                softmax(m.output[args.context_size - 1], args.vocab_size, 1.0)
                target = m.tokens[args.context_size]
                prob = m.output[args.context_size - 1][target]
                validation_loss += -math.log(max(prob, 1e-30))
            validation_loss /= args.testing_length

            with open(args.loss_file, 'a') as f:
                f.write(f"{t}, {validation_loss:.6f}\n")

            sys.stdout.write(f"\rt={t // 1000},000, loss={validation_loss:5.3f}: ")
            model_prompt_response(
                m,
                b"insert your validation prompt here ",
                80, args)
            print()

        sys.stdout.write(f"\rt={t}")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
