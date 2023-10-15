# llama2.py
import os
import sys
import time
import random
import math
import struct
from typing import List


class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len


class TransformerWeights:
    token_embedding_table: List[float]
    rms_att_weight: List[float]
    wq: List[float]
    wk: List[float]
    wv: List[float]
    wo: List[float]
    rms_ffn_weight: List[float]
    w1: List[float]
    w3: List[float]
    w2: List[float]
    rms_final_weight: List[float]
    freq_cis_real: List[float]
    freq_cis_imag: List[float]
    wcls: List[float]

# ----------------------------------------------------------------------------
# initialization: read from checkpoint

def checkpoint_init_weights(weights: TransformerWeights,
                            conf: Config,
                            file,
                            shared_weights: int,
                            file_size) -> None:
    def read_floats(count):
        values = struct.unpack('f' * count, file.read(count * 4))
        return values

    weights.token_embedding_table = read_floats(conf.vocab_size * conf.dim)
    weights.rms_att_weight = read_floats(conf.n_layers * conf.dim)
    weights.wq = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.wk = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.wv = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.wo = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.rms_ffn_weight = read_floats(conf.n_layers * conf.dim)
    weights.w1 = read_floats(conf.n_layers * conf.dim * conf.hidden_dim)
    weights.w2 = read_floats(conf.n_layers * conf.hidden_dim * conf.dim)
    weights.w3 = read_floats(conf.n_layers * conf.dim * conf.hidden_dim)
    weights.rms_final_weight = read_floats(conf.dim)
    weights.freq_cis_real = read_floats(conf.seq_len * (conf.dim // conf.n_heads) // 2)
    weights.freq_cis_imag = read_floats(conf.seq_len * (conf.dim // conf.n_heads) // 2)
    weights.wcls = weights.token_embedding_table if shared_weights else read_floats((file_size - file.tell()) // 4)


def tokenizer_init(conf: Config, file):
    vocab, vocab_scores, max_token_length = [], [], 0

    def read_types(count, type='i', length=4):
        values = struct.unpack(type * count, file.read(count * length))
        return values

    max_token_length = read_types(1)[0]
    for i in range(0, conf.vocab_size):
        vocab_scores.append(read_types(1, 'f')[0])
        len = read_types(1)[0]
        bstr = read_types(1, f'{len}s', len)
        vocab.append(bstr[0])
    return vocab, vocab_scores, max_token_length


def accum(a, b):
    for i in range(len(a)):
        a[i] += b[i]
    return a


def rmsnorm(out, x, weight):
    size = len(x)
    # calculate sum of squares
    ss = 0.0
    for j in range(size):
        ss += x[j] * x[j]
    ss /= size
    ss += 1e-5
    ss = 1.0 / math.sqrt(ss)
    # normalize and scale
    for j in range(size):
        out[j] = weight[j] * (ss * x[j])
    return out


def softmax(x, size):
    # find max value (for numerical stability)
    max_val = x[0]
    for i in range(1, size):
        if x[i] > max_val:
            max_val = x[i]
    # exp and sum
    exp_sum = 0.0
    for i in range(size):
        x[i] = math.exp(x[i] - max_val)
        exp_sum += x[i]
    # normalize
    for i in range(size):
        x[i] /= exp_sum
    return x


def matmul(xout, x, w, n, d):
    # W (d,n) @ x (n,) -> xout (d,)
    # by far the most amount of time is spent inside this little function
    for i in range(d):
        val = 0.0
        for j in range(n):
            val += w[i * n + j] * x[j]
        xout[i] = val
    return xout


class RunState:
    x: List[float]
    xb: List[float]
    q: List[float]
    k: List[float]
    v: List[float]
    att: List[float]
    key_cache: List[float]
    value_cache: List[float]
    xb2: List[float]
    hb: List[float]
    hb2: List[float]
    logits: List[float]


# token, pos, config, state, weights
def transformer(token: int, pos: int, conf: Config, state: RunState, weights: TransformerWeights) -> None:
    # A few convenience variables
    x = state.x
    dim = conf.dim
    hidden_dim = conf.hidden_dim
    head_size = dim // conf.n_heads

    # Copy the token embedding into x
    content_row = weights.token_embedding_table[token * dim: (token + 1) * dim]
    x[:] = content_row

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    freq_cis_real_row = weights.freq_cis_real[pos *
                                              head_size // 2: (pos + 1) * head_size // 2]
    freq_cis_imag_row = weights.freq_cis_imag[pos *
                                              head_size // 2: (pos + 1) * head_size // 2]

    # Forward all the layers
    for l in range(conf.n_layers):
        # Attention rmsnorm
        state.xb = rmsnorm(state.xb, x, weights.rms_att_weight[l * dim: (l + 1) * dim])

        # QKV matmuls for this position
        state.q = matmul(state.q, state.xb, weights.wq[l * dim * dim: (l + 1) * dim * dim], dim, dim)
        state.k = matmul(state.k, state.xb, weights.wk[l * dim * dim: (l + 1) * dim * dim], dim, dim)
        state.v = matmul(state.v, state.xb, weights.wv[l * dim * dim: (l + 1) * dim * dim], dim, dim)

        # Apply RoPE rotation to the q and k vectors for each head
        for h in range(conf.n_heads):
            # Get the q and k vectors for this head
            q = state.q[h * head_size: (h + 1) * head_size]
            k = state.k[h * head_size: (h + 1) * head_size]

            # Rotate q and k by the freq_cis_real and freq_cis_imag
            for i in range(0, head_size, 2):
                q0, q1 = q[i], q[i + 1]
                k0, k1 = k[i], k[i + 1]
                fcr = freq_cis_real_row[i // 2]
                fci = freq_cis_imag_row[i // 2]
                q[i] = q0 * fcr - q1 * fci
                q[i + 1] = q0 * fci + q1 * fcr
                k[i] = k0 * fcr - k1 * fci
                k[i + 1] = k0 * fci + k1 * fcr

            # reassigned back to state.q and state.k
            state.q[h * head_size: (h + 1) * head_size] = q
            state.k[h * head_size: (h + 1) * head_size] = k

        # Save key,value at this time step (pos) to our kv cache
        loff = l * conf.seq_len * dim  # kv cache layer offset for convenience
        state.key_cache[loff + pos * dim: loff + (pos + 1) * dim] = state.k
        state.value_cache[loff + pos * dim: loff + (pos + 1) * dim] = state.v

        # Multihead attention. Iterate over all heads
        for h in range(conf.n_heads):
            # Get the query vector for this head
            q = state.q[h * head_size: (h + 1) * head_size]

            # Attention scores for this head
            att = state.att[h * conf.seq_len: (h + 1) * conf.seq_len]

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Get the key vector for this head and at this timestep
                k = state.key_cache[loff + t * dim + h * head_size: loff + (t + 1) * dim + h * head_size]

                # Calculate the attention score as the dot product of q and k
                score = sum(q[i] * k[i] for i in range(head_size))
                score /= math.sqrt(head_size)

                # Save the score to the attention buffer
                att[t] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            att = softmax(att, pos + 1)

            xb_ptr = h * head_size
            # Weighted sum of the values, store back into xb
            state.xb[xb_ptr: (h + 1) * head_size] = [0.0] * head_size
            for t in range(pos + 1):
                # Get the value vector for this head and at this timestep
                v = state.value_cache[loff + t * dim + h *
                                      head_size: loff + (t + 1) * dim + h * head_size]
                # Get the attention weight for this timestep
                a = att[t]
                # Accumulate the weighted value into xb
                for i in range(head_size):
                    state.xb[xb_ptr + i] += a * v[i]

        # Final matrix multiplication to get the output of the attention
        state.xb2 = matmul(state.xb2, state.xb, weights.wo[l * dim * dim:(l + 1) * dim * dim], dim, dim)

        # Residual connection back into x
        x = accum(x, state.xb2)

        # FFN rmsnorm
        state.xb = rmsnorm(state.xb, x, weights.rms_ffn_weight[l * dim:(l + 1) * dim])

        # Calculate self.w1(x) and self.w3(x) for FFN
        state.hb = matmul(state.hb, state.xb,
                          weights.w1[l * dim * hidden_dim:
                                     (l + 1) * dim * hidden_dim],
                          dim, hidden_dim)

        state.hb2 = matmul(state.hb2, state.xb, weights.w3[l * dim * hidden_dim:
                                                           (l + 1) * dim * hidden_dim],
                           dim, hidden_dim)

        # Apply SiLU activation function (silu(x) = x * sigmoid(x))
        state.hb = [state.hb[i] * (1.0 / (1.0 + math.exp(-state.hb[i])))
                    for i in range(hidden_dim)]

        # Elementwise multiply with w3(x)
        state.hb = [state.hb[i] * state.hb2[i] for i in range(hidden_dim)]

        # Final matrix multiplication to get the output of the FFN
        state.xb = matmul(state.xb, state.hb, weights.w2[l * dim * hidden_dim:
                                                         (
                                                                 (l + 1)
                                                                 * dim * hidden_dim
                                                         )], hidden_dim, dim)

        # Residual connection
        x = accum(x, state.xb)

    # Final rmsnorm
    x = rmsnorm(x, x, weights.rms_final_weight)

    # Classifier into logits
    state.logits = matmul(state.logits, x, weights.wcls, dim, conf.vocab_size)


def str_lookup(string, vocab):
    # little trick to make string always compatible with byte-encoded tokens in vocab list
    string = string.encode() if isinstance(string, str) else string

    # Find the first perfect match for string in vocab, return its index or -1 if not found
    try:
        index = vocab.index(string)
        return index
    except ValueError as err:
        return -1


def bpe_encode(text, vocab, vocab_scores):
    tokens = []

    # First encode every individual character in the input text
    for pos, char in enumerate(text):
        string = char
        id = str_lookup(string, vocab)
        if id == -1:
            print(f"not a good prompt at pos {pos}")
            sys.exit(1)
        tokens.append(id)

    # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            # string = vocab[tokens[i]].rstrip(b'\x00') + vocab[tokens[i + 1]].rstrip(b'\x00')
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            id = str_lookup(string, vocab)
            if id != -1 and vocab_scores[id] > best_score:
                # This merge pair exists in vocab! Record its score and position
                best_score = vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break  # We couldn't find any more pairs to merge, so we're done

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        tokens = tokens[0:best_idx + 1] + tokens[best_idx + 2:]

    return tokens


def time_in_ms():
    # Returns time in milliseconds for benchmarking the model speed
    return int(time.time() * 1000)


def sample(probabilities):
    n = len(probabilities)
    # Sample index from probabilities, they must sum to 1
    r = random.random()
    cdf = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r < cdf:
            return i
    return n - 1  # In case of rounding errors


def argmax(v):
    # return argmax of v
    max_i = 0
    max_p = v[0]
    for i in range(1, len(v)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i


def init_run_state(state, config):
    state.x = [0.0] * config.dim
    state.xb = [0.0] * config.dim
    state.xb2 = [0.0] * config.dim
    state.hb = [0.0] * config.hidden_dim
    state.hb2 = [0.0] * config.hidden_dim
    state.q = [0.0] * config.dim
    state.k = [0.0] * config.dim
    state.v = [0.0] * config.dim
    state.att = [0.0] * (config.n_heads * config.seq_len)
    state.logits = [0.0] * config.vocab_size
    state.key_cache = [0.0] * (config.n_layers * config.seq_len * config.dim)
    state.value_cache = [0.0] * (config.n_layers * config.seq_len * config.dim)


def run(args):
    checkpoint = args["checkpoint"]
    temperature = float(args["temperature"])
    steps = int(args["steps"])
    prompt = args["prompt"]

    rng_seed = int(time.time())
    random.seed(rng_seed)

    # Read in the model.bin file
    weights = TransformerWeights()
    file_size = 0  # size of the checkpoint file in bytes

    with open(checkpoint, "rb") as file:
        if not file:
            print(f"Couldn't open file {checkpoint}")
            sys.exit(1)
        # Read in the config header
        _config = file.read(struct.calcsize('7i'))
        # Unpacking the data
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', _config)
        # Creating a Config object
        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)

        # negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = 1 if config.vocab_size > 0 else 0
        config.vocab_size = abs(config.vocab_size)
        file_size = os.path.getsize(checkpoint)  # get the file size, in bytes

        checkpoint_init_weights(weights, config, file, shared_weights, file_size)

    # Right now we cannot run for more than config.seq_len steps
    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    # Read in the tokenizer.bin file
    with open("tokenizer.bin", "rb") as file:
        if not file:
            print("Couldn't load tokenizer.bin")
            sys.exit(1)
        vocab, vocab_scores, max_token_length = tokenizer_init(config, file)

    # Create and initialize the application RunState
    state = RunState()
    init_run_state(state, config)

    # Process the prompt, if any
    prompt_tokens = []
    if prompt:
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores)

    # Start the main loop
    start = 0  # Used to time our code, only initialized after the first iteration
    next_token = 0  # Will store the next token in the sequence
    # Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    token = 1
    pos = 0  # Position in the sequence
    # Explicitly print the initial BOS token for stylistic symmetry reasons
    print("<s>")

    while pos < steps:
        # Forward the transformer to get logits for the next token
        transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
            # If we are still processing the input prompt, force the next prompt token
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highest probability
                next_token = argmax(state.logits)
            else:
                # Apply the temperature to the logits
                state.logits = [i / temperature for i in state.logits]
                # Apply softmax to the logits to get the probabilities for the next token
                softmax(state.logits, config.vocab_size)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)

        # Following BOS token (1), sentencepiece decoder strips any leading whitespace
        token_str = (
            vocab[next_token].lstrip()
            if token == 1 and vocab[next_token][0] == ' ' else vocab[next_token]
        )

        token_str = token_str.decode("utf-8")
        print(token_str, end="")
        sys.stdout.flush()

        # Advance forward
        token = next_token
        pos += 1

        # Initialize our timer here because the first iteration could be time consuming due to IO operations
        if start == 0:
            start = time_in_ms()

    # Report achieved tok/s
    end = time_in_ms()
    print(f"\nachieved tok/s: {(steps - 1) / (end - start) * 1000}")


if __name__ == "__main__":
    args = {
        "checkpoint": './out/stories15M.bin',
        "temperature": "0.0",
        "steps": "256",
        "prompt": None
    }
    # if len(sys.argv) < 2:
    #     print(
    #         "Usage: python script.py <checkpoint_file> [temperature] [steps] [prompt]")
    #     sys.exit(1)

    if len(sys.argv) >= 2:
        args["checkpoint"] = sys.argv[1]

    if len(sys.argv) >= 3:
        args["temperature"] = sys.argv[2]

    if len(sys.argv) >= 4:
        args["steps"] = sys.argv[3]

    if len(sys.argv) >= 5:
        args["prompt"] = sys.argv[4]

    run(args)
