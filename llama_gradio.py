import gradio as gr
import time
import random
import struct
import sys

from llama2 import TransformerWeights, checkpoint_init_weights, Config, tokenizer_init,RunState, bpe_encode, init_run_state, transformer, argmax, softmax, sample, time_in_ms

def run_model(checkpoint, temperature, steps, prompt):
    rng_seed = int(time.time())
    random.seed(rng_seed)

    weights = TransformerWeights()

    with open(checkpoint, "rb") as file:
        _config = file.read(struct.calcsize('7i'))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', _config)
        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)

        shared_weights = 1 if config.vocab_size > 0 else 0
        config.vocab_size = abs(config.vocab_size)

        checkpoint_init_weights(weights, config, file, shared_weights)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    with open("tokenizer.bin", "rb") as file:
        vocab, vocab_scores, max_token_length = tokenizer_init(config, file)

    state = RunState()
    init_run_state(state, config)

    prompt_tokens = []
    if prompt:
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores)

    start = 0  
    next_token = 0  
    token = 1
    pos = 0  
    
    output = "<s>\n"

    while pos < steps:
        transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            if temperature == 0.0:
                next_token = argmax(state.logits)
            else:
                state.logits = [i / temperature for i in state.logits]
                softmax(state.logits, config.vocab_size)
                next_token = sample(state.logits)

        token_str = (
            vocab[next_token].lstrip()
            if token == 1 and vocab[next_token][0] == ' ' else vocab[next_token]
        )

        output += token_str
        if next_token == 1:
            break

        token = next_token
        pos += 1

        if start == 0:
            start = time_in_ms()

    end = time_in_ms()
    output += f"\nachieved tok/s: {(steps - 1) / (end - start) * 1000}"
    return output

# Gradio interface setup
def gradio_interface(checkpoint, temperature, steps, prompt):
    return run_model(checkpoint, float(temperature), int(steps), prompt)

checkpoint_input = gr.Textbox(label="Checkpoint Path", value='./out/stories15M.bin')
temperature_input = gr.Textbox(label="Temperature", value="0.0")
steps_input = gr.Textbox(label="Steps", value="256")
prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", value=None)

interface = gr.Interface(fn=gradio_interface,
                         inputs=[checkpoint_input, temperature_input, steps_input, prompt_input],
                         outputs="text",
                         title="Baby Llama Transformer",
                         description="Run a Baby Llama transformer model by providing checkpoint, temperature, steps, and prompt inputs.")

interface.launch()
