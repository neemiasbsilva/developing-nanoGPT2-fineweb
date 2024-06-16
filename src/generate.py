import logging
import os
import torch
import json
from config import setup
from config.setup import create_config, create_eval_config
from model.gpt2_model import GPT2
from torch.nn import functional as F
import tiktoken

def generation_answer():
    config = create_config()
    eval_config = create_eval_config()
    logging.info(f"{config}")

    
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    logging.info(f"Using device: {device}")
    device_type = "cuda" if device.startswith("cuda") else "cpu" 
    encoder = tiktoken.get_encoding("gpt2")

    log_dir = os.path.join(setup.ROOT_DIR, 'models')

    # load checkpoint
    checkpoint_path = os.path.join(log_dir, f"model_{1750:05d}.pt")
    if (os.path.exists(checkpoint_path)):
        checkpoint = torch.load(checkpoint_path, mmap=True)
        config_model = checkpoint['config']
        model_state_dict = checkpoint['model']
        step = checkpoint['step']
        val_loss = checkpoint['val_loss']
        print(step, val_loss)

        model = GPT2(config_model)
        model.to(device)
        model = torch.compile(model)
        model.load_state_dict(model_state_dict, assign=True)
    else:
        model = GPT2(config.gpt_config)
        
    model.eval()

    num_return_sequences = eval_config.eval_config.num_return_sequence
    max_length = eval_config.eval_config.max_length
    tokens = encoder.encode(eval_config.eval_config.message)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen)

            # take the logits at last position
            logits = logits[:, -1, :]
            # get the probabilities
            probs = F.softmax(logits, dim=-1)

            # get top-k sampling of 50 token indices
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)

            xcol = torch.gather(topk_indices, -1, ix)

            xgen = torch.cat((xgen, xcol), dim=1)

    answers = {}
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = encoder.decode(tokens)
        answers[f'{i+1:2d} generation'] = decoded

    reports_filename = os.path.join(setup.ROOT_DIR, 'reports/generation.json')
    with open(reports_filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    
if __name__ == "__main__":
    generation_answer()