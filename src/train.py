import logging
import numpy as np
import math
import os
import time
from preprocess.manager_data import CustomDataLoader
from config import setup
from config.setup import create_config
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
from model.gpt2_model import GPT2


FORMAT = "%(asctime)s %(filename)s %(message)s"
logging.basicConfig(filename=setup.LOG_DIR, level=logging.INFO, format=FORMAT)

def run_eval():
    pass


def get_learning_rate(it, warmup_steps, max_lr, min_lr, max_steps):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > warmup_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def run_train():
    pass


def train():
    """Fuction to train the model"""

    config = create_config()
    logging.info(f"{config}")

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        # run according to rank
        assert torch.cuda.is_available(), "need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # vanilla
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        logging.info(f"Using device: {device}")
    device_type = "cuda" if device.startswith("cuda") else "cpu" 
    encoder = tiktoken.get_encoding("gpt2")

    total_batch_size = config.train_config.total_batch_size
    batch_size = config.train_config.batch_size
    sequence_length  = config.train_config.sequence_length
    assert total_batch_size % (batch_size * sequence_length * ddp_world_size)
    grad_accum_steps = total_batch_size // (batch_size * sequence_length * ddp_world_size)
    if master_process:
        logging.info(f"Total desired batch size: {total_batch_size}")
        logging.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = CustomDataLoader(data_root=config.data_config.data_root, master_process=master_process, batch_size=batch_size, 
                                    sequence_length=sequence_length, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = CustomDataLoader(data_root=config.data_config.data_root, master_process=master_process, batch_size=batch_size, 
                                  sequence_length=sequence_length, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
    
    torch.set_float32_matmul_precision('high')

    model = GPT2(config.gpt_config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    max_lr = config.train_config.max_lr
    min_lr = max_lr * 0.1
    warmup_steps = config.train_config.warmup_steps
    max_steps = config.train_config.max_steps
    weight_decay = config.train_config.weight_decay

    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device)

    log_dir = os.path.join(setup.ROOT_DIR, 'models')
    log_file = os.path.join(setup.ROOT_DIR, 'logs/train_log.txt')
    with open(log_file, 'w') as f: 
        pass
    
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps -1)

        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                logging.info(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)


        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_learning_rate(step, warmup_steps, max_lr, min_lr, max_steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.batch_size * train_loader.sequence_length * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            logging.info(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
                
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    np.random.seed(42)
    train()