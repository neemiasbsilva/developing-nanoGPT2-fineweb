import logging
import numpy as np
import os
from preprocess.manager_data import CustomDataLoader
from config import setup
from config.setup import create_config
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP



FORMAT = "%(asctime)s %(filename)s %(message)s"
logging.basicConfig(filename=setup.LOG_DIR, level=logging.INFO, format=FORMAT)

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
        ddp_world_rank = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # vanilla
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_rank = 1
        master_process = True
        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        logging.info(f"Using device: {device}")

    train_loader = CustomDataLoader(data_root=config.data_config.data_root, master_process=master_process, batch_size=config.train_config.batch_size, 
                                    token_length=config.train_config.token_length, process_rank=ddp_rank, num_processes=ddp_world_rank, split='train')
    val_loader = CustomDataLoader(data_root=config.data_config.data_root, master_process=master_process, batch_size=config.train_config.batch_size, 
                                  token_length=config.train_config.token_length, process_rank=ddp_rank, num_processes=ddp_world_rank, split='val')

if __name__ == "__main__":
    np.random.seed(42)
    train()