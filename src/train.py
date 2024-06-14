import logging
import numpy as np
from preprocess.manager_data import CustomDataLoader
from config import setup
from config.setup import create_config

FORMAT = "%(asctime)s %(filename)s %(message)s"
logging.basicConfig(filename=setup.LOG_DIR, level=logging.INFO, format=FORMAT)

def train():
    """Fuction to train the model"""

    _config = create_config()
    logging.info(f"{_config}")
    # train_loader = CustomDataLoader(data_root=, master_process=, batch_size=, token_length=,
    #                                 process_rank=, num_processes=, split='train')
    # val_loader = CustomDataLoader(data_root=, master_process=, batch_size=, token_length=,
    #                                 process_rank=, num_processes=, split='train')

if __name__ == "__main__":
    np.random.seed(42)
    train()