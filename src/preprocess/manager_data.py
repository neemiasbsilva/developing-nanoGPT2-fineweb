import os
import logging
import numpy as np
import torch
from config import setup


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class CustomDataLoader:
    def __init__(self, data_root, master_process, batch_size, sequence_length, process_rank, num_processes, split):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        data_root = os.path.join(setup.ROOT_DIR, data_root)
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            logging.info(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def reset(self):
        # state init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.sequence_length * self.process_rank
    
    def next_batch(self):
        batch_size, sequence_length = self.batch_size, self.sequence_length
        chunk = self.tokens[self.current_position: self.current_position + batch_size * sequence_length + 1]
        x = (chunk[:-1]).view(batch_size, sequence_length) # inputs
        y = (chunk[1:]).view(batch_size, sequence_length)

        self.current_position += batch_size * self.sequence_length * self.num_processes

        if self.current_position + (batch_size * sequence_length + 1) > len(self.tokens):
            self.current_position = self.batch_size * self.sequence_length * self.process_rank
        return x, y