import os
import yaml
from pydantic import BaseModel

ROOT_DIR = '/'.join((s for s in os.path.dirname(__file__).split('/')[:-2]))
LOG_DIR = os.path.join(ROOT_DIR, "logs/log")

class DataConfig(BaseModel):
    data_root: str

class GPTConfig(BaseModel):
    block_size: int # max sequence length
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


class TrainConfig(BaseModel):
    random_state: int
    batch_size: int
    token_length: int
    max_lr: float
    warmup_steps: int
    max_steps: int


class Config(BaseModel):
    gpt_config: GPTConfig
    train_config: TrainConfig
    data_config: DataConfig


def create_config() -> dict:
    """"
    Run validation on config values.
    """
    CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(CONFIG_FILE_PATH, 'r') as f:
        parsed_config = yaml.load(f, Loader=yaml.Loader)

    _config = Config(
        gpt_config=GPTConfig(**parsed_config),
        train_config=TrainConfig(**parsed_config),
        data_config=DataConfig(**parsed_config)
    )
    return _config

