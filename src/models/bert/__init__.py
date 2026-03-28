from src.models.bert.bert import bert_init
import src.config as config
import src.trainer as trainer
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "bert"

def init_model(conf: config.Config, dataset: data.Dataset):
    return bert_init(conf, dataset)