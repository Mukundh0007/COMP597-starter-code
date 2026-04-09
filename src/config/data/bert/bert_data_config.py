"""Config for BERT synthetic data (milabench-style). See data_configs.bert.* in --help."""
from src.config.util.base_config import _Arg, _BaseConfig


class DataConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self._arg_vocab_size = _Arg(type=int, help="Vocabulary size for synthetic tokens.", default=30522)
        self._arg_train_length = _Arg(type=int, help="Sequence length (train_length, like milabench).", default=512)
        self._arg_n = _Arg(type=int, help="Number of unique synthetic samples (milabench n). 0 => use batch_size.", default=0)
        self._arg_repeat = _Arg(type=int, help="Repeat factor: dataset length = n * repeat (milabench repeat). Use 100000 for full milabench-length run.", default=100000)
        self._arg_seed = _Arg(type=int, help="Random seed for synthetic sample generation and BERT MLM masking/model init reproducibility.", default=42)
