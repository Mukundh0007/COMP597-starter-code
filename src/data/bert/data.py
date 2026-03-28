"""
Synthetic data for BERT (AutoModelForMaskedLM), milabench-style.

Structure follows https://github.com/mila-iqia/milabench/blob/255b759c787f378a35454f2a271a7e2f35dc0f5a/benchmarks/huggingface/bench/synth.py
with a load_data(conf) interface for COMP597. The dataset pre-generates n samples
and repeats them (__len__ = n * repeat, __getitem__(i) = data[i % n]) so it is
deterministic and matches milabench's SyntheticData behaviour.
Outputs input_ids and attention_mask; DataCollatorForLanguageModeling adds labels.
"""

import torch
import torch.utils.data as data
import src.config as config

data_load_name = "bert"


def _make_generators(vocab_size: int, train_length: int):
    """Generators for MaskedLM-style synthetic data (milabench gen_AutoModelForMaskedLM)."""

    def gen_input_ids():
        return torch.randint(0, vocab_size, (train_length,), dtype=torch.long)

    def gen_attention_mask():
        return torch.ones(train_length, dtype=torch.long)

    return {
        "input_ids": gen_input_ids,
        "attention_mask": gen_attention_mask,
    }


class SyntheticData(data.Dataset):
    """
    Milabench-style synthetic dataset: n unique samples, repeated so len = n * repeat.
    Each sample is a dict of tensors produced by the generators.
    """

    def __init__(self, generators: dict, n: int, repeat: int):
        self.n = n
        self.repeat = repeat
        self.generators = generators
        self.data = [self._gen() for _ in range(n)]

    def _gen(self):
        return {name: gen() for name, gen in self.generators.items()}

    def __getitem__(self, i: int):
        return self.data[i % self.n]

    def __len__(self) -> int:
        return self.n * self.repeat


def load_data(conf: config.Config) -> data.Dataset:
    """
    Load synthetic BERT dataset (milabench-style) from config.

    Uses data_configs.bert when available (vocab_size, train_length, n, repeat).
    If n is 0, uses conf.batch_size for n so number of unique samples = batch_size.
    """
    vocab_size = 30522
    train_length = 512
    n = 4
    repeat = 100000 #needs to change as required

    if hasattr(conf, "data_configs") and hasattr(conf.data_configs, "bert"):
        b = conf.data_configs.bert
        vocab_size = getattr(b, "vocab_size", vocab_size)
        train_length = getattr(b, "train_length", train_length)
        n = getattr(b, "n", n)
        repeat = getattr(b, "repeat", repeat)
    if n <= 0:
        n = getattr(conf, "batch_size", 4)
    print(n)
    generators = _make_generators(vocab_size, train_length)
    return SyntheticData(generators=generators, n=n, repeat=repeat)
