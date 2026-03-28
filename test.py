from src.models.bert.bert import pre_init_bert
import torch
import src.config as config
from types import SimpleNamespace

# 1. Set up a minimal conf + args
conf = config.Config()  # or your existing config object
args = SimpleNamespace(batch_size=2)  # tiny batch for sanity test

# 2. Initialize model, dataset, tokenizer, data collator
model, dataset, tokenizer, data_collator = pre_init_bert(conf, args)

# 3. Create DataLoader and fetch first batch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
batch = next(iter(loader))

# 4. Move batch to GPU
batch = {k: v.to(model.device) for k, v in batch.items()}

# 5. Forward pass & print loss
loss = model(**batch).loss
print("Sanity check loss:", loss.item())