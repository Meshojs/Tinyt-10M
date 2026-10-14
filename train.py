import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from model import TinyT


class ModelConfig:
    d_model: int = 192
    n_blocks: int = 4
    seq_len: int = 256
    n_heads: int = 4
    d_ff: int = 768
    vocab_size: int = 12000
    dropout: float = 0.0


class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)

    max_steps: int = 12208
    warmup_steps: int = 300

    batch_size: int = 32
    min_lr: float = 3e-5


model_config = ModelConfig()
training_config = TrainingConfig()
model = TinyT(
    vocab_size=model_config.vocab_size,
    d_model=model_config.d_model,
    n_blocks=model_config.n_blocks,
    seq_len=model_config.seq_len,
    n_heads=model_config.n_heads,
    d_ff=model_config.d_ff,
    dropout=model_config.dropout,
)


class Training(nn.Module):
    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        model: TinyT,
    ):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.model = model
        self.data = np.memmap("./data/data.bin", dtype=np.uint16, mode="r")
        self.c = nn.CrossEntropyLoss()
        self.optimizer = model.optim_config(
            betas=training_config.betas,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        warm = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=training_config.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.max_steps - training_config.warmup_steps,
            eta_min=training_config.min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warm, cosine], milestones=[training_config.warmup_steps]
        )

    def _iter_token_chunks(self, path, chunk_bytes=50_000_000):
        tok = Tokenizer.from_file("./tinyt_tokenizer.json")
        with open(path, encoding="utf-8") as f:
            while lines := f.readlines(chunk_bytes):  # := , if not line : break
                enc = tok.encode_batch(lines)
                yield np.fromiter((i for e in enc for i in e.ids), dtype=np.uint16)

    def _build_bin(self):  # one-time run -> "./data/data.bin"
        with open("./data/data.bin", "wb") as out:
            for chunk in self._iter_token_chunks("./data/data.txt"):
                chunk.tofile(out)

    def get_batch(self):
        seq_len = self.model_config.seq_len
        ix = np.random.randint(
            0, len(self.data) - seq_len - 1, size=self.training_config.batch_size
        )
        x = np.stack([self.data[i : i + seq_len] for i in ix])
        y = np.stack([self.data[i + 1 : i + seq_len + 1] for i in ix])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(
            y.astype(np.int64)
        )

    def fit(self):
        device = next(self.model.parameters()).device
        self.model.train()

        pbar = tqdm(range(self.training_config.max_steps))

        for step in pbar:
            x, y = self.get_batch()
            x = x.to(device)
            y = y.to(device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                logits = self.model(x)
                loss = self.c(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": step + 1})

            if (step + 1) % 2000 == 0 or step == 0:
                torch.save(
                    self.model.state_dict(), f"./checkpoints/checkpoint-{step + 1}.pt"
                )


t = Training(training_config, model_config, model).fit()
print(t)
