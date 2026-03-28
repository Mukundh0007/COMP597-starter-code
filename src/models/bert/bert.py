"""
BERT (AutoModelForMaskedLM) for workload evaluation, milabench-aligned.

Model setup follows https://github.com/mila-iqia/milabench/blob/255b759c787f378a35454f2a271a7e2f35dc0f5a/benchmarks/huggingface/bench/models.py
(Bert: BertConfig(), train_length=512, category AutoModelForMaskedLM).
Use with synthetic data (--data bert) and:
  - --trainer_stats simple   → loss, per-step timing, GPU utilization (pynvml), GPU memory,
    system memory (psutil), throughput (samples/sec); CSV + plots in cwd.
  - --trainer_stats codecarbon → GPU power/energy and emissions (and loss CSV).
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from typing import Dict, Optional, Tuple
import tqdm.auto

import transformers
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer

import src.trainer as trainer
import src.trainer.stats as trainer_stats
import src.trainer.stats.simple as simple_stats
import src.config as config

import pynvml
import psutil  
import matplotlib
import matplotlib.pyplot as plt

def init_bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def pre_init_bert(conf: config.Config, dataset: data.Dataset):
    tokenizer = init_bert_tokenizer()

    model_config = BertConfig()
    model = BertForMaskedLM(config=model_config)

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    return model, dataset, tokenizer, data_collator


class BertTrainerStats(simple_stats.SimpleTrainerStats):
    """Collects loss, timings, GPU utilization (pynvml), GPU memory, system memory (psutil),
    and throughput (samples/sec). Writes CSV and plots to output_dir (default cwd).
    """

    def __init__(self, device: torch.device, output_dir: Optional[str] = None, batch_size: int = 1):
        super().__init__(device=device)
        self.output_dir = output_dir or os.getcwd()
        self.batch_size = batch_size
        self.losses = []
        self.step_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        # GPU utilization (pynvml), GPU memory, system memory (psutil), throughput
        self.gpu_util_pct = []
        self.gpu_mem_util_pct = []
        self.gpu_mem_mb = []
        self.sys_mem_pct = []
        self.sys_mem_mb = []
        self.throughput_samp_per_s = []
        self.timeline_s = []
        self._train_start_t = None
        self._last_metrics_sample_t = None
        self._metrics_interval_s = 0.5
        self._nvml_handle = None
        self._pynvml_ok = False
        self._psutil_ok = False
        if device.type == "cuda":
            try:
                pynvml.nvmlInit()
                gpu_index = device.index if device.index is not None else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._pynvml_ok = True
            except Exception:
                pass
        try:
            self._psutil_ok = True
        except Exception:
            pass

    def log_loss(self, loss: torch.Tensor) -> None:
        super().log_loss(loss)
        if loss is not None and isinstance(loss, torch.Tensor):
            self.losses.append(loss.detach().item())

    def _sample_gpu_system_metrics(self) -> None:
        """Sample GPU/system metrics at most once every 500ms (NVML-friendly cadence)."""
        n = len(self.step_times)
        if n == 0:
            return
        step_s = self.step_times[-1]
        self.throughput_samp_per_s.append(self.batch_size / step_s if step_s > 0 else 0.0)
        now = time.perf_counter()
        if self._train_start_t is None:
            self._train_start_t = now
        if self._last_metrics_sample_t is not None and (now - self._last_metrics_sample_t) < self._metrics_interval_s:
            return
        self._last_metrics_sample_t = now
        self.timeline_s.append(now - self._train_start_t)

        if self._pynvml_ok and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                self.gpu_util_pct.append(util.gpu)
                self.gpu_mem_util_pct.append(util.memory)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                self.gpu_mem_mb.append(mem.used / (1024 ** 2))
            except Exception:
                self.gpu_util_pct.append(-1)
                self.gpu_mem_util_pct.append(-1)
                self.gpu_mem_mb.append(-1.0)
        else:
            try:
                self.gpu_mem_mb.append(torch.cuda.memory_allocated(self.device) / (1024 ** 2))
            except Exception:
                self.gpu_mem_mb.append(-1.0)
            self.gpu_util_pct.append(-1)
            self.gpu_mem_util_pct.append(-1)
        if self._psutil_ok:
            try:
                vmem = psutil.virtual_memory()
                self.sys_mem_pct.append(vmem.percent)
                self.sys_mem_mb.append(vmem.used / (1024 ** 2))
            except Exception:
                self.sys_mem_pct.append(-1)
                self.sys_mem_mb.append(-1.0)
        else:
            self.sys_mem_pct.append(-1)
            self.sys_mem_mb.append(-1.0)

    def log_step(self) -> None:
        super().log_step()
        ns = 1e9
        self.step_times.append(self.step_stats.get_last() / ns)
        self.forward_times.append(self.forward_stats.get_last() / ns)
        self.backward_times.append(self.backward_stats.get_last() / ns)
        self.optimizer_times.append(self.optimizer_step_stats.get_last() / ns)
        self._sample_gpu_system_metrics()

    def log_stats(self) -> None:
        super().log_stats()
        self._write_results_and_plot()

    def _write_results_and_plot(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "bert_run_results.csv")
        n = len(self.step_times)
        with open(csv_path, "w") as f:
            f.write("step,loss,step_s,forward_s,backward_s,optimizer_s,throughput_samp_per_s\n")
            for i in range(n):
                loss_str = f"{self.losses[i]:.6f}" if i < len(self.losses) else ""
                thr = f"{self.throughput_samp_per_s[i]:.4f}" if i < len(self.throughput_samp_per_s) else ""
                f.write(
                    f"{i},{loss_str},{self.step_times[i]:.4f},{self.forward_times[i]:.4f},"
                    f"{self.backward_times[i]:.4f},{self.optimizer_times[i]:.4f},{thr}\n"
                )

        timeline_csv_path = os.path.join(self.output_dir, "bert_run_timeline.csv")
        tn = len(self.timeline_s)
        with open(timeline_csv_path, "w") as f:
            f.write("timeline_s,gpu_util_pct,gpu_mem_util_pct,gpu_mem_mb,sys_mem_pct,sys_mem_mb\n")
            for i in range(tn):
                f.write(
                    f"{self.timeline_s[i]:.4f},{self.gpu_util_pct[i]},{self.gpu_mem_util_pct[i]},"
                    f"{self.gpu_mem_mb[i]:.2f},{self.sys_mem_pct[i]},{self.sys_mem_mb[i]:.2f}\n"
                )
        print(f"BERT results written to {csv_path}")
        print(f"BERT timeline written to {timeline_csv_path}")

        try:
        #     matplotlib.use("Agg")
        #     steps = list(range(n))
        #     # Figure 1: loss + step timing (existing)
        #     fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))
        #     if self.losses:
        #         axes1[0].plot(steps[: len(self.losses)], self.losses, "b-", alpha=0.8)
        #     axes1[0].set_xlabel("Step")
        #     axes1[0].set_ylabel("Loss")
        #     axes1[0].set_title("Training loss")
        #     axes1[0].grid(True, alpha=0.3)
        #     axes1[1].fill_between(steps, 0, self.forward_times, label="forward", alpha=0.7)
        #     axes1[1].fill_between(
        #         steps,
        #         self.forward_times,
        #         [a + b for a, b in zip(self.forward_times, self.backward_times)],
        #         label="backward",
        #         alpha=0.7,
        #     )
        #     axes1[1].fill_between(
        #         steps,
        #         [a + b for a, b in zip(self.forward_times, self.backward_times)],
        #         self.step_times,
        #         label="optimizer",
        #         alpha=0.7,
        #     )
        #     axes1[1].set_xlabel("Step")
        #     axes1[1].set_ylabel("Time (s)")
        #     axes1[1].set_title("Time per step")
        #     axes1[1].legend()
        #     axes1[1].grid(True, alpha=0.3)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(self.output_dir, "bert_run_plots.png"), dpi=150)
        #     plt.close()

        #     # Figure 2: GPU utilization, GPU memory, system memory, throughput
        #     fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        #     ax = axes2[0, 0]
        #     if self.gpu_util_pct and any(x >= 0 for x in self.gpu_util_pct):
        #         ax.plot(steps[: len(self.gpu_util_pct)], self.gpu_util_pct, "g-", alpha=0.8, label="GPU compute %")
        #     if self.gpu_mem_util_pct and any(x >= 0 for x in self.gpu_mem_util_pct):
        #         ax.plot(steps[: len(self.gpu_mem_util_pct)], self.gpu_mem_util_pct, "m-", alpha=0.8, label="GPU mem %")
        #     ax.set_xlabel("Step")
        #     ax.set_ylabel("%")
        #     ax.set_title("GPU utilization (pynvml)")
        #     ax.legend()
        #     ax.grid(True, alpha=0.3)

        #     ax = axes2[0, 1]
        #     if self.gpu_mem_mb and any(x >= 0 for x in self.gpu_mem_mb):
        #         ax.plot(steps[: len(self.gpu_mem_mb)], self.gpu_mem_mb, "c-", alpha=0.8)
        #     ax.set_xlabel("Step")
        #     ax.set_ylabel("MB")
        #     ax.set_title("GPU memory allocated (torch.cuda / nvml)")
        #     ax.grid(True, alpha=0.3)

        #     ax = axes2[1, 0]
        #     if self.sys_mem_pct and any(x >= 0 for x in self.sys_mem_pct):
        #         ax.plot(steps[: len(self.sys_mem_pct)], self.sys_mem_pct, "orange", alpha=0.8, label="% used")
        #     ax.set_xlabel("Step")
        #     ax.set_ylabel("%")
        #     ax.set_title("System memory (psutil)")
        #     ax.legend()
        #     ax.grid(True, alpha=0.3)

        #     ax = axes2[1, 1]
        #     if self.throughput_samp_per_s:
        #         ax.plot(steps[: len(self.throughput_samp_per_s)], self.throughput_samp_per_s, "r-", alpha=0.8)
        #     ax.set_xlabel("Step")
        #     ax.set_ylabel("samples/sec")
        #     ax.set_title("Throughput")
        #     ax.grid(True, alpha=0.3)

        #     plt.tight_layout()
        #     plot_path2 = os.path.join(self.output_dir, "bert_run_plots_util_throughput.png")
        #     plt.savefig(plot_path2, dpi=150)
        #     plt.close()
            print(f"BERT plots saved to {self.output_dir} (loss+timing, util+throughput)")
        except Exception as e:
            print(f"Could not generate plots (matplotlib may be missing): {e}")


class BertSimpleTrainer(trainer.SimpleTrainer):
    """SimpleTrainer with 5-minute time-bounded training loop."""

    MAX_DURATION_S = 5 * 60  # 5 minutes per experiment protocol

    def forward(self, i, batch, model_kwargs):
        self.optimizer.zero_grad()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch, **model_kwargs)
        return outputs.loss

    def train(self, model_kwargs):
        progress_bar = tqdm.auto.tqdm(range(len(self.loader)), desc="loss: N/A")
        t_start = time.perf_counter()
        completed_steps = 0
        self.stats.start_train()
        for i, batch in enumerate(self.loader):
            if time.perf_counter() - t_start >= self.MAX_DURATION_S:
                print(f"[BERT] 5-minute limit reached after {i} steps. Stopping.", flush=True)
                break
            self.stats.start_step()
            loss, descr = self.step(i, batch, model_kwargs)
            self.stats.stop_step()

            if self.enable_checkpointing and self.should_save_checkpoint(i):
                self.stats.start_save_checkpoint()
                self.save_checkpoint(i)
                self.stats.stop_save_checkpoint()

            self.stats.log_loss(loss)
            self.stats.log_step()

            if loss is not None and isinstance(loss, torch.Tensor):
                loss_val = loss.detach().item()
                progress_bar.set_description(f"loss: {loss_val:.4f}")
                print(f"loss: {loss_val:.4f}", flush=True)
            else:
                progress_bar.set_description("loss: N/A")

            if descr is not None:
                progress_bar.clear()
                print(descr)
            progress_bar.clear()
            progress_bar.update(1)
            completed_steps = i + 1

        self.stats.stop_train()
        progress_bar.close()
        total_elapsed_s = time.perf_counter() - t_start
        print(
            f"[BERT] Total training time: {total_elapsed_s:.2f} s "
            f"({total_elapsed_s / 60:.2f} min) over {completed_steps} steps.",
            flush=True,
        )
        self.stats.log_stats()


def simple_trainer(conf, model, dataset, tokenizer, data_collator):

    loader = data.DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=data_collator,
        num_workers=getattr(conf, "num_workers", 0),
    )
    # # ===== DEBUG: inspect first batch =====
    # for batch in loader:
    #     print("===== FIRST BATCH KEYS =====")
    #     print(batch.keys())
    #     print("===== FIRST BATCH SHAPES =====")
    #     for k, v in batch.items():
    #         print(f"{k}: {v.shape}")
    #     break  # only first batch
    model = model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate)

    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )

    # Use BertTrainerStats for loss + timing + GPU util/memory/throughput + plots; use codecarbon for power/energy.
    if getattr(conf, "trainer_stats", "simple") == "simple":
        stats = BertTrainerStats(device=model.device, batch_size=conf.batch_size)
    else:
        stats = trainer_stats.init_from_conf(conf=conf, device=model.device, num_train_steps=len(loader))

    return BertSimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=model.device,
        stats=stats,
    ), None


def bert_init(conf: config.Config, dataset):
    model, dataset, tokenizer, data_collator = pre_init_bert(conf, dataset)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, data_collator)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")