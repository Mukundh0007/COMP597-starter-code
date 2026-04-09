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
import random
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


def _seed_bert_rng(conf: config.Config) -> None:
    seed = 42
    if hasattr(conf, "data_configs") and hasattr(conf.data_configs, "bert"):
        seed = getattr(conf.data_configs.bert, "seed", seed)

    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def pre_init_bert(conf: config.Config, dataset: data.Dataset):
    _seed_bert_rng(conf)
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

    def __init__(
        self,
        device: torch.device,
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        sync_every_steps: int = 5,
        metrics_interval_s: float = 0.5,
    ):
        super().__init__(device=device)
        self.output_dir = output_dir or os.getcwd()
        self.batch_size = batch_size
        self.sync_every_steps = max(1, int(sync_every_steps))
        self.losses = []
        self.sampled_steps = []
        self.step_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        # GPU utilization (pynvml), GPU memory, system memory (psutil), throughput
        self.gpu_util_pct = []
        self.gpu_mem_util_pct = []
        self.gpu_mem_mb = []
        self.cpu_util_pct = []
        self.sys_mem_pct = []
        self.sys_mem_mb = []
        self.gpu_energy_delta_j = []
        self.gpu_energy_j = []
        self.gpu_power_w = []
        self.throughput_samp_per_s = []
        self.timeline_s = []
        self._train_start_t = None
        self._last_metrics_sample_t = None
        self._metrics_interval_s = max(0.5, float(metrics_interval_s))
        self._nvml_handle = None
        self._pynvml_ok = False
        self._psutil_ok = False
        self._step_idx = 0
        self._timing_enabled_this_step = False
        self._last_loss = None
        self._last_energy_mj = None
        self._energy_base_mj = None
        self._last_energy_t = None
        
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
        if not self._timing_enabled_this_step:
            self._last_loss = None
            return
        if loss is not None and isinstance(loss, torch.Tensor):
            self._last_loss = loss.detach().item()
        else:
            self._last_loss = None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start_step(self) -> None:
        self._step_idx += 1
        self._timing_enabled_this_step = (self._step_idx % self.sync_every_steps) == 0
        if self._timing_enabled_this_step:
            self._sync()
            self.step_stats.start()

    def stop_step(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.step_stats.stop()

    def start_forward(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.forward_stats.start()

    def stop_forward(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.forward_stats.stop()

    def start_backward(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.backward_stats.start()

    def stop_backward(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.backward_stats.stop()

    def start_optimizer_step(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.optimizer_step_stats.start()

    def stop_optimizer_step(self) -> None:
        if self._timing_enabled_this_step:
            self._sync()
            self.optimizer_step_stats.stop()

    def _sample_gpu_system_metrics(self) -> None:
        """Sample GPU/system metrics at most once every 500ms (NVML-friendly cadence)."""
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

                energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle)
                if self._energy_base_mj is None:
                    self._energy_base_mj = energy_mj
                    self._last_energy_mj = energy_mj
                    self._last_energy_t = now
                    self.gpu_energy_delta_j.append(0.0)
                    self.gpu_energy_j.append(0.0)
                    self.gpu_power_w.append(0.0)
                else:
                    if self._last_energy_mj is None or energy_mj < self._last_energy_mj:
                        self._last_energy_mj = energy_mj
                        self._last_energy_t = now
                        self.gpu_energy_delta_j.append(0.0)
                        self.gpu_energy_j.append(max((energy_mj - self._energy_base_mj) / 1000.0, 0.0))
                        self.gpu_power_w.append(0.0)
                    else:
                        dt = now - self._last_energy_t if self._last_energy_t is not None else self._metrics_interval_s
                        if dt <= 0:
                            dt = self._metrics_interval_s
                        delta_j = (energy_mj - self._last_energy_mj) / 1000.0
                        cumulative_j = (energy_mj - self._energy_base_mj) / 1000.0
                        self.gpu_energy_delta_j.append(max(delta_j, 0.0))
                        self.gpu_energy_j.append(max(cumulative_j, 0.0))
                        self.gpu_power_w.append(max(delta_j / dt, 0.0))
                        self._last_energy_mj = energy_mj
                        self._last_energy_t = now
            except Exception:
                self.gpu_util_pct.append(-1)
                self.gpu_mem_util_pct.append(-1)
                self.gpu_mem_mb.append(-1.0)
                self.gpu_energy_delta_j.append(-1.0)
                self.gpu_energy_j.append(-1.0)
                self.gpu_power_w.append(-1.0)
        else:
            try:
                self.gpu_mem_mb.append(torch.cuda.memory_allocated(self.device) / (1024 ** 2))
            except Exception:
                self.gpu_mem_mb.append(-1.0)
            self.gpu_util_pct.append(-1)
            self.gpu_mem_util_pct.append(-1)
            self.gpu_energy_delta_j.append(-1.0)
            self.gpu_energy_j.append(-1.0)
            self.gpu_power_w.append(-1.0)
        
        if self._psutil_ok:
            try:
                self.cpu_util_pct.append(psutil.cpu_percent(interval=None))
                vmem = psutil.virtual_memory()
                self.sys_mem_pct.append(vmem.percent)
                self.sys_mem_mb.append(vmem.used / (1024 ** 2))
            except Exception:
                self.cpu_util_pct.append(-1)
                self.sys_mem_pct.append(-1)
                self.sys_mem_mb.append(-1.0)
        else:
            self.cpu_util_pct.append(-1)
            self.sys_mem_pct.append(-1)
            self.sys_mem_mb.append(-1.0)

    def log_step(self) -> None:
        if self._timing_enabled_this_step:
            ns = 1e9
            self.sampled_steps.append(self._step_idx)
            self.step_times.append(self.step_stats.get_last() / ns)
            self.forward_times.append(self.forward_stats.get_last() / ns)
            self.backward_times.append(self.backward_stats.get_last() / ns)
            self.optimizer_times.append(self.optimizer_step_stats.get_last() / ns)
            self.losses.append(self._last_loss)
            step_s = self.step_times[-1]
            self.throughput_samp_per_s.append(self.batch_size / step_s if step_s > 0 else 0.0)
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
                loss_val = self.losses[i] if i < len(self.losses) else None
                loss_str = f"{loss_val:.6f}" if loss_val is not None else ""
                thr = f"{self.throughput_samp_per_s[i]:.4f}" if i < len(self.throughput_samp_per_s) else ""
                f.write(
                    f"{self.sampled_steps[i]},{loss_str},{self.step_times[i]:.4f},{self.forward_times[i]:.4f},"
                    f"{self.backward_times[i]:.4f},{self.optimizer_times[i]:.4f},{thr}\n"
                )

        timeline_csv_path = os.path.join(self.output_dir, "bert_run_timeline.csv")
        tn = len(self.timeline_s)
        with open(timeline_csv_path, "w") as f:
            f.write("timeline_s,gpu_util_pct,gpu_mem_util_pct,gpu_mem_mb,cpu_util_pct,sys_mem_pct,sys_mem_mb,gpu_energy_delta_j,gpu_energy_j,gpu_power_w\n")
            for i in range(tn):
                f.write(
                    f"{self.timeline_s[i]:.4f},{self.gpu_util_pct[i]},{self.gpu_mem_util_pct[i]},"
                    f"{self.gpu_mem_mb[i]:.2f},{self.cpu_util_pct[i]},{self.sys_mem_pct[i]},{self.sys_mem_mb[i]:.2f},"
                    f"{self.gpu_energy_delta_j[i]:.6f},{self.gpu_energy_j[i]:.6f},{self.gpu_power_w[i]:.6f}\n"
                )
        print(f"BERT results written to {csv_path}")
        print(f"BERT timeline written to {timeline_csv_path}")

class BertSimpleTrainer(trainer.SimpleTrainer):
    """SimpleTrainer with 5-minute time-bounded training loop."""

    MAX_DURATION_S = 5*60  #5 minutes per experiment protocol

    def forward(self, i, batch, model_kwargs):
        self.optimizer.zero_grad()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch, **model_kwargs)
        return outputs.loss

    def train(self, model_kwargs):
        progress_bar = tqdm.auto.tqdm(range(len(self.loader)), desc="training")
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

            if ((i + 1) % 50) == 0:
                progress_bar.set_description(f"step: {i + 1}")

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
        stats_output_dir = os.environ.get("BERT_STATS_OUTPUT_DIR")
        sync_every_steps = int(os.environ.get("BERT_SYNC_EVERY_STEPS", "5"))
        metrics_interval_s = float(os.environ.get("BERT_METRICS_INTERVAL_S", "0.5"))
        stats = BertTrainerStats(
            device=model.device,
            batch_size=conf.batch_size,
            output_dir=stats_output_dir,
            sync_every_steps=sync_every_steps,
            metrics_interval_s=metrics_interval_s,
        )
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