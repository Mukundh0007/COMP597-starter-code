"""
CodeCarbon v2: Minimal end-to-end energy tracking for E2.

This implementation tracks ONLY full-run energy consumption with minimal overhead:
- Single OfflineEmissionsTracker for total training duration
- No per-step or per-substep task tracking
- Minimal CUDA synchronization (start/stop of full run only)
- Single CSV output at end
- <5% overhead target vs E1 baseline (no-op)

Protocol: E2 end-to-end energy measurement.
"""

from codecarbon import OfflineEmissionsTracker
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData
import codecarbon.core.cpu
import logging
import os
import csv
import pandas as pd
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import torch

logger = logging.getLogger(__name__)

# artificially force psutil to fail, so that CodeCarbon uses constant mode for CPU measurements
codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name = "codecarbon_v2"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to codecarbon_v2 trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return CodeCarbonStatsV2(
        device, 
        conf.trainer_stats_configs.codecarbon_v2.run_num, 
        conf.trainer_stats_configs.codecarbon_v2.project_name, 
        conf.trainer_stats_configs.codecarbon_v2.output_dir
    )


class SimpleFileOutput(BaseOutput):
    """Minimal CSV output handler."""

    def __init__(self, output_file_name: str = "codecarbon.csv", output_dir: str = "."):
        self.output_file_name: str = output_file_name
        if not os.path.exists(output_dir):
            raise OSError(f"Folder '{output_dir}' doesn't exist !")
        self.output_dir: str = output_dir
        self.save_file_path = os.path.join(self.output_dir, self.output_file_name)
        logger.info(
            f"Emissions data will be saved to file {os.path.abspath(self.save_file_path)}"
        )

    def has_valid_headers(self, data: EmissionsData) -> bool:
        with open(self.save_file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dict_from_csv = dict(list(csv_reader)[0])
            list_of_column_names = list(dict_from_csv.keys())
            return list(data.values.keys()) == list_of_column_names

    def to_csv(self, total: EmissionsData, delta: EmissionsData) -> None:
        """Save full-run emissions data to CSV. Append if file exists."""
        file_exists: bool = os.path.isfile(self.save_file_path)

        if file_exists and not self.has_valid_headers(total):
            logger.warning("CSV format changed, backing up old file.")
            os.rename(self.save_file_path, self.save_file_path + ".bak")
            file_exists = False

        new_df = pd.DataFrame.from_records([dict(total.values)])

        if not file_exists:
            df = new_df
        else:
            df = pd.read_csv(self.save_file_path)
            df = pd.concat([df, new_df], ignore_index=True)

        df.to_csv(self.save_file_path, index=False)

    def out(self, total: EmissionsData, delta: EmissionsData) -> None:
        """Called by stop() to persist full-run emissions."""
        self.to_csv(total, delta)

    def live_out(self, total: EmissionsData, delta: EmissionsData) -> None:
        """No live output for E2."""
        pass

    def task_out(self, data, experiment_name: str) -> None:
        """E2 does not use task-level tracking."""
        pass


class CodeCarbonStatsV2(base.TrainerStats):
    """
    CodeCarbon tracker for E2 with full-run energy + per-step phase timings.

    Measures total energy consumed during training and records timing for:
    - full step
    - forward
    - backward
    - optimizer step

    Timing rows are accumulated in memory and flushed once at the end.

    Parameters
    ----------
    device : torch.device
        GPU device to track.
    run_num : int
        Run identifier (for merging into shared CSV).
    project_name : str
        CodeCarbon project name.
    output_dir : str
        Directory for outputs (cc_full_rank_*.csv).
    """

    def __init__(self, device: torch.device, run_num: int, project_name: str, output_dir: str) -> None:
        self.device = device
        self.run_num = run_num
        self.project_name = project_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._gpu_id = self.device.index if self.device.index is not None else 0

        # Timing uses the same mechanism as SimpleTrainerStats.
        self.step_stats = utils.RunningTimer()
        self.forward_stats = utils.RunningTimer()
        self.backward_stats = utils.RunningTimer()
        self.optimizer_step_stats = utils.RunningTimer()

        # Current-row state
        self._current_step = 0
        self._current_loss = None

        # Per-step timing rows (single flush at end)
        self._rows = []

        run_number = f"run_{run_num}_"

        # Single full-run tracker: covers entire training duration
        self.total_training_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{run_number}cc_full_rank_{self._gpu_id}.csv",
                    output_dir=output_dir,
                )
            ],
            allow_multiple_runs=True,
            log_level="warning",
            gpu_ids=[self._gpu_id],
        )

        logger.info(
            f"CodeCarbonStatsV2 initialized: run_num={run_num}, project={project_name}, "
            f"output_dir={output_dir}, gpu_id={self._gpu_id}"
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _write_timing_csvs(self) -> None:
        run_number = f"run_{self.run_num}_"
        timings_csv = os.path.join(self.output_dir, f"{run_number}timings_rank_{self._gpu_id}.csv")
        summary_csv = os.path.join(self.output_dir, f"{run_number}timing_summary_rank_{self._gpu_id}.csv")

        with open(timings_csv, "w") as f:
            f.write("step,loss,step_s,forward_s,backward_s,optimizer_s\n")
            for row in self._rows:
                loss_str = "" if row["loss"] is None else f"{row['loss']:.6f}"
                f.write(
                    f"{row['step']},{loss_str},{row['step_s']:.6f},{row['forward_s']:.6f},"
                    f"{row['backward_s']:.6f},{row['optimizer_s']:.6f}\n"
                )

        def _mean_std_s(timer: utils.RunningTimer):
            history_ns = timer.stat.history
            if len(history_ns) == 0:
                return 0.0, 0.0
            values_s = torch.tensor(history_ns, dtype=torch.float64) / 1e9
            mean_s = values_s.mean().item()
            std_s = values_s.std(unbiased=False).item()
            return mean_s, std_s

        step_mean, step_std = _mean_std_s(self.step_stats)
        forward_mean, forward_std = _mean_std_s(self.forward_stats)
        backward_mean, backward_std = _mean_std_s(self.backward_stats)
        optimizer_mean, optimizer_std = _mean_std_s(self.optimizer_step_stats)

        with open(summary_csv, "w") as f:
            f.write("num_steps,step_mean_s,step_std_s,forward_mean_s,forward_std_s,backward_mean_s,backward_std_s,optimizer_mean_s,optimizer_std_s\n")
            f.write(
                f"{len(self._rows)},{step_mean:.6f},{step_std:.6f},{forward_mean:.6f},{forward_std:.6f},"
                f"{backward_mean:.6f},{backward_std:.6f},{optimizer_mean:.6f},{optimizer_std:.6f}\n"
            )

        logger.info(f"Per-step timing CSV written: {timings_csv}")
        logger.info(f"Timing summary CSV written: {summary_csv}")

    def start_train(self) -> None:
        """Start full-run tracking. One synchronization at train start."""
        self._sync()
        self.total_training_tracker.start()

    def stop_train(self) -> None:
        """Stop full-run tracking. One synchronization at train end."""
        self._sync()
        self.total_training_tracker.stop()

    def start_step(self) -> None:
        self._current_step += 1
        self._sync()
        self.step_stats.start()

    def stop_step(self) -> None:
        self._sync()
        self.step_stats.stop()

    def start_forward(self) -> None:
        self._sync()
        self.forward_stats.start()

    def stop_forward(self) -> None:
        self._sync()
        self.forward_stats.stop()

    def start_backward(self) -> None:
        self._sync()
        self.backward_stats.start()

    def stop_backward(self) -> None:
        self._sync()
        self.backward_stats.stop()

    def start_optimizer_step(self) -> None:
        self._sync()
        self.optimizer_step_stats.start()

    def stop_optimizer_step(self) -> None:
        self._sync()
        self.optimizer_step_stats.stop()

    def start_save_checkpoint(self) -> None:
        """E2 does not instrument checkpoint saves."""
        pass

    def stop_save_checkpoint(self) -> None:
        """E2 does not instrument checkpoint saves."""
        pass

    def log_step(self) -> None:
        step_s = self.step_stats.get_last() / 1e9
        forward_s = self.forward_stats.get_last() / 1e9
        backward_s = self.backward_stats.get_last() / 1e9
        optimizer_s = self.optimizer_step_stats.get_last() / 1e9

        if step_s < 0:
            return

        self._rows.append(
            {
                "step": self._current_step,
                "loss": self._current_loss,
                "step_s": step_s,
                "forward_s": max(forward_s, 0.0),
                "backward_s": max(backward_s, 0.0),
                "optimizer_s": max(optimizer_s, 0.0),
            }
        )

    def log_loss(self, loss: torch.Tensor) -> None:
        if loss is None:
            self._current_loss = None
            return
        if isinstance(loss, torch.Tensor):
            self._current_loss = loss.detach().item()
        else:
            self._current_loss = float(loss)

    def log_stats(self) -> None:
        """E2 final logging: persist per-step timing CSVs and full-run energy output."""
        self._write_timing_csvs()
        logger.info(
            f"CodeCarbonStatsV2 complete: run_num={self.run_num}, gpu_id={self._gpu_id}, "
            f"full-run energy and timing data saved to {self.output_dir}"
        )
