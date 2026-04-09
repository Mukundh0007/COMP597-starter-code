from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_v2"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(type=int, help="The run number used for codecarbon_v2 file tracking.", default=0)
        self._arg_project_name = _Arg(type=str, help="The name of the project used for codecarbon_v2 file tracking.", default="energy-efficiency")
        self._arg_output_dir = _Arg(type=str, help="The path of the output directory where files will be saved.", default=".")
        self._arg_sync_every_steps = _Arg(type=int, help="Measure per-phase timings with CUDA sync every N steps (>=1).", default=5)
        self._arg_measure_power_secs = _Arg(type=float, help="CodeCarbon power sampling interval in seconds.", default=0.5)
