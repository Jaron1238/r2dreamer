import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from dreamer import Dreamer
from trainer import FPVDataset, OfflineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)
    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    dataset = FPVDataset(
        config,
        batch_length=config.trainer.batch_length,
        require_osd=bool(config.dataset.get("require_osd", False)),
    )

    agent = Dreamer(config.model).to(config.device)

    if config.get("checkpoint", None):
        ckpt_path = pathlib.Path(config.checkpoint).expanduser()
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=config.device)
            agent.load_state_dict(ckpt["model"])
            print(f"Checkpoint geladen: {ckpt_path}")

    trainer = OfflineTrainer(config.trainer, dataset, logger, logdir)
    trainer.begin(agent)

    torch.save(
        {"model": agent.state_dict()},
        logdir / "latest.pt",
    )


if __name__ == "__main__":
    main()

  
