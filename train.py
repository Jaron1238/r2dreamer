import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from dreamer import Dreamer
from trainer import FPVDataset, OfflineTrainer, OnlineTrainer

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

    agent = Dreamer(config).to(config.device)

    if config.get("checkpoint", None):
        ckpt_path = pathlib.Path(config.checkpoint).expanduser()
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=config.device)
            ckpt_phase = ckpt.get("phase", None)
            if ckpt_phase is not None and int(ckpt_phase) != int(config.phase):
                raise ValueError(
                    f"Checkpoint Phase {ckpt_phase} != Config Phase {config.phase}. "
                    f"Entweder phase: {ckpt_phase} in der Config setzen oder Checkpoint prüfen."
                )
            agent.load_state_dict(ckpt["model"])
            print(f"Checkpoint geladen: {ckpt_path} (Phase {ckpt_phase})")

    if int(getattr(config, "phase", 1)) >= 3:
        from envs.drone_sim import DroneSimEnv
        trainer = OnlineTrainer(config.trainer, logger, logdir)
        env = DroneSimEnv(config)
        trainer.begin(agent, env)
    else:
        dataset = FPVDataset(
            config,
            batch_length=config.trainer.batch_length,
            require_osd=bool(config.dataset.get("require_osd", False)),
        )
        trainer = OfflineTrainer(config.trainer, dataset, logger, logdir)
        trainer.begin(agent)

    torch.save(
        {"model": agent.state_dict(), "phase": config.phase},
        logdir / "latest.pt",
    )


if __name__ == "__main__":
    main()

  
