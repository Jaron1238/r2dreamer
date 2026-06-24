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

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    
    
    if str(getattr(config, "device", "cpu")) == "mps":
        config.trainer.num_workers = 0
        print("[train] MPS device detected → DataLoader num_workers forced to 0")

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
                print(
                    f"[WARNUNG] Checkpoint Phase {ckpt_phase} != Config Phase {config.phase}. "
                    "Fortsetzen im Curriculum-Modus mit geladenen Gewichten."
                )
            agent.load_state_dict(ckpt["model"])
            print(f"Checkpoint geladen: {ckpt_path} (Phase {ckpt_phase})")

    ran_mlx = False
    if bool(getattr(config, "use_mlx", False)):
        import mlx.core as mx
        from native.mlx_models import MLXDreamer
        from native.mlx_utils import load_pytorch_to_mlx
        from native.mlx_trainer import MLXOnlineTrainer, _mx_save
        from mlx.utils import tree_flatten

        mlx_agent = MLXDreamer(config)
        mlx_params = dict(tree_flatten(mlx_agent.parameters()))
        converted, report = load_pytorch_to_mlx(mlx_params, agent.state_dict())
        print(
            f"[MLX] Weight transfer: loaded={report['loaded']}  "
            f"skipped={report['skipped']}  missing={report['missing']}"
        )
        if report["skipped"] or report["missing"]:
            print("[MLX] Skipped/missing details:", report["details"])
        mlx_agent.load_weights(list(converted.items()), strict=False)
        mx.eval(mlx_agent.parameters())
        trainer = MLXOnlineTrainer(config.trainer, logger, logdir)
        if int(getattr(config, "phase", 1)) >= 3:
            from envs.drone_sim import DroneSimEnv
            env = DroneSimEnv(config)
            trainer.begin(mlx_agent, env)
        else:
            trainer.begin(mlx_agent)
        _mx_save(logdir / "latest.safetensors", mlx_agent)
        ran_mlx = True
    elif int(getattr(config, "phase", 1)) >= 3:
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
        trainer = OfflineTrainer(config.trainer, dataset, logger, logdir, phase=int(getattr(config, 'phase', 1)))  # Bug #7 fix
        trainer.begin(agent)

    if not ran_mlx:
        raw_agent = trainer.accelerator.unwrap_model(agent) if hasattr(trainer, "accelerator") else agent
        torch.save(
            {"model": raw_agent.state_dict(), "phase": config.phase},
            logdir / "latest.pt",
        )

if __name__ == "__main__":
    main()

  
