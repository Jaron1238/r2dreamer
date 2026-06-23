import contextlib
import io
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init
from torch.utils.tensorboard import SummaryWriter

class Tee(io.TextIOBase):
    

    def __init__(self, *streams):
        super().__init__()
        
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        
        
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        
        return any(hasattr(stream, "isatty") and stream.isatty() for stream in self._streams)

def setup_console_log(logdir, filename="console.log"):
    

    import sys

    
    path = logdir / filename
    f = path.open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    return f

def to_np(x):
    return x.detach().cpu().numpy()

def to_f32(x):
    return x.to(dtype=torch.float32)

def to_i32(x):
    return x.to(dtype=torch.int32)

def weight_init_(m, fan_type="in"):
    
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None:
        return

    if weight.numel() == 0:
        return

    
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)

    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)

class CudaBenchmark:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)

class Logger:
    def __init__(self, logdir, filename="metrics.jsonl"):
        self._logdir = logdir
        self._filename = filename
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def histogram(self, name, value):
        self._histograms[name] = np.array(value)

    def write(self, step, fps=False):
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps/fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / self._filename).open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)
        for name, value in self._histograms.items():
            self._writer.add_histogram(name, value, step)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def log_hydra_config(self, config, name="config", step=0, log_hparams=False, hparams_run_name="."):
        

        
        yaml_str = None
        try:
            from omegaconf import (
                OmegaConf,  
            )

            yaml_str = OmegaConf.to_yaml(config, resolve=True)
        except ImportError:
            
            yaml_str = str(config)
        self._writer.add_text(f"{name}/yaml", f"```yaml\n{yaml_str}\n```", step)

        
        flat = {}
        container = None
        try:
            from omegaconf import OmegaConf  

            container = OmegaConf.to_container(config, resolve=True)
        except Exception:
            container = None

        if log_hparams and container is not None:

            def _flatten(prefix, obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatten(f"{prefix}.{k}" if prefix else k, v)
                elif isinstance(obj, (list, tuple)):
                    flat[prefix] = str(obj)
                elif isinstance(obj, (int, float, bool, str)) or obj is None:
                    flat[prefix] = obj if obj is not None else "null"
                else:
                    flat[prefix] = str(obj)

            _flatten("", container)
            
            with contextlib.suppress(TypeError):
                
                self._writer.add_hparams(flat, {"_": 0}, run_name=hparams_run_name)

def convert(value, precision=32):
    if isinstance(value, dict):
        return {key: convert(val) for key, val in value.items()}
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)

class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count

class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False

def tensorstats(tensor, prefix):
    return {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
    }

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def recursively_collect_optim_state_dict(obj, path="", optimizers_state_dicts=None, visited=None):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    
    if id(obj) in visited:
        return optimizers_state_dicts
    visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update({k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr})
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(attr, new_path, optimizers_state_dicts, visited)
            )
    return optimizers_state_dicts

def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)

def build_module_tree(module: nn.Module, module_name: str = "") -> dict:
    
    
    direct_param_count = 0
    param_details = {}
    for pname, p in module.named_parameters(recurse=False):
        nump = p.numel()
        param_details[pname] = nump
        direct_param_count += nump

    
    children_info = {}
    for cname, child in module.named_children():
        children_info[cname] = build_module_tree(child, cname)

    
    total = direct_param_count + sum(child["total"] for child in children_info.values())

    return {
        "name": module_name,
        "params": param_details,
        "children": children_info,
        "total": total,
    }

def print_module_tree(info: dict, parent_path: str = "", indent: int = 0):
    

    
    name = info["name"]
    if not parent_path:
        full_path = name  
    else:
        if name:  
            full_path = f"{parent_path}/{name}"
        else:
            full_path = parent_path

    
    line = f"{info['total']:11,d} {full_path}"
    print(" " * indent + line)

    
    param_nodes = []
    for param_name, count in info["params"].items():
        param_nodes.append({
            "name": param_name,
            "params": {},
            "children": {},
            "total": count,
        })

    child_nodes = list(info["children"].values())

    
    combined = param_nodes + child_nodes
    combined.sort(key=lambda x: x["total"], reverse=True)

    
    for child_info in combined:
        print_module_tree(child_info, full_path, indent + 2)

def compute_rms(tensors):
    
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)

def compute_global_norm(tensors):
    
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2)

def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x

def print_param_stats(model):
    

    
    stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            mean_val = data.mean().item()
            std_val = data.std(unbiased=False).item()
            l2_val = data.norm().item()
            rms_val = data.pow(2).mean().sqrt().item()

            hierarchical_name = name.replace(".", "/")
            stats.append((hierarchical_name, mean_val, std_val, l2_val, rms_val))

    
    def fmt(v):
        return f"{v:.3e}"

    
    col_widths = [60, 15, 15, 15, 15]
    header_format = (
        f"{{:<{col_widths[0]}}}{{:>{col_widths[1]}}}{{:>{col_widths[2]}}}{{:>{col_widths[3]}}}{{:>{col_widths[4]}}}"
    )
    row_format = header_format

    
    print(header_format.format("Parameter", "Mean", "Std", "L2 norm", "RMS"))
    print("-" * (sum(col_widths) + 1))

    
    for hname, mean_val, std_val, l2_val, rms_val in stats:
        print(
            row_format.format(
                hname,
                fmt(mean_val),
                fmt(std_val),
                fmt(l2_val),
                fmt(rms_val),
            )
        )
