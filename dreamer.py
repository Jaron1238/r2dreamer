import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32


_DEFAULT = object()


class Dreamer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.model if hasattr(config, "model") else config
        self.device        = torch.device(getattr(config, "device", getattr(cfg, "device", "cpu")))
        self.act_entropy   = float(cfg.act_entropy)
        self.kl_free       = float(cfg.kl_free)
        self.imag_horizon  = int(cfg.imag_horizon)
        self.horizon       = int(cfg.horizon)
        self.lamb          = float(cfg.lamb)
        self.return_ema    = networks.ReturnEMA(device=self.device)
        self.act_dim       = int(cfg.act_dim)
        self.rep_loss      = str(cfg.rep_loss)
        self.phase         = int(getattr(config, "phase", getattr(cfg, "phase", 1)))
        self.use_depth     = bool(getattr(config, "use_depth", getattr(cfg, "use_depth", False)))
        self.safety_threshold = float(getattr(cfg, "safety_threshold", 0.4))
        self.safety_input_key = str(getattr(cfg, "safety_input_key", "image"))
        hover_throttle = float(getattr(cfg, "hover_throttle", 0.5))
        throttle_index = int(getattr(cfg, "throttle_index", 0))
        self.brake_vector  = torch.zeros(self.act_dim, device=self.device)
        if 0 <= throttle_index < self.act_dim:
            self.brake_vector[throttle_index] = hover_throttle
        self.drone_embed_dim = int(getattr(cfg, "drone_embed_dim", 16))
        self.infonce_temperature = float(getattr(cfg, "infonce_temperature", 0.1))
        self.video_pred_batch = int(getattr(cfg, "video_pred_batch", 6))
        self.motor_inertia_range = tuple(getattr(cfg, "motor_inertia_range", (0.1, 0.8)))
        self.use_latent_goals = bool(getattr(cfg, "use_latent_goals", True))
        self.latent_goal_noise_scale = float(getattr(cfg, "latent_goal_noise_scale", 0.2))

        shapes = {
            "image": (int(cfg.img_height), int(cfg.img_width), 6)
        }
        if bool(getattr(cfg, "use_cam_overlay", False)):
            shapes["cam_overlay"] = (int(cfg.img_height), int(cfg.img_width), 1)
        if self.safety_input_key == "raw_image":
            shapes["raw_image"] = (
                int(cfg.img_height),
                int(cfg.img_width),
                int(getattr(cfg, "safety_in_channels", 1)),
            )

        self.encoder    = networks.MultiEncoder(cfg.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        cfg.rssm.d_emb_dim = self.drone_embed_dim
        self.rssm       = rssm.RSSM(cfg.rssm, self.embed_size, self.act_dim)
        self.reward     = networks.MLPHead(cfg.reward, self.rssm.feat_size)
        self.cont       = networks.MLPHead(cfg.cont, self.rssm.feat_size)

        cfg.actor.shape   = (self.act_dim,)
        cfg.actor.dist    = cfg.actor.dist.cont
        self.act_discrete    = False

        self._loss_scales = dict(cfg.loss_scales)
        self._log_grads   = bool(cfg.log_grads)
        self.num_drones = int(cfg.get("num_drone_classes", 10))
        self.drone_embed = nn.Embedding(self.num_drones, self.drone_embed_dim)
        self.ctx_len = int(getattr(cfg, "ctx_len", 16))
        self.ctx_consistency_weight = float(getattr(cfg, "ctx_consistency_weight", 0.1))
        self.ctx_warmup_steps = int(getattr(cfg, "ctx_warmup_steps", 1000))
        self._ctx_updates = 0
        self.context_encoder = networks.ContextEncoder(
            self.rssm.flat_stoch, self.rssm._deter, self.act_dim, ctx_len=self.ctx_len, out_dim=self.drone_embed_dim
        )
        self.safety_net = networks.SafetyNet(
            in_channels=shapes[self.safety_input_key][-1],
            action_dim=self.act_dim,
            speed_dim=1,
        )
        self.use_depth_aux = bool(getattr(cfg, "use_depth_aux", False))
        if self.phase >= 3 and self.use_depth_aux:
            self.depth_aux_head = networks.DepthAuxHead(
                in_dim=self.rssm.flat_stoch + self.rssm._deter,
                out_h=int(cfg.img_height),
                out_w=int(cfg.img_width),
            )
        actor_input_dim = self.rssm.feat_size + self.drone_embed_dim
        self.actor = networks.MLPHead(cfg.actor, actor_input_dim)
        self.value = networks.MLPHead(cfg.critic, actor_input_dim)
        self.latent_goal_buffer_size = int(getattr(cfg, "latent_goal_buffer_size", 65536))
        self.register_buffer(
            "_goal_feat_buffer",
            torch.zeros(self.latent_goal_buffer_size, self.rssm.feat_size, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_goal_feat_count",
            torch.zeros(1, dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.slow_target_update    = int(cfg.slow_target_update)
        self.slow_target_fraction  = float(cfg.slow_target_fraction)
        self._slow_value           = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0
        modules = {
            "rssm":    self.rssm,
            "actor":   self.actor,
            "value":   self.value,
            "reward":  self.reward,
            "cont":    self.cont,
            "encoder": self.encoder,
            "drone_embed": self.drone_embed, 
            "context_encoder": self.context_encoder,
            "safety_net": self.safety_net,
        }
        if self.phase >= 3 and self.use_depth_aux:
            modules.update({"depth_aux_head": self.depth_aux_head})

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                cfg.decoder, self.rssm._deter, self.rssm.flat_stoch, shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            self.prj          = Projector(self.rssm.feat_size, self.embed_size)
            self.prj_target   = Projector(self.embed_size, self.embed_size)
            self.barlow_lambd = float(cfg.r2dreamer.lambd)
            modules.update({"projector": self.prj, "projector_target": self.prj_target})
        elif self.rep_loss == "nedreamer":
            ne_cfg = cfg.nedreamer
            self.barlow_lambd = float(ne_cfg.lambd)
            self.use_ema_target = bool(ne_cfg.use_ema_target)
            self.nedreamer_transformer = networks.CausalTemporalTransformer(
                state_dim=self.rssm.feat_size,
                action_dim=self.act_dim,
                model_dim=int(getattr(ne_cfg, "hidden_dim", 256)),
                num_layers=int(ne_cfg.transformer_layers),
                num_heads=int(ne_cfg.transformer_heads),
                dropout=float(ne_cfg.transformer_dropout),
                max_len=int(getattr(config, "batch_length", 64)),
                action_proj_dim=32,
            )
            self.nedreamer_predictor = networks.NEPredictorHead(
                int(getattr(ne_cfg, "hidden_dim", 256)),
                self.embed_size,
            )
            modules.update(
                {
                    "nedreamer_transformer": self.nedreamer_transformer,
                    "nedreamer_predictor": self.nedreamer_predictor,
                }
            )
            self.target_encoder = None
            if self.use_ema_target:
                self.target_encoder = copy.deepcopy(self.encoder)
                for param in self.target_encoder.parameters():
                    param.requires_grad = False
                modules.update({"target_encoder": self.target_encoder})
        elif self.rep_loss == "dreamerpro":
            dpc                         = cfg.dreamer_pro
            self.warm_up                = int(dpc.warm_up)
            self.num_prototypes         = int(dpc.num_prototypes)
            self.proto_dim              = int(dpc.proto_dim)
            self.temperature            = float(dpc.temperature)
            self.sinkhorn_eps           = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters         = int(dpc.sinkhorn_iters)
            self.ema_update_every       = int(dpc.ema_update_every)
            self.ema_update_fraction    = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters= int(dpc.freeze_prototypes_iters)
            self.aug_max_delta          = float(dpc.aug.max_delta)
            self.aug_same_across_time   = bool(dpc.aug.same_across_time)
            self.aug_bilinear           = bool(dpc.aug.bilinear)
            self._prototypes            = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj               = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj              = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder           = copy.deepcopy(self.encoder)
            self._ema_obs_proj          = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes":   self._prototypes,
                "obs_proj":     self.obs_proj,
                "feat_proj":    self.feat_proj,
            })

        self._model_modules = modules

        for key, module in self._model_modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")

        self._named_params = self._collect_named_params()
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(cfg.agc), float(cfg.pmin), foreach=True)

        self._agc       = _agc
        self._base_lr   = float(cfg.lr)
        self._opt_betas = (float(cfg.beta1), float(cfg.beta2))
        self._opt_eps   = float(cfg.eps)
        self._optimizer = self._build_optimizer()
        self._scaler = GradScaler()

        def lr_lambda(step):
            if cfg.warmup:
                return min(1.0, (step + 1) / cfg.warmup)
            return 1.0

        self._lr_lambda = lr_lambda
        self._scheduler = self.build_scheduler(self._optimizer)

        self.train()
        self._configure_trainable_modules()
        self.clone_and_freeze()
        if cfg.compile:
            print("Compiling loss computation with torch.compile...")
            self.compute_losses = torch.compile(self.compute_losses, mode="reduce-overhead")

    def _maybe_mark_cudagraph_step(self):
        if self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()

    def encode(self, images):
        obs = {"image": images}
        return self.encoder(obs)

    def barlow_loss(self, z_a, z_b):
        N, D     = z_a.shape
        z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
        z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)
        c        = torch.mm(z_a_norm.T, z_b_norm) / N

        invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag_mask   = ~torch.eye(D, dtype=torch.bool, device=z_a.device)
        redundancy_loss = c[off_diag_mask].pow(2).sum()
        loss            = invariance_loss + self.barlow_lambd * redundancy_loss

        metrics = {
            "barlow/invariance": invariance_loss.item(),
            "barlow/redundancy": redundancy_loss.item(),
            "barlow/std_mean":   z_a_norm.std(0).mean().item(),
        }
        return loss, metrics

    def _iter_trainable_named_parameters(self, module_name, module):
        if isinstance(module, nn.Parameter):
            if module.requires_grad:
                yield module_name, module
            return
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                yield f"{module_name}.{param_name}", param

    def _collect_named_params(self):
        named_params = OrderedDict()
        for module_name, module in self._model_modules.items():
            for param_name, param in self._iter_trainable_named_parameters(module_name, module):
                named_params[param_name] = param
        return named_params

    def _get_encoder_param_groups(self):
        encoder_named_params = [
            (name, param) for name, param in self._named_params.items()
            if name.startswith("encoder.")
        ]
        if not encoder_named_params:
            return []

        backbone_params = [
            param for name, param in encoder_named_params
            if ".backbone." in name
        ]
        adapter_params = [
            param for name, param in encoder_named_params
            if ".backbone." not in name
        ]

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self._base_lr * 0.3})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": self._base_lr})
        return param_groups

    def _build_optimizer_param_groups(self):
        param_groups = self._get_encoder_param_groups()
        grouped_param_ids = {
            id(param)
            for group in param_groups
            for param in group["params"]
        }

        for module_name, module in self._model_modules.items():
            if module_name == "encoder":
                continue

            module_params = [
                param
                for _, param in self._iter_trainable_named_parameters(module_name, module)
                if id(param) not in grouped_param_ids
            ]
            if module_params:
                param_groups.append({"params": module_params, "lr": self._base_lr})

        return param_groups

    def _build_optimizer(self):
        return LaProp(
            self._build_optimizer_param_groups(),
            betas=self._opt_betas,
            eps=self._opt_eps,
        )

    def _update_slow_target(self):
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def _freeze_copy(self, module):
        frozen = copy.deepcopy(module)
        for param in frozen.parameters():
            param.requires_grad_(False)
        return frozen

    def train(self, mode=True):
        super().train(mode)
        self._slow_value.train(False)
        if mode and self.phase == 2:
            self.encoder.eval()
            self.rssm.eval()
        return self

    def clone_and_freeze(self):
        self._frozen_encoder = self._freeze_copy(self.encoder)
        self._frozen_rssm = self._freeze_copy(self.rssm)
        self._frozen_reward = self._freeze_copy(self.reward)
        self._frozen_cont = self._freeze_copy(self.cont)
        self._frozen_actor = self._freeze_copy(self.actor)
        self._frozen_value = self._freeze_copy(self.value)
        self._frozen_slow_value = self._freeze_copy(self._slow_value)
        self._frozen_drone_embed = self._freeze_copy(self.drone_embed)

    def _get_drone_embedding(self, drone_id=None, *, batch_shape=None, frozen=False, device=None):
        module = self._frozen_drone_embed if frozen else self.drone_embed
        if drone_id is None:
            if batch_shape is None:
                raise ValueError("batch_shape is required when drone_id is not provided.")
            drone_id = torch.zeros(*batch_shape, dtype=torch.long, device=device or self.device)
        else:
            drone_id = drone_id.to(device=device or self.device, dtype=torch.long)
        return module(drone_id)

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        self._maybe_mark_cudagraph_step()
        p_obs                          = self.preprocess(obs)
        embed                          = self._frozen_encoder(p_obs)
        teacher_d_emb                  = self._get_drone_embedding(
            obs.get("drone_id"), batch_shape=embed.shape[:-1], frozen=True, device=embed.device
        )
        if self.phase >= 3:
            ctx_ready = state["ctx_valid_steps"] >= self.ctx_len
            ctx_d_emb = self.context_encoder(state["ctx_flat_stoch"], state["ctx_deter"], state["ctx_action"])
            d_emb = torch.where(ctx_ready.unsqueeze(-1), ctx_d_emb, teacher_d_emb)
        else:
            d_emb = teacher_d_emb
        prev_stoch, prev_deter, prev_action, prev_filtered_action = (
            state["stoch"], state["deter"], state["prev_action"], state["prev_filtered_action"],
        )
        stoch, deter, _, filtered_action = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_action, embed, obs["is_first"], d_emb, prev_filtered_action
        )
        feat         = self._frozen_rssm.get_feat(stoch, deter)
        policy_input = torch.cat([feat, d_emb], dim=-1)
        action_dist  = self._frozen_actor(policy_input)
        action       = action_dist.mode if eval else action_dist.rsample()
        speed = obs.get("speed", torch.zeros(*action.shape[:-1], 1, device=action.device))
        safety_image = p_obs.get(self.safety_input_key, p_obs["image"])
        img_t = safety_image.unsqueeze(1)
        speed_t = speed.unsqueeze(1)
        action_t = action.unsqueeze(1)
        safety_score = self.safety_net(img_t, speed_t, action_t).squeeze(1)
        brake = self.brake_vector.view(*([1] * (action.ndim - 1)), -1).expand_as(action)
        action = torch.where(safety_score > self.safety_threshold, brake, action)
        flat_stoch = stoch.reshape(stoch.shape[0], -1)
        ctx_flat_stoch = torch.cat([state["ctx_flat_stoch"][:, 1:], flat_stoch.unsqueeze(1)], dim=1)
        ctx_deter = torch.cat([state["ctx_deter"][:, 1:], deter.unsqueeze(1)], dim=1)
        ctx_action = torch.cat([state["ctx_action"][:, 1:], action.unsqueeze(1)], dim=1)
        ctx_valid_steps = torch.clamp(state["ctx_valid_steps"] + 1, max=self.ctx_len)
        return action, TensorDict(
            {
                "stoch": stoch,
                "deter": deter,
                "prev_action": action,
                "prev_filtered_action": filtered_action,
                "ctx_flat_stoch": ctx_flat_stoch,
                "ctx_deter": ctx_deter,
                "ctx_action": ctx_action,
                "ctx_valid_steps": ctx_valid_steps,
            },
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter, prev_filtered_action = self.rssm.initial(B)
        action       = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        ctx_flat_stoch = torch.zeros(B, self.ctx_len, self.rssm.flat_stoch, dtype=torch.float32, device=self.device)
        ctx_deter = torch.zeros(B, self.ctx_len, self.rssm._deter, dtype=torch.float32, device=self.device)
        ctx_action = torch.zeros(B, self.ctx_len, self.act_dim, dtype=torch.float32, device=self.device)
        ctx_valid_steps = torch.zeros(B, dtype=torch.long, device=self.device)
        return TensorDict(
            {
                "stoch": stoch,
                "deter": deter,
                "prev_action": action,
                "prev_filtered_action": prev_filtered_action,
                "ctx_flat_stoch": ctx_flat_stoch,
                "ctx_deter": ctx_deter,
                "ctx_action": ctx_action,
                "ctx_valid_steps": ctx_valid_steps,
            }, batch_size=(B,)
        )

    @torch.no_grad()
    def video_pred(self, data, initial):
        self._maybe_mark_cudagraph_step()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")
        B     = min(data["action"].shape[0], self.video_pred_batch)
        embed = self.encoder(data)
        d_emb = self._get_drone_embedding(data.get("drone_id"), batch_shape=data["action"].shape[:2], device=embed.device)
        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5], data["action"][:B, :5],
            tuple(val[:B] for val in initial), data["is_first"][:B, :5], d_emb[:B, :5],
        )
        recon               = self.decoder(post_stoch, post_deter)["image"].mode[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch, init_deter, data["action"][:B, 5:], d_emb[:B, 5:]
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def build_scheduler(self, optimizer):
        return LambdaLR(optimizer, lr_lambda=self._lr_lambda)

    def _optimizer_params(self, optimizer):
        return [param for group in optimizer.param_groups for param in group["params"] if param is not None]

    def train_step(
        self,
        data,
        initial=None,
        *,
        optimizer=_DEFAULT,
        scheduler=_DEFAULT,
        scaler=_DEFAULT,
        backward_fn=None,
        clip_grad_fn=None,
        grad_params=None,
        zero_grad_kwargs=None,
        autocast_enabled=True,
        post_backward_hook=None,
    ):
        if optimizer is _DEFAULT:
            optimizer = self._optimizer
        if scheduler is _DEFAULT:
            scheduler = self._scheduler
        if scaler is _DEFAULT:
            scaler = self._scaler
        zero_grad_kwargs = zero_grad_kwargs or {"set_to_none": True}
        grad_params = list(grad_params) if grad_params is not None else self._optimizer_params(optimizer)

        metrics = {}
        old_params = None
        if self._log_grads:
            old_params = [param.data.clone().detach() for param in grad_params]

        autocast_context = autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        )
        with autocast_context:
            (stoch, deter), loss, mets = self.compute_losses(data, initial)

        if backward_fn is not None:
            backward_fn(loss)
        elif scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)

        if post_backward_hook is not None:
            post_backward_hook()

        if self._log_grads:
            grads = [param.grad for param in grad_params if param.grad is not None]
            if grads:
                mets["opt/grad_norm"] = tools.compute_global_norm(grads)
                mets["opt/grad_rms"] = tools.compute_rms(grads)

        self._agc(grad_params)
        if clip_grad_fn is not None:
            clip_grad_fn(grad_params)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            mets["opt/grad_scale"] = scaler.get_scale()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
            mets["opt/lr"] = scheduler.get_last_lr()[0]

        optimizer.zero_grad(**zero_grad_kwargs)

        self._update_slow_target()

        if self._log_grads and old_params is not None:
            updates = [new.detach() - old for new, old in zip(grad_params, old_params)]
            mets["opt/param_rms"] = tools.compute_rms(grad_params)
            mets["opt/update_rms"] = tools.compute_rms(updates)

        metrics.update(mets)
        return (stoch, deter), metrics

    def update(self, replay_buffer):
        data, index, initial = replay_buffer.sample()
        self._maybe_mark_cudagraph_step()
        p_data = self.preprocess(data)
        if self.rep_loss == "dreamerpro":
            self.ema_update()

        def post_backward_hook():
            if (
                self.rep_loss == "dreamerpro"
                and self._ema_updates < self.freeze_prototypes_iters
                and self._prototypes.grad is not None
            ):
                self._prototypes.grad.zero_()

        (stoch, deter), metrics = self.train_step(
            p_data,
            initial,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            scaler=self._scaler,
            zero_grad_kwargs={"set_to_none": True},
            post_backward_hook=post_backward_hook,
        )
        if self.rep_loss == "nedreamer" and getattr(self, "use_ema_target", False):
            self._update_nedreamer_ema_target()
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def compute_losses(self, data, initial):
        losses  = {}
        metrics = {}
        B, T    = data.shape
        if initial is None:
            initial = self.rssm.initial(B)
        if self.phase == 2:
            with torch.no_grad():
                embed = self.encoder(data)
                d_emb = self.drone_embed(data["drone_id"])
                post_stoch, post_deter, post_logit = self.rssm.observe(
                    embed, data["action"], initial, data["is_first"], d_emb
                )
        else:
            embed = self.encoder(data)
            d_emb = self.drone_embed(data["drone_id"])
            post_stoch, post_deter, post_logit = self.rssm.observe(
                embed, data["action"], initial, data["is_first"], d_emb
            )
        _, prior_logit    = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        burn_in_mask = data.get("burn_in_mask", None)
        losses["dyn"] = networks.masked_mean(dyn_loss, burn_in_mask)
        losses["rep"] = networks.masked_mean(rep_loss, burn_in_mask)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        self._update_goal_feat_buffer(feat.detach().reshape(-1, self.rssm.feat_size))
        flat_post_stoch = post_stoch.reshape(B, T, -1)
        ctx_student = self.context_encoder(flat_post_stoch, post_deter, data["action"])
        ctx_student_time = ctx_student.unsqueeze(1).expand(-1, T, -1)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key]))
                for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            x1      = self.prj(feat[:, :].reshape(B * T, -1))
            x2      = self.prj_target(embed.reshape(B * T, -1).detach())
            losses["barlow"], barlow_metrics = self.barlow_loss(x1, x2)
            metrics.update(barlow_metrics)
        elif self.rep_loss == "infonce":
            x1          = self.prj(feat[:, :].reshape(B * T, -1))
            x2          = self.prj_target(embed.reshape(B * T, -1).detach())
            x1          = F.normalize(x1, dim=-1)
            x2          = F.normalize(x2, dim=-1)
            logits      = torch.matmul(x1, x2.T) / self.infonce_temperature
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels      = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "nedreamer":
            state_seq = self.rssm.get_feat(post_stoch, post_deter)
            prev_actions = torch.cat(
                [torch.zeros(B, 1, self.act_dim, device=data["action"].device, dtype=data["action"].dtype), data["action"][:, :-1]],
                dim=1,
            )
            tr_out = self.nedreamer_transformer(state_seq, prev_actions)
            pred = self.nedreamer_predictor(tr_out[:, :-1])
            with torch.no_grad():
                if getattr(self, "use_ema_target", False):
                    target_embed = self.target_encoder(data)
                else:
                    target_embed = self.encoder(data)
            target = target_embed[:, 1:]
            valid = (~data["is_terminal"][:, :-1].bool()).reshape(-1)
            if burn_in_mask is not None:
                valid = valid & burn_in_mask[:, :-1].reshape(-1).bool()
            flat_pred = pred.reshape(B * (T - 1), -1)
            flat_target = target.reshape(B * (T - 1), -1)
            if valid.any():
                flat_pred = flat_pred[valid]
                flat_target = flat_target[valid]
                losses["nedreamer"], barlow_metrics = self.barlow_loss(flat_pred, flat_target)
            else:
                losses["nedreamer"] = torch.zeros((), device=pred.device, dtype=pred.dtype)
                barlow_metrics = {"barlow/invariance": 0.0, "barlow/redundancy": 0.0, "barlow/std_mean": 0.0}
            metrics.update({f"nedreamer/{k.split('/')[-1]}": v for k, v in barlow_metrics.items()})
        elif self.rep_loss == "dreamerpro":
            with torch.no_grad():
                data_aug    = self.augment_data(data)
                initial_aug = (
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)
            embed_aug               = self.encoder(data_aug)
            d_emb_aug               = self._get_drone_embedding(
                data_aug.get("drone_id"), batch_shape=data_aug["action"].shape[:2], device=embed_aug.device
            )
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"], d_emb_aug
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont          = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))

        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        if self.phase == 2:
            policy_input = torch.cat([feat.detach(), d_emb.detach()], dim=-1)
            actor_dist = self.actor(policy_input)
            bc_loss = -actor_dist.log_prob(data["action"])
            losses["bc"] = networks.masked_mean(bc_loss, burn_in_mask)
            losses["ctx_align"] = torch.mean((ctx_student_time - d_emb.detach()) ** 2)

            speed = data.get("speed", torch.zeros(B, T, 1, device=feat.device))
            crash_target = data.get("crash", to_f32(data["is_terminal"]).unsqueeze(-1))
            safety_image = data.get(self.safety_input_key, data["image"])
            safety_pred = self.safety_net(safety_image, speed, data["action"])
            focal_alpha = 0.8
            focal_gamma = 2.0
            eps = 1e-4
            safety_prob = torch.clamp(safety_pred, min=eps, max=1.0 - eps)
            safety_logits = torch.logit(safety_prob)
            bce = F.binary_cross_entropy_with_logits(safety_logits, crash_target, reduction="none")
            p_t = torch.exp(-bce)
            alpha_t = crash_target * focal_alpha + (1.0 - crash_target) * (1.0 - focal_alpha)
            focal_loss = alpha_t * ((1.0 - p_t) ** focal_gamma) * bce
            losses["safety"] = torch.mean(focal_loss)

            if "inj_raw_image" in data:
                inj_speed = torch.zeros_like(speed)
                inj_action = torch.zeros_like(data["action"])
                inj_pred = self.safety_net(data["inj_raw_image"], inj_speed, inj_action)
                inj_prob = torch.clamp(inj_pred, min=eps, max=1.0 - eps)
                inj_logits = torch.logit(inj_prob)
                inj_target = data["inj_crash"]
                inj_bce = F.binary_cross_entropy_with_logits(inj_logits, inj_target, reduction="none")
                inj_pt = torch.exp(-inj_bce)
                inj_alpha = inj_target * focal_alpha + (1.0 - inj_target) * (1.0 - focal_alpha)
                inj_loss = torch.mean(inj_alpha * ((1.0 - inj_pt) ** focal_gamma) * inj_bce)
                losses["safety"] = losses["safety"] + inj_loss

            metrics["safety/score_mean"] = safety_prob.mean()
            metrics["bc/mse"] = losses["bc"].detach()
        elif self.phase >= 3:
            self._ctx_updates += 1
            d_emb = ctx_student_time
            valid_mask = burn_in_mask if burn_in_mask is not None else torch.ones_like(data["is_first"], dtype=torch.bool)
            flat_mask = valid_mask.reshape(-1)
            flat_post_stoch = post_stoch.reshape(-1, *post_stoch.shape[2:]).detach()
            flat_post_deter = post_deter.reshape(-1, *post_deter.shape[2:]).detach()
            if flat_mask.any():
                flat_post_stoch = flat_post_stoch[flat_mask]
                flat_post_deter = flat_post_deter[flat_mask]
            start = (flat_post_stoch, flat_post_deter)
            flat_d_emb = d_emb.reshape(-1, d_emb.shape[-1])
            if flat_mask.any():
                flat_d_emb = flat_d_emb[flat_mask]
            d_emb_imag = flat_d_emb
            goal_feat = self._sample_goal_feat(post_stoch.detach(), post_deter.detach(), batch_size=d_emb_imag.shape[0])
            with torch.no_grad():
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    imag_feat, imag_action, imag_goal_reward = self._imagine(
                        start, self.imag_horizon + 1, d_emb_imag, goal_feat=goal_feat
                    )
            imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
            if imag_goal_reward is not None:
                imag_reward = imag_goal_reward.detach()
            else:
                imag_reward = self._frozen_reward(imag_feat).mode
            imag_cont = self._frozen_cont(imag_feat).mean
            imag_d_emb = d_emb_imag.unsqueeze(1).expand(-1, imag_feat.shape[1], -1)
            imag_policy_input = torch.cat([imag_feat, imag_d_emb], dim=-1)
            imag_value = self._frozen_value(imag_policy_input).mode
            imag_slow_value = self._frozen_slow_value(imag_policy_input).mode
            disc = 1 - 1 / self.horizon
            weight = torch.cumprod(imag_cont * disc, dim=1)
            last = torch.zeros_like(imag_cont)
            term = 1 - imag_cont
            ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
            ret_offset, ret_scale = self.return_ema(ret)
            adv = (ret - imag_value[:, :-1]) / ret_scale
            policy = self.actor(imag_policy_input)
            logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
            entropy = policy.entropy()[:, :-1].unsqueeze(-1)
            losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))
            imag_value_dist = self.value(imag_policy_input)
            tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
            losses["value"] = torch.mean(
                weight[:, :-1].detach()
                * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[:, :-1].unsqueeze(-1)
            )
            ret_normed = (ret - ret_offset) / ret_scale
            metrics["ret"] = torch.mean(ret_normed)
            metrics["ret_005"] = self.return_ema.ema_vals[0]
            metrics["ret_095"] = self.return_ema.ema_vals[1]
            metrics["adv"] = torch.mean(adv)
            metrics["adv_std"] = torch.std(adv)
            metrics["con"] = torch.mean(imag_cont)
            metrics["rew"] = torch.mean(imag_reward)
            metrics["val"] = torch.mean(imag_value)
            metrics["tar"] = torch.mean(ret)
            metrics["slowval"] = torch.mean(imag_slow_value)
            metrics["weight"] = torch.mean(weight)
            metrics["action_entropy"] = torch.mean(entropy)
            metrics.update(tools.tensorstats(imag_action, "action"))
            last, term, reward = (to_f32(data["is_last"]), to_f32(data["is_terminal"]), to_f32(data["reward"]))
            feat = self.rssm.get_feat(post_stoch, post_deter)
            boot = ret[:, 0].reshape(B, -1, 1)[:, :T]
            replay_policy_input = torch.cat([feat, d_emb], dim=-1)
            value = self._frozen_value(replay_policy_input).mode
            slow_value = self._frozen_slow_value(replay_policy_input).mode
            disc = 1 - 1 / self.horizon
            weight = 1.0 - last
            ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
            ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
            value_dist = self.value(replay_policy_input)
            losses["repval"] = torch.mean(
                weight[:, :-1]
                * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(-1)
            )
            metrics.update(tools.tensorstats(ret, "ret_replay"))
            metrics.update(tools.tensorstats(value, "value_replay"))
            metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))
            half = max(1, T // 2)
            z_a = self.context_encoder(flat_post_stoch[:, :half], post_deter[:, :half], data["action"][:, :half])
            z_b = self.context_encoder(flat_post_stoch[:, half:], post_deter[:, half:], data["action"][:, half:])
            warm = min(1.0, self._ctx_updates / max(1, self.ctx_warmup_steps))
            losses["ctx_consistency"] = self.ctx_consistency_weight * warm * torch.mean((z_a - z_b.detach()) ** 2)
            if bool(getattr(self.config.model, "use_depth_aux", False)) and "depth_target" in data:
                post_feat = torch.cat([post_stoch.reshape(B, T, -1), post_deter], dim=-1)
                depth_pred = self.depth_aux_head(post_feat)
                depth_target = data["depth_target"]
                losses["depth_aux"] = self._scale_invariant_depth_loss(depth_pred, depth_target)

        total_loss = sum([v * self._loss_scales.get(k, 1.0) for k, v in losses.items()])
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), total_loss, metrics

    def _scale_invariant_depth_loss(self, pred, target):
        pred_log = torch.log(pred + 1e-6)
        target_log = torch.log(target + 1e-6)
        diff = pred_log - target_log
        mse = torch.mean(diff**2)
        mean_term = torch.mean(diff) ** 2
        return mse - 0.5 * mean_term

    @torch.no_grad()
    def _update_nedreamer_ema_target(self):
        ema_rate = float(self.config.model.nedreamer.ema_rate)
        for ema_param, online_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            ema_param.data.lerp_(online_param.data, 1.0 - ema_rate)

    @torch.no_grad()
    def _update_goal_feat_buffer(self, feat):
        if feat.numel() == 0:
            return
        n = feat.shape[0]
        count = int(self._goal_feat_count.item())
        if n >= self.latent_goal_buffer_size:
            self._goal_feat_buffer.copy_(feat[-self.latent_goal_buffer_size :])
            self._goal_feat_count[0] = count + n
            return
        write_pos = count % self.latent_goal_buffer_size
        first = min(self.latent_goal_buffer_size - write_pos, n)
        self._goal_feat_buffer[write_pos : write_pos + first] = feat[:first]
        if first < n:
            self._goal_feat_buffer[: n - first] = feat[first:]
        self._goal_feat_count[0] = count + n

    @torch.no_grad()
    def _sample_goal_feat(self, post_stoch, post_deter, batch_size):
        if not self.use_latent_goals or batch_size == 0:
            return None
        count = min(int(self._goal_feat_count.item()), self.latent_goal_buffer_size)
        if count > 0:
            feat_pool = self._goal_feat_buffer[:count]
        else:
            feat_pool = self.rssm.get_feat(post_stoch, post_deter).reshape(-1, self.rssm.feat_size)

        # Distance-Weighting: prefer goals in the reachable middle range
        current_feat = feat_pool.mean(dim=0, keepdim=True)  # representative current state
        dists = 1.0 - F.cosine_similarity(
            current_feat.expand(feat_pool.shape[0], -1), feat_pool, dim=-1
        )  # (N,) — 0=identical, 1=opposite
        weights = torch.zeros_like(dists)
        mask = (dists > 0.2) & (dists < 0.8)  # Goldilocks zone: not trivial, not impossible
        weights[mask] = 1.0
        if weights.sum() == 0:
            weights = torch.ones_like(dists)  # fallback: uniform sampling

        idx = torch.multinomial(weights, batch_size, replacement=True)
        goals = feat_pool[idx]
        feat_var = torch.var(feat_pool, dim=0, unbiased=False)
        noise = torch.randn_like(goals) * torch.sqrt(feat_var + 1e-6) * self.latent_goal_noise_scale
        return goals + noise

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, d_emb, goal_feat=None):
        feats   = []
        actions = []
        stoch, deter = start
        alpha_min, alpha_max = self.motor_inertia_range
        alpha = torch.rand(stoch.shape[0], 1, device=stoch.device) * (alpha_max - alpha_min) + alpha_min
        prev_filtered_action = torch.zeros(stoch.shape[0], self.act_dim, dtype=torch.float32, device=stoch.device)
        for _ in range(imag_horizon):
            feat   = self._frozen_rssm.get_feat(stoch, deter)
            policy_input = torch.cat([feat, d_emb], dim=-1)
            action = self._frozen_actor(policy_input).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter, prev_filtered_action = self._frozen_rssm.img_step(
                stoch, deter, action, d_emb, prev_filtered_action=prev_filtered_action, alpha=alpha
            )
        feats = torch.stack(feats, dim=1)
        actions = torch.stack(actions, dim=1)
        goal_reward = None
        if goal_feat is not None:
            goal_feat = goal_feat.unsqueeze(1).expand(-1, feats.shape[1], -1)
            sim = F.cosine_similarity(feats, goal_feat, dim=-1)  # (B, T)

            # Progress bonus: reward getting closer to goal each step
            progress = sim[:, 1:] - sim[:, :-1]                 # (B, T-1)
            progress = F.pad(progress, (1, 0), value=0.0)        # (B, T) — first step = 0

            goal_reward = (sim * 0.7 + progress * 0.3).unsqueeze(-1)  # (B, T, 1)
        return feats, actions, goal_reward

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live   = (1 - to_f32(term))[:, 1:] * disc
        cont   = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out    = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        data = data.copy()
        if "image" in data:
            image_in = data["image"]
            image = to_f32(image_in)
            if not torch.is_floating_point(image_in):
                image = image / 255.0
            data["image"] = image
        if "raw_image" in data:
            raw_image_in = data["raw_image"]
            raw_image = to_f32(raw_image_in)
            if not torch.is_floating_point(raw_image_in):
                raw_image = raw_image / 255.0
            data["raw_image"] = raw_image
        if "cam_overlay" in data:
            cam_overlay_in = data["cam_overlay"]
            cam_overlay = to_f32(cam_overlay_in)
            if not torch.is_floating_point(cam_overlay_in):
                cam_overlay = cam_overlay / 255.0
            data["cam_overlay"] = cam_overlay
        if "depth_target" in data:
            depth_target_in = data["depth_target"]
            depth_target = to_f32(depth_target_in)
            if not torch.is_floating_point(depth_target_in):
                depth_target = depth_target / 255.0
            data["depth_target"] = depth_target
        return data

    def _configure_trainable_modules(self):
        # Phase 1: world model only; no policy/value updates.
        if self.phase == 1:
            for p in self.actor.parameters():
                p.requires_grad = False
            for p in self.value.parameters():
                p.requires_grad = False
        # Phase 2: freeze encoder/rssm, train actor via BC and safety net.
        elif self.phase == 2:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.rssm.parameters():
                p.requires_grad = False
            for p in self.reward.parameters():
                p.requires_grad = False
            for p in self.cont.parameters():
                p.requires_grad = False
        elif self.phase >= 3:
            for p in self.drone_embed.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = data.apply(lambda value: torch.cat([value, value], dim=0))
        image           = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image, self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj  = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        shape   = scores.shape
        K       = shape[0]
        scores  = scores.reshape(-1)
        log_Q   = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q   = log_Q.reshape(K, -1)
        N       = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q        = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q        = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q     = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        obs_proj   = self.obs_proj(embed)
        obs_norm   = torch.norm(obs_proj, dim=-1)
        obs_proj   = F.normalize(obs_proj, p=2, dim=-1)
        B, T       = obs_proj.shape[:2]
        obs_proj   = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up:]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)
        ema_proj   = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up:]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)
        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)
        feat        = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj   = self.feat_proj(feat)
        feat_norm   = torch.norm(feat_proj, dim=-1)
        feat_proj   = F.normalize(feat_proj, p=2, dim=-1)
        feat_proj   = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up:]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)
        swav_loss   = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) \
                    - 0.5 * torch.mean(torch.sum(ema_targets_1 * obs_logits_2, dim=0))
        temp_loss   = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss   = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))
        return {"swav": swav_loss, "temp": temp_loss, "norm": norm_loss}

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W  = x.shape
        x_flat         = x.reshape(B * T, C, H, W)
        pad            = int(max_delta)
        x_padded       = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad
        eps_h   = 1.0 / h_padded
        eps_w   = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift        = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)
        grid         = base_grid + shift
        mode         = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)
        return x_translated.reshape(B, T, C, H, W)
