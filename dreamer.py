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
        self.device        = torch.device(config.device)
        self.act_entropy   = float(config.act_entropy)
        self.kl_free       = float(config.kl_free)
        self.imag_horizon  = int(config.imag_horizon)
        self.horizon       = int(config.horizon)
        self.lamb          = float(config.lamb)
        self.return_ema    = networks.ReturnEMA(device=self.device)
        self.act_dim       = int(config.act_dim)
        self.rep_loss      = str(config.rep_loss)
        self.phase         = int(getattr(config, "phase", 1))
        self.use_depth     = bool(getattr(config, "use_depth", False))
        self.safety_threshold = float(getattr(config, "safety_threshold", 0.4))
        self.brake_vector  = torch.zeros(self.act_dim, device=self.device)

        image_shape = (int(config.img_height), int(config.img_width), 6)
        shapes = {
            "image": (int(config.img_height), int(config.img_width), 2 if self.use_depth else 6)
        }

        self.encoder    = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm       = rssm.RSSM(config.rssm, self.embed_size, self.act_dim)
        self.reward     = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont       = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape   = (self.act_dim,)
        config.actor.dist    = config.actor.dist.cont
        self.act_discrete    = False

        self.actor      = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value      = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update    = int(config.slow_target_update)
        self.slow_target_fraction  = float(config.slow_target_fraction)
        self._slow_value           = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads   = bool(config.log_grads)
        self.num_drones = int(config.get("num_drone_classes", 10))
        self.drone_embed = nn.Embedding(self.num_drones, 16)
        self.safety_net = networks.SafetyNet(
            in_channels=shapes["image"][-1],
            action_dim=self.act_dim,
            speed_dim=1,
        )
        modules = {
            "rssm":    self.rssm,
            "actor":   self.actor,
            "value":   self.value,
            "reward":  self.reward,
            "cont":    self.cont,
            "encoder": self.encoder,
            "drone_embed": self.drone_embed, 
            "safety_net": self.safety_net,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder, self.rssm._deter, self.rssm.flat_stoch, shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            self.prj          = Projector(self.rssm.feat_size, self.embed_size)
            self.barlow_lambd = float(config.r2dreamer.lambd)
            modules.update({"projector": self.prj})
        elif self.rep_loss == "dreamerpro":
            dpc                         = config.dreamer_pro
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

        self._modules = modules

        for key, module in self._modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")

        self._named_params = self._collect_named_params()
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc       = _agc
        self._base_lr   = float(config.lr)
        self._opt_betas = (float(config.beta1), float(config.beta2))
        self._opt_eps   = float(config.eps)
        self._optimizer = self._build_optimizer()
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._lr_lambda = lr_lambda
        self._scheduler = self.build_scheduler(self._optimizer)

        self.train()
        self._configure_trainable_modules()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling loss computation with torch.compile...")
            self.compute_losses = torch.compile(self.compute_losses, mode="reduce-overhead")

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
        for module_name, module in self._modules.items():
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

        stem_params = [
            param for name, param in encoder_named_params
            if ".backbone.0.0." in name
        ]
        other_encoder_params = [
            param for name, param in encoder_named_params
            if ".backbone.0.0." not in name
        ]

        param_groups = []
        if stem_params:
            param_groups.append({"params": stem_params, "lr": self._base_lr * 10})
        if other_encoder_params:
            param_groups.append({"params": other_encoder_params, "lr": self._base_lr})
        return param_groups

    def _build_optimizer_param_groups(self):
        param_groups = self._get_encoder_param_groups()
        grouped_param_ids = {
            id(param)
            for group in param_groups
            for param in group["params"]
        }

        for module_name, module in self._modules.items():
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

    def get_optimizer(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return LaProp(
            [{"params": params, "lr": self._base_lr}],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _update_slow_target(self):
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)
        
        self._frozen_drone_embed = copy.deepcopy(self.drone_embed)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.drone_embed.named_parameters(), self._frozen_drone_embed.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clone_and_freeze()
        return self

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
        torch.compiler.cudagraph_mark_step_begin()
        p_obs                          = self.preprocess(obs)
        embed                          = self._frozen_encoder(p_obs)
        d_emb                          = self._get_drone_embedding(
            obs.get("drone_id"), batch_shape=embed.shape[:-1], frozen=True, device=embed.device
        )
        prev_stoch, prev_deter, prev_action = (
            state["stoch"], state["deter"], state["prev_action"],
        )
        stoch, deter, _ = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_action, embed, obs["is_first"], d_emb
        )
        feat         = self._frozen_rssm.get_feat(stoch, deter)
        action_dist  = self._frozen_actor(feat)
        action       = action_dist.mode if eval else action_dist.rsample()
        speed = obs.get("speed", torch.zeros(*action.shape[:-1], 1, device=action.device))
        safety_score = self.safety_net(p_obs["image"], speed, action)
        brake = self.brake_vector.view(*([1] * (action.ndim - 1)), -1).expand_as(action)
        action = torch.where(safety_score < self.safety_threshold, brake, action)
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action       = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,)
        )

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")
        B     = min(data["action"].shape[0], 6)
        embed = self.encoder(data)
        d_emb = self._get_drone_embedding(data.get("drone_id"), batch_shape=data["action"].shape[:2], device=embed.device)
        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5], data["action"][:B, :5],
            tuple(val[:B] for val in initial), data["is_first"][:B, :5], d_emb[:B, :5],
        )
        recon               = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch, init_deter, data["action"][:B, 5:], d_emb[:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
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

        self.zero_grad(**zero_grad_kwargs)

        if self._log_grads and old_params is not None:
            updates = [new.detach() - old for new, old in zip(grad_params, old_params)]
            mets["opt/param_rms"] = tools.compute_rms(grad_params)
            mets["opt/update_rms"] = tools.compute_rms(updates)

        metrics.update(mets)
        return (stoch, deter), metrics

    def update(self, replay_buffer):
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
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
        losses["dyn"]     = torch.mean(dyn_loss)
        losses["rep"]     = torch.mean(rep_loss)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key]))
                for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            x1      = self.prj(feat[:, :].reshape(B * T, -1))
            x2      = embed.reshape(B * T, -1).detach()
            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c       = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss  = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask    = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss  = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
            metrics["barlow/invariance"] = invariance_loss.item()
            metrics["barlow/redundancy"] = redundancy_loss.item()
            metrics["barlow/std_mean"]   = x1_norm.std(0).mean().item()
        elif self.rep_loss == "infonce":
            x1          = self.prj(feat[:, :].reshape(B * T, -1))
            x2          = embed.reshape(B * T, -1).detach()
            logits      = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels      = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
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
            actor_dist = self.actor(feat.detach())
            actor_mode = actor_dist.mode()
            losses["bc"] = torch.mean((actor_mode - data["action"]) ** 2)
            speed = data.get("speed", torch.zeros(B, T, 1, device=feat.device))
            crash_target = data.get("crash", to_f32(data["is_terminal"]).unsqueeze(-1))
            safety_pred = self.safety_net(data["image"], speed, data["action"])
            losses["safety"] = F.binary_cross_entropy(safety_pred, crash_target)
            metrics["safety/score_mean"] = safety_pred.mean()
            metrics["bc/mse"] = losses["bc"].detach()
        elif self.phase >= 3:
            start = (
                post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
                post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
            )
            d_emb_imag = d_emb[:, -1].repeat_interleave(T, dim=0)
            imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1, d_emb_imag)
            imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
            imag_reward = self._frozen_reward(imag_feat).mode()
            imag_cont = self._frozen_cont(imag_feat).mean
            imag_value = self._frozen_value(imag_feat).mode()
            imag_slow_value = self._frozen_slow_value(imag_feat).mode()
            disc = 1 - 1 / self.horizon
            weight = torch.cumprod(imag_cont * disc, dim=1)
            last = torch.zeros_like(imag_cont)
            term = 1 - imag_cont
            ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
            ret_offset, ret_scale = self.return_ema(ret)
            adv = (ret - imag_value[:, :-1]) / ret_scale
            policy = self.actor(imag_feat)
            logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
            entropy = policy.entropy()[:, :-1].unsqueeze(-1)
            losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))
            imag_value_dist = self.value(imag_feat)
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
            boot = ret[:, 0].reshape(B, T, 1)
            value = self._frozen_value(feat).mode()
            slow_value = self._frozen_slow_value(feat).mode()
            disc = 1 - 1 / self.horizon
            weight = 1.0 - last
            ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
            ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
            value_dist = self.value(feat)
            losses["repval"] = torch.mean(
                weight[:, :-1]
                * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(-1)
            )
            metrics.update(tools.tensorstats(ret, "ret_replay"))
            metrics.update(tools.tensorstats(value, "value_replay"))
            metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales.get(k, 1.0) for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), total_loss, metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, d_emb):
        feats   = []
        actions = []
        stoch, deter = start
        for _ in range(imag_horizon):
            feat   = self._frozen_rssm.get_feat(stoch, deter)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action, d_emb)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

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
        if "image" in data:
            image = to_f32(data["image"])
            if torch.max(image) > 1.5:
                image = image / 255.0
            data["image"] = image
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

    @torch.no_grad()
    def augment_data(self, data):
        data_aug        = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
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
