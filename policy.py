import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython

e = IPython.embed

from einops import rearrange
import torch
import pickle
import os
import numpy as np
import json
from openpi_client import base_policy as _base_policy

BasePolicy = _base_policy.BasePolicy


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            # actions = actions[:, :self.model.num_queries]
            # is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class SEVERPolicy(BasePolicy):
    def __init__(
            self,
            policy_config,
            ckpt_dir,
            ckpt_name,
            temporal_agg
    ):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        ckpt_dir = '/data/huangguang/checkpoint/act/pick_pen/chunk30'
        ckpt_path = "/data/huangguang/checkpoint/act/pick_pen/chunk30/policy_epoch_350_seed_0.ckpt"
        # ckpt_path = '/data/huangguang/act/checkpoint4/pick_pen/policy_best.ckpt'
        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.left_mean = stats['joint_position_left']['mean']
            self.left_std = stats['joint_position_left']['std']
            self.right_mean = stats['joint_position_right']['mean']
            self.right_std = stats['joint_position_right']['std']

        self.query_frequency = policy_config['num_queries']
        self.temporal_agg = temporal_agg
        if temporal_agg:
            self.query_frequency = 1
            self.num_queries = policy_config['num_queries']
            self.all_time_actions = []

        self.all_actions = None
        self.camera_names = policy_config['camera_names']
        self._metadata = {}
        self.t = 0

    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        with torch.inference_mode():
            left_qpos = torch.from_numpy(obs['observation_joint_position_left']).float().cuda()
            right_qpos = torch.from_numpy(obs['observation_joint_position_right']).float().cuda()

            qpos = torch.cat([
                (left_qpos - self.left_mean) / self.left_std,
                (right_qpos - self.right_mean) / self.right_std
            ], dim=0).unsqueeze(0)

            curr_images = []
            for cam_name in ['observation_front_image', 'observation_left_wrist_image',
                             'observation_right_wrist_image']:
                curr_image = rearrange(obs[cam_name], 'h w c -> c h w')
                curr_images.append(curr_image)
            curr_image = np.stack(curr_images, axis=0)
            curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
            if self.temporal_agg:
                actions_for_curr_step = []
                self.all_time_actions.append(self.all_actions)
                for i in range(self.num_queries):
                    if len(self.all_time_actions) < i + 1:
                        break
                    action = self.all_time_actions[-1 - i][0, i].unsqueeze(0)  # (1, 14)
                    actions_for_curr_step.append(action)
                actions_for_curr_step.reverse()
                actions_for_curr_step = torch.cat(actions_for_curr_step, dim=0)

                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)  # 时序越靠前权重越大
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.all_actions[:, self.t % self.query_frequency]
            self.t += 1

            raw_action = self.policy(qpos, curr_image)

            ### post-process actions
            action = torch.cat([
                raw_action[..., :7] * self.left_std + self.left_mean,
                raw_action[..., 7:] * self.right_std + self.right_mean
            ], dim=2).squeeze(dim=0)

            # action = torch.cat([
            #     raw_action[:,  : 7] * self.left_std + self.left_mean,
            #     raw_action[:,  7:] * self.right_std + self.right_mean
            # ], dim=1)
            action = action.cpu().numpy()

            # === 记录当前动作 ===
            log_entry = {
                "timestep": self.t,
                "raw_action": raw_action.cpu().numpy()[0].tolist(),  # 归一化前输出
                "post_action": action.tolist(),  # 实际输出动作
                "left_qpos": left_qpos.cpu().numpy().tolist(),
                "right_qpos": right_qpos.cpu().numpy().tolist(),
            }

            # 写入文件（追加模式）
            log_file_path = "hglog.jsonl"
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            return {"actions": action}

    @property
    def metadata(self):
        return self._metadata


class SEVERPolicy1(BasePolicy):
    def __init__(
            self,
            policy_config,
            ckpt_dir,
            ckpt_name,
            temporal_agg
    ):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        ckpt_path = "/data/huangguang/act/checkpoint2/hold_cup/policy_last.ckpt"
        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        self.query_frequency = policy_config['num_queries']
        self.temporal_agg = temporal_agg
        if temporal_agg:
            self.query_frequency = 1
            self.num_queries = policy_config['num_queries']
            self.all_time_actions = []

        self.all_actions = None
        self.camera_names = policy_config['camera_names']
        self._metadata = {}
        self.t = 0

    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        with torch.inference_mode():
            qpos_numpy = np.concatenate(
                (obs['observation_joint_position_left'], obs['observation_joint_position_right']), axis=0)
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            curr_images = []
            for cam_name in ['observation_front_image', 'observation_left_wrist_image',
                             'observation_right_wrist_image']:
                curr_image = rearrange(obs[cam_name], 'h w c -> c h w')
                curr_images.append(curr_image)
            curr_image = np.stack(curr_images, axis=0)
            curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

            # qpos_numpy = np.array(obs['qpos'])
            # qpos = self.pre_process(qpos_numpy)
            # qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            #
            # curr_images = []
            # for cam_name in self.camera_names:
            #     curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
            #     curr_images.append(curr_image)
            # curr_image = np.stack(curr_images, axis=0)
            # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
            if self.temporal_agg:
                actions_for_curr_step = []
                self.all_time_actions.append(self.all_actions)
                for i in range(self.num_queries):
                    if len(self.all_time_actions) < i + 1:
                        break
                    action = self.all_time_actions[-1 - i][0, i].unsqueeze(0)  # (1, 14)
                    actions_for_curr_step.append(action)
                actions_for_curr_step.reverse()
                actions_for_curr_step = torch.cat(actions_for_curr_step, dim=0)

                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)  # 时序越靠前权重越大
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.all_actions[:, self.t % self.query_frequency]
            self.t += 1

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)

            action = action[None, :]
            return {"actions": action}

    @property
    def metadata(self):
        return self._metadata


class FixedPolicy(BasePolicy):
    def __init__(self, json_path: str):
        """
        固定策略，从json文件中加载动作。
        JSON 文件格式应为：
        {
            "actions": [[...], [...], ...]  # 每个动作是一个 list，组成一个动作序列
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.actions = np.array(data['actions'])  # shape: (T, action_dim)
        self._metadata = {}
        self.t = 0  # 当前时间步

    def infer(self, obs: dict) -> dict:  # obs 可以忽略
        if self.t < len(self.actions):
            action = self.actions[self.t]
        else:
            action = self.actions[-1]  # 超出范围时重复最后一个动作

        self.t += 1
        return {"actions": action[None, :]}  # 添加 batch 维度

    @property
    def metadata(self):
        return self._metadata
