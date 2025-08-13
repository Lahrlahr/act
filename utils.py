import numpy as np
import torch
import os
import h5py
import json
import random
from PIL import Image
from pathlib import Path
import glob
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, num_queries):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.num_queries = num_queries
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        action_data = action_data[:self.num_queries]
        is_pad = is_pad[:self.num_queries]

        return image_data, qpos_data, action_data, is_pad


class EpisodicDataset1(torch.utils.data.Dataset):
    def __init__(self, episode_list, norm_stats, num_queries):
        self.episode_list = episode_list
        self.norm_stats = norm_stats
        self.num_queries = num_queries

    def __len__(self) -> int:
        return len(self.episode_list)

    def __getitem__(self, idx: int):
        episode_data = self.episode_list[idx]
        start_idx = random.randint(0, len(episode_data) - self.num_queries - 1)
        end_idx = start_idx + self.num_queries

        current_step = episode_data[f"step_{start_idx}"]

        front_image_path = current_step["observations_rgb_images_camera_front_image"]
        left_image_path = current_step["observations_rgb_images_camera_left_wrist_image"]
        right_image_path = current_step["observations_rgb_images_camera_right_wrist_image"]
        front_image = Image.open(front_image_path).convert("RGB")
        left_image = Image.open(left_image_path).convert("RGB")
        right_image = Image.open(right_image_path).convert("RGB")
        front_tensor = torch.tensor(np.array(front_image)).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        left_tensor = torch.tensor(np.array(left_image)).permute(2, 0, 1)
        right_tensor = torch.tensor(np.array(right_image)).permute(2, 0, 1)
        image_data = torch.stack([front_tensor, left_tensor, right_tensor], dim=0)  # (3, 3, H, W)

        qpos_data = torch.cat([
            (torch.tensor(current_step["puppet/joint_position_left"]) - self.norm_stats["joint_position_left"][
                'mean']) / \
            self.norm_stats["joint_position_left"]['std'],
            (torch.tensor(
                current_step["puppet/joint_position_right"]) - self.norm_stats["joint_position_right"]['mean']) / \
            self.norm_stats["joint_position_right"]['std']
        ], dim=0)

        left_list = []
        right_list = []
        for i in range(start_idx, end_idx):
            step = episode_data[f"step_{i + 1}"]
            left_list.append(step["puppet/joint_position_left"])
            right_list.append(step["puppet/joint_position_right"])
        action_left = (torch.tensor(left_list) - self.norm_stats["joint_position_left"]['mean']) / \
                      self.norm_stats["joint_position_left"]['std']
        action_right = (torch.tensor(right_list) - self.norm_stats["joint_position_right"]['mean']) / \
                       self.norm_stats["joint_position_right"]['std']
        action_data = torch.cat([action_left, action_right], dim=1)

        image_data = image_data / 255.0

        is_pad = np.zeros(self.num_queries)
        is_pad = torch.from_numpy(is_pad).bool()
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    # all_qpos_data = torch.stack(all_qpos_data)
    # all_action_data = torch.stack(all_action_data)
    # all_action_data = all_action_data
    #
    # # normalize action data
    # action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    # action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    #
    # # normalize qpos data
    # qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    # qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    # qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    cat_qpos = torch.cat(all_qpos_data, dim=0)  # shape: (sum(T_i), D)
    cat_action = torch.cat(all_action_data, dim=0)

    # 计算 mean 和 std
    qpos_mean = cat_qpos.mean(dim=0, keepdim=True)
    qpos_std = cat_qpos.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, float('inf'))

    action_mean = cat_action.mean(dim=0, keepdim=True)
    action_std = cat_action.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, float('inf'))

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def get_first_level_dirs(root_path):
    first_level_dirs = []
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            first_level_dirs.append(item_path)
    return sorted(first_level_dirs)


def preprocess_dataset(root_dir: str, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    all_episodes = []
    first_level_dirs = get_first_level_dirs(root_dir)
    for dir_path in first_level_dirs:
        # json_path = os.path.join(dir_path, "data.json")
        json_files = glob.glob(os.path.join(dir_path, "*.json"))
        if not json_files:
            continue
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        for episode_name, episode_data in data.items():
            for step_name, step_data in episode_data.items():
                for key, value in step_data.items():
                    if isinstance(value, str) and value.startswith('episode'):
                        full_path = os.path.join(dir_path, value)
                        step_data[key] = full_path
        all_episodes.extend(data.values())

    all_joint_positions_left = []
    all_joint_positions_right = []
    for episode in all_episodes:
        for step_data in episode.values():
            joint_position_left = step_data.get("puppet/joint_position_left")
            joint_position_right = step_data.get("puppet/joint_position_right")
            all_joint_positions_left.append(joint_position_left)
            all_joint_positions_right.append(joint_position_right)
    all_joint_positions_left = np.array(all_joint_positions_left).flatten()
    all_joint_positions_right = np.array(all_joint_positions_right).flatten()
    mean_left = np.mean(all_joint_positions_left)
    std_left = np.std(all_joint_positions_left)
    mean_right = np.mean(all_joint_positions_right)
    std_right = np.std(all_joint_positions_right)
    norm_stats = {
        "joint_position_left": {"mean": mean_left, "std": std_left},
        "joint_position_right": {"mean": mean_right, "std": std_right}
    }

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_episodes), generator=generator).tolist()

    split_idx = int(len(all_episodes) * train_ratio)
    train_episodes = [all_episodes[i] for i in indices[:split_idx]]
    val_episodes = [all_episodes[i] for i in indices[split_idx:]]

    return train_episodes, val_episodes, norm_stats


def get_norm_stats1(root_dir: str):
    all_joint_positions_left = []
    all_joint_positions_right = []
    json_files = [
        os.path.join(root_dir, f) for f in os.listdir(root_dir)
        if f.endswith('.json') and os.path.isfile(os.path.join(root_dir, f))
    ]

    for file_name in json_files:
        with open(file_name, 'r') as f:
            data = json.load(f)
        # 遍历每个 episode
        for episode_name, episode_data in data.items():
            # 遍历每个 step
            for step_name, step_data in episode_data.items():
                joint_position_left = step_data.get("puppet/joint_position_left")
                joint_position_right = step_data.get("puppet/joint_position_right")
                all_joint_positions_left.append(joint_position_left)
                all_joint_positions_right.append(joint_position_right)

    all_joint_positions_left = np.array(all_joint_positions_left).flatten()
    all_joint_positions_right = np.array(all_joint_positions_right).flatten()

    mean_left = np.mean(all_joint_positions_left)
    std_left = np.std(all_joint_positions_left)
    mean_right = np.mean(all_joint_positions_right)
    std_right = np.std(all_joint_positions_right)

    return {
        "joint_position_left": {"mean": mean_left, "std": std_left},
        "joint_position_right": {"mean": mean_right, "std": std_right}
    }


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, num_queries):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, num_queries)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, num_queries)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def load_data1(dataset_dir, camera_names, batch_size_train, batch_size_val, num_queries):
    train_ratio = 0.8
    train_episodes, val_episodes, norm_stats = preprocess_dataset(dataset_dir, train_ratio, 42)
    print(f"📁 {dataset_dir} | ✅ Total: {len(train_episodes) + len(val_episodes)} episodes "
          f"(Train: {len(train_episodes)}, Val: {len(val_episodes)})")

    train_dataset = EpisodicDataset1(train_episodes, norm_stats, num_queries)
    val_dataset = EpisodicDataset1(val_episodes, norm_stats, num_queries)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    return train_dataloader, val_dataloader, norm_stats


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
