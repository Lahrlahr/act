import numpy as np
import torch
import os
import h5py

import os
import re

folder = '/data/huangguang/data'
files = os.listdir(folder)

# 筛选并提取原文件名中的数字
episode_files = [f for f in files if re.match(r'episode_\d+\.hdf5$', f)]
episode_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 按数字排序

# 临时重命名避免冲突
for i, old_name in enumerate(episode_files):
    old_path = os.path.join(folder, old_name)
    temp_path = os.path.join(folder, f'temp_{i}.hdf5')
    os.rename(old_path, temp_path)

# 再次重命名为 episode_数字.hdf5（不补0）
for i, temp_name in enumerate(sorted(os.listdir(folder))):
    if not temp_name.startswith('temp_'):
        continue
    temp_path = os.path.join(folder, temp_name)
    final_path = os.path.join(folder, f'episode_{i}.hdf5')
    print(f'Renaming {temp_name} -> episode_{i}.hdf5')
    os.rename(temp_path, final_path)

dataset_path = "/data/huangguang/data/episode_27.hdf5"

with h5py.File(dataset_path, 'r') as root:
    is_sim = root.attrs['sim']
    original_action_shape = root['/action'].shape
    episode_len = original_action_shape[0]
    # if sample_full_episode:
    #     start_ts = 0
    # else:
    #     start_ts = np.random.choice(episode_len)
    start_ts = np.random.choice(episode_len)
    # get observation at start_ts only
    qpos = root['/observations/qpos'][start_ts]
    qvel = root['/observations/qvel'][start_ts]
    image_dict = dict()
    for cam_name in ['top', 'left', 'right']:
        image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
    # get all actions after and including start_ts
    if is_sim:
        action = root['/action'][start_ts:]
        action_len = episode_len - start_ts
    else:
        action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

    padded_action = np.zeros(original_action_shape, dtype=np.float32)
    padded_action[:action_len] = action
    is_pad = np.zeros(episode_len)
    is_pad[action_len:] = 1

    # new axis for different cameras
    all_cam_images = []
    for cam_name in ['top', 'left', 'right']:
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
    # action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
    # qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]