import os
import json
import re
from collections import defaultdict
from tqdm import tqdm


def find_second_level_dirs(root_dir):
    second_level_dirs = []

    # 遍历一级子目录（depth=1）
    for dir1_name in os.listdir(root_dir):
        dir1_path = os.path.join(root_dir, dir1_name)
        if not os.path.isdir(dir1_path):
            continue  # 跳过文件

        # 遍历一级目录下的内容（即二级目录）
        for dir2_name in os.listdir(dir1_path):
            dir2_path = os.path.join(dir1_path, dir2_name)
            if os.path.isdir(dir2_path):
                second_level_dirs.append(dir2_path)

    return second_level_dirs

def find_first_level_dirs(root_dir):
    first_level_dirs = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isdir(item_path):
            first_level_dirs.append(item_path)
    return first_level_dirs

def extract_episode_number(path):
    # 从路径的文件夹名中提取数字
    basename = os.path.basename(path)
    match = re.search(r'(\d+)', basename)
    return int(match.group(1)) if match else float('inf')  # 没有数字的排最后


prompt_map = {
    'put_pen_left_arm_pure_white_bg': 'Use the left arm to put the pen into the pen holder.',
    'put_pen_left_arm_gray_white_bg': 'Use the left arm to put the pen into the pen holder.',
    'put_pen_left_arm_cyan_white_bg': 'Use the left arm to put the pen into the pen holder.',
    'put_pen_right_arm_pure_white_bg': 'Use the right arm to put the pen into the pen holder.',
    'put_pen_right_arm_gray_white_bg': 'Use the right arm to put the pen into the pen holder.',
    'put_pen_right_arm_cyan_white_bg': 'Use the right arm to put the pen into the pen holder.',

    'take_pen_left_arm_black_white_bg': 'Use the left arm to take the pen out of the pen holder.',
    'take_pen_left_arm_gray_white_bg': 'Use the left arm to take the pen out of the pen holder.',
    'take_pen_left_arm_green_white_bg': 'Use the left arm to take the pen out of the pen holder.',
    'take_pen_right_arm_black_white_bg': 'Use the right arm to take the pen out of the pen holder.',
    'take_pen_right_arm_gray_white_bg': 'Use the right arm to take the pen out of the pen holder.',
    'take_pen_right_arm_green_white_bg': 'Use the right arm to take the pen out of the pen holder.',
    
    'sort_blocks_green_bg': 'Sort the blocks by color.',
    'sort_blocks_gray_white_bg': 'Sort the blocks by color.',
    'sort_blocks_kd': 'Sort the blocks by color.',

    'set_the_cup_upright_black_white_bg': 'set the cup upright.',
    'set_the_cup_upright_gray_white_bg': 'set the cup upright.',
    'set_the_cup_upright_cyan_white_bg': 'set the cup upright.',
}
joint_left = "arm/jointState/puppetLeft/"
joint_right = "arm/jointState/puppetRight/"
camera_front_image = "camera/color/front/"
camera_left_image = "camera/color/left/"
camera_right_image = "camera/color/right/"
txt_file = "sync.txt"
for root_dir in find_first_level_dirs('/data/share_nips/robot/ario1/hold_the_cup'):
    all_episodes = [
        os.path.join(root_dir, f) for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and re.search(r'^episode\d+$', f)
    ]
    all_episodes = sorted(all_episodes, key=extract_episode_number)

    data = defaultdict(dict)
    for episode in tqdm(all_episodes, desc=f"Processing episodes in {root_dir}😁😁", total=len(all_episodes),
                        colour='green'):
        total_steps = 0
        paths = {
            'joint_left': os.path.join(episode, joint_left, txt_file),
            'joint_right': os.path.join(episode, joint_right, txt_file),
            'camera_front': os.path.join(episode, camera_front_image, txt_file),
            'camera_left': os.path.join(episode, camera_left_image, txt_file),
            'camera_right': os.path.join(episode, camera_right_image, txt_file),
        }
        with open(paths['joint_left'], 'r') as f:
            jl_lines = [line.strip() for line in f if line.strip()]
        with open(paths['joint_right'], 'r') as f:
            jr_lines = [line.strip() for line in f if line.strip()]
        with open(paths['camera_front'], 'r') as f:
            cf_lines = [line.strip() for line in f if line.strip()]
        with open(paths['camera_left'], 'r') as f:
            cl_lines = [line.strip() for line in f if line.strip()]
        with open(paths['camera_right'], 'r') as f:
            cr_lines = [line.strip() for line in f if line.strip()]
        lengths = [len(jl_lines), len(jr_lines), len(cf_lines), len(cl_lines), len(cr_lines)]
        if len(set(lengths)) != 1:
            raise ValueError(f"txt文件数量不一致: {lengths}")

        for jl_ts, jr_ts, cf_ts, cl_ts, cr_ts in zip(jl_lines, jr_lines, cf_lines, cl_lines, cr_lines):
            jl_path = os.path.join(episode, joint_left, jl_ts)
            jr_path = os.path.join(episode, joint_right, jr_ts)
            with open(jl_path, 'r') as f:
                jl_pos = json.load(f).get("position")
            with open(jr_path, 'r') as f:
                jr_pos = json.load(f).get("position")

            front_img = os.path.join(os.path.basename(episode), camera_front_image, cf_ts)
            left_wrist_img = os.path.join(os.path.basename(episode), camera_left_image, cl_ts)
            right_wrist_img = os.path.join(os.path.basename(episode), camera_right_image, cr_ts)

            prompt = prompt_map[os.path.basename(root_dir)]

            data[os.path.basename(episode).replace('episode', 'episode_')][f'step_{total_steps}'] = {
                'observations_rgb_images_camera_front_image': front_img,
                'observations_rgb_images_camera_left_wrist_image': left_wrist_img,
                'observations_rgb_images_camera_right_wrist_image': right_wrist_img,
                'puppet/joint_position_left': jl_pos,
                'puppet/joint_position_right': jr_pos,
                'prompt': prompt
            }
            total_steps += 1
    output_json = os.path.join(root_dir, "data.json")
    with open(output_json, 'w') as out_f:
        json.dump(data, out_f, indent=2)
    print(f"🤣🤣🤣 Saved {len(data)} combined episodes to {output_json}")
