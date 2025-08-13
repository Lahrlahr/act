import os
import cv2


def extract_step_and_view(filename):
    # 假设文件名为 "step_XXX_left.jpg" 或类似格式
    parts = filename.split('_')
    step = int(parts[1])  # 提取步骤编号
    view = parts[2].split('.')[0]  # 提取视角（left/right/front）
    return step, view

output_prefix = "puppet_video_"
FPS = 30
image_dir = os.listdir('/data/share_nips/robot/aidlux/pick_pen')
image_dir = '/data/share_nips/robot/aidlux/pick_pen/episode_50'
views = {'left': [], 'right': [], 'front': []}

for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):  # 确保只处理 .jpg 文件
        continue
    step, view = extract_step_and_view(filename)
    views[view].append((step, filename))

for view in views:
    views[view].sort(key=lambda x: x[0])
    output_video = os.path.join(image_dir, f"{output_prefix}{view}.mp4")
    video_out = None
    for _ , img_name in views[view]:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if video_out is None:
            height, width = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 格式
            video_out = cv2.VideoWriter(output_video, fourcc, FPS, (width, height))
        video_out.write(img)

    if video_out is not None:
        video_out.release()
        print(f"✅😂 视频已保存: {output_video}")
    else:
        print(f"❌ [{image_dir}] 没有成功读取任何图片，视频未生成")