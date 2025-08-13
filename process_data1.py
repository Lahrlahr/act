import os
import cv2

# 参数配置
image_dirs = [
    "/data/huangguang/datas/episode1/camera/color/front/",
    "/data/huangguang/datas/episode1/camera/color/left/",
    "/data/huangguang/datas/episode1/camera/color/right/",
    # 可以继续添加更多目录
]

txt_file = "sync.txt"
output_prefix = "puppet_video_"
FPS = 10

for image_dir in image_dirs:
    dir_name = os.path.basename(os.path.normpath(image_dir))
    output_video = f"{output_prefix}{dir_name}.mp4"

    txt_path = os.path.join(image_dir, txt_file)
    with open(txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    video_out = None
    for img_name in image_names:
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
        print(f"❌ [{dir_name}] 没有成功读取任何图片，视频未生成")