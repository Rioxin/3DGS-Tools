import os
import shutil
import subprocess
import cv2
import glob
import argparse
from tqdm import tqdm

def prepare_output_dir(output_dir):
    # 如果输出目录已经存在
    if os.path.exists(output_dir):
        # 询问是否要删除原有目录
        user_input = input(f"The output directory '{output_dir}' already exists. Do you want to delete it and create a new one? (yes/no): ")

        # 如果yes，则删除原有目录并创建新目录
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Deleted existing directory '{output_dir}' and created a new one.")
        # 如果no，则退出程序
        elif user_input.lower() == "no":
            print("Exiting program.")
            exit()
        # 如果输入的既不是yes也不是no，则提示无效输入并退出程序
        else:
            print("Invalid input. Please enter 'yes' or 'no'. Exiting program.")
            exit()
    # 如果输出目录不存在，则直接创建一个新目录
    else:
        os.makedirs(output_dir)
        print(f"Created new directory '{output_dir}'.")

        
def convert_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '22',
        '-c:a', 'copy',
        output_path
    ]
    
    # 使用 subprocess 运行 ffmpeg 命令
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 检查是否成功
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode('utf-8')}")
    else:
        print(f"Video converted successfully: {output_path}")
        os.remove(input_path)

def frames2vid(images, save_path):
    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, 10, (WIDTH, HEIGHT))

    with tqdm(total=len(images), desc="Writing frames to video") as pbar:
        for image in images:
            video.write(image)
            pbar.update(1)

    video.release()
    output_path = os.path.splitext(save_path)[0] + "_h264.mp4"
    convert_video(save_path, output_path)
    return

def get_image_files(directory):
    image_files = sorted(glob.glob(os.path.join(directory, '*.png')))
    with tqdm(total=len(image_files), desc="Collecting image files") as pbar:
        for _ in image_files:
            pbar.update(1)
    return image_files

def main():
    parser = argparse.ArgumentParser(description="Convert images to videos")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for videos")
    args = parser.parse_args()

    image_files = get_image_files(args.directory)
    # 检测output是否已存在
    prepare_output_dir(args.output)
    # 读取图片文件
    images = [cv2.imread(file) for file in image_files]

    # 将图片转换成视频并保存
    save_paths = [os.path.join(args.output, f"video{i}.mp4") for i in range(len(images))]
    for images, save_path in zip([images[0::3], images[1::3], images[2::3]], save_paths):
        frames2vid(images, save_path)

if __name__ == "__main__":
    main()
