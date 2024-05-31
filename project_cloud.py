import argparse
import io
import json
import os
import random
import shutil
import subprocess

import cv2
import nori2
import numpy as np
import open3d as o3d
import refile
from numpy.lib import recfunctions as rfn
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# 全局变量，用于存储 map1 和 map2
MAP_CACHE = {}
POINTS = {}  # 定义一个空字典来存储 POINTS
TRANSFORM = {}  # 定义一个空字典来存储 TRANSFORM


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def filter_and_sort_keys(data, keyword, min_suffix, max_suffix):
    # 确保"front"在前，"back"在后
    def custom_sort(key):
        if "front" in key:
            return (0, key)
        elif "back" in key:
            return (2, key)
        else:
            return (1, key)

    # 从字典中匹配相应的键并过滤后缀
    filtered_keys = [key for key in data.keys() if keyword in key and min_suffix <= int(key.split('_')[-1]) <= max_suffix]

    # 对键进行排序
    sorted_keys = sorted(filtered_keys, key=custom_sort)

    return sorted_keys


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


def load_json(path):
    with refile.smart_open(path, "r") as f:
        return json.load(f)


def load_img_from_nid(nid):
    # img 是一个三维的 NumPy 数组，其形状是 (height, width, 3)
    nori_fetcher = nori2.Fetcher()
    img_bin = np.frombuffer(nori_fetcher.get(nid), dtype=np.uint8)
    img = cv2.imdecode(img_bin, 1)
    return img


def get_transform_matrix(sensor_info, transform_key):
    trans = sensor_info[transform_key]["transform"]["translation"]
    quan = sensor_info[transform_key]["transform"]["rotation"]
    trans = np.array([trans["x"], trans["y"], trans["z"]], dtype=np.float32)
    quan = np.array([quan["x"], quan["y"], quan["z"],
                    quan["w"]], dtype=np.float32)
    rot_mat = R.from_quat(quan).as_matrix()
    res = np.eye(4, dtype=np.float32)
    res[:3, :3] = rot_mat
    res[:3, 3] = trans
    return res


def init_calibration_info(calibrated_data, camera_names):
    print("camera_names : ",camera_names)
    calibration_info = dict()
    calibration_info["ins2ego"] = get_transform_matrix(
        calibrated_data["gnss"], "gnss_ego")

    with tqdm(total=len(camera_names), desc="Extracting Calibration Info") as pbar:
        for cam in camera_names:
            cam_rfu2ego = get_transform_matrix(
                calibrated_data[cam], "extrinsic")
            calibration_info[cam] = cam_rfu2ego
            pbar.update(1)  
    calibration_info["fuser_lidar_rfu2ego"] = get_transform_matrix(
        calibrated_data["lidar_ego"], "extrinsic")
    return calibration_info


def get_camera2rfu_transform(sensor_info):
    trans = sensor_info["transform"]["translation"]
    quan = sensor_info["transform"]["rotation"]
    trans = np.array([trans["x"], trans["y"], trans["z"]], dtype=np.float32)
    quan = np.array([quan["x"], quan["y"], quan["z"],
                    quan["w"]], dtype=np.float32)
    rot_mat = R.from_quat(quan).as_matrix()
    res = np.eye(4, dtype=np.float32)
    res[:3, :3] = rot_mat
    res[:3, 3] = trans
    return res


def get_gnss_to_world(frame_info):
    # 之前private data初期ins 数据有误，因此存的是odom 数据，之后 ins 数据正确后，改存ins 数据，更多信息见：
    # https://wiki.megvii-inc.com/pages/viewpage.action?pageId=312323474
    if "odom_data" in frame_info:
        odom_data = frame_info["odom_data"]
        trans = odom_data["pose"]["pose"]["position"]  # dict
        quan = odom_data["pose"]["pose"]["orientation"]  # dict
    elif "ins_data" in frame_info:
        ins_data = frame_info["ins_data"]
        trans = ins_data["localization"]["position"]  # dict
        quan = ins_data["localization"]["orientation"]  # dict
    elif "timestamp" in frame_info:
        trans = frame_info["position"]  # dict
        quan = frame_info["orientation"]  # dict
    else:
        raise ValueError(f"{frame_info['frame_id']} lack odom/ins data")
    trans = np.array([trans["x"], trans["y"], trans["z"]], dtype=np.float32)
    quan = np.array([quan["x"], quan["y"], quan["z"],
                    quan["w"]], dtype=np.float32)
    rot_mat = R.from_quat(quan).as_matrix()
    res = np.eye(4, dtype=np.float32)
    res[:3, :3] = rot_mat
    res[:3, 3] = trans
    return res


# 提取相机内参
def extract_camera_calibration(data):
    camera_calibrations = {}

    for sensor_name, sensor_data in data["calibrated_sensors"].items():
        if "sensor_type" in sensor_data and sensor_data["sensor_type"] == "camera":
            calibration_info = {}
            calibration_info["distortion_model"] = sensor_data["intrinsic"]["distortion_model"]
            calibration_info["K"] = sensor_data["intrinsic"]["K"]
            calibration_info["D"] = sensor_data["intrinsic"]["D"]
            calibration_info["resolution"] = sensor_data["intrinsic"]["resolution"]
            camera_calibrations[sensor_name] = calibration_info

    return camera_calibrations


def warp(wm_img_nid, intrinsic, camera_id, images_size, R=None):
    global MAP_CACHE

    # 加载图像
    ori_img = load_img_from_nid(wm_img_nid)
    dim = intrinsic["resolution"]
    K, D = np.array(intrinsic["K"]), np.array(intrinsic["D"])
    
    new_dim = images_size  # 新的图像尺寸
    cx, cy = new_dim[0] / 2, new_dim[1] / 2  # 新的相机中心坐标
    fx, fy = K[0, 0], K[1, 1]  # 原始相机焦距
    new_fx, new_fy = fx * new_dim[0] / dim[0], fy * new_dim[1] / dim[1]  # 新的焦距
    new_cx, new_cy = cx, cy  # 新的相机中心

    new_K = np.array([[new_fx, 0, new_cx],
                      [0, new_fy, new_cy],
                      [0, 0, 1]])
    if R is None:
        R = np.eye(3)

    # 判断是否已经缓存过该相机参数的 map
    cache_key = tuple(K.flatten()) + tuple(D.flatten()) + \
        tuple(new_K.flatten()) + tuple(dim)
    if cache_key in MAP_CACHE:
        map1, map2 = MAP_CACHE[cache_key]
    else:
        # 重新计算 map 并缓存
        if intrinsic["distortion_model"] == "fisheye":
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K,
                D,
                R,
                new_K,
                new_dim,
                cv2.CV_16SC2,
            )
        else:
            map1, map2 = cv2.initUndistortRectifyMap(
                K,
                D,
                R,
                new_K,
                new_dim,
                cv2.CV_16SC2,
            )
        MAP_CACHE[cache_key] = (map1, map2)

    # 对图像进行变换
    img = cv2.remap(ori_img, map1, map2, cv2.INTER_LINEAR)
    if camera_id == "cam_front_120":
        cropped_img = img[:images_size[1]*3//4, :]
        img = cropped_img
    return img, new_K


def _get_point_cloud_plus(data, Transform, frame_idx, plus):
    nori_fetcher = nori2.Fetcher()
    if plus:
        d = nori_fetcher.get(data["frames"][frame_idx]
                             ["sensor_data"]["fix_lidar"]["cam_left_200"])
        d = np.frombuffer(d)
        pc_data = np.reshape(d, (-1, 8))
    else:
        nori_id = data["frames"][frame_idx]["sensor_data"]["fuser_lidar"]["nori_id"]
        d = nori_fetcher.get(nori_id)
        pc_data_raw = np.load(io.BytesIO(d)).copy()
        pc_data = rfn.structured_to_unstructured(pc_data_raw, dtype=np.float64)
    # 应用旋转和平移变换
    pc_data_transformed = np.matmul(
        pc_data[:, :3], Transform[:3, :3].T) + Transform[:3, 3]

    return pc_data_transformed


def get_camera_data(data, calibration_data, frame_idx, camera_id, images_size):

    # 获取nori_id
    nori_id = data["frames"][frame_idx]["sensor_data"][camera_id]["nori_id"]

    # 使用warp函数获取原始图像
    ori_img, K = warp(nori_id, calibration_data[camera_id], camera_id, images_size, None)

    # 获取相机信息
    camera_info_test = data["calibrated_sensors"][camera_id]

    return ori_img, camera_info_test, K


def downsample_points(points, voxel_size):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    downsampled_points = np.asarray(downsampled_pcd.points)

    return downsampled_points


def random_downsample_points(points, factor):

    sample_size = int(len(points) / factor)
    sample_size = max(1, sample_size)  # Ensure at least one point is selected

    sampled_indices = random.sample(range(len(points)), sample_size)
    downsampled_points = points[sampled_indices]

    return downsampled_points


def get_points(data, calibration_info, key_gnss2world, start_frame, end_frame, plus, downsample_ratio, random_flag):
    global POINTS
    global TRANSFORM
    points = None

    progress_bar = tqdm(range(start_frame, end_frame), desc="Concatenating point clouds")
    for frame_idx in progress_bar:
        # 获取当前帧的点云数据和变换矩阵
        sweep_gnss2world = get_gnss_to_world(data["frames"][frame_idx])
        Transform_curr = get_sweeprfu2keyrfu(calibration_info, key_gnss2world, sweep_gnss2world)
        TRANSFORM[frame_idx] = Transform_curr

        # 对当前帧的点云数据进行变换
        points_curr = _get_point_cloud_plus(data, Transform_curr, frame_idx, plus)
        POINTS[frame_idx] = points_curr

        if points is None:
            points = points_curr
        else:
            # 拼接当前帧的点云数据到累积的点云数据中
            points = np.concatenate((points, points_curr))
    if random_flag:
        point_downsample = random_downsample_points(points, downsample_ratio)
    else:
        point_downsample = downsample_points(points, downsample_ratio)
    return point_downsample


def get_points_except_currs(start_frame, end_frame, downsample_ratio, random_flag):
    global POINTS
    points_except_currs = {}

    # 外层循环遍历每一帧
    progress_bar = tqdm(range(start_frame, end_frame), desc="Processing cloud")
    for frame_idx in progress_bar:
        # 初始化缓存数组
        points_except = None

        # 内层循环拼接除了当前帧之外的所有点云数据
        for i in range(start_frame, end_frame):
            if i != frame_idx:
                points_curr = POINTS[i]
                if points_except is None:
                    points_except = points_curr
                else:
                    points_except = np.concatenate((points_except, points_curr))        
        if random_flag:
            point_downsample = random_downsample_points(points_except, downsample_ratio)
        else:
            point_downsample = downsample_points(points_except, downsample_ratio)
        # 将所有除了当前帧之外的点云数据拼接成一个数组，并添加到列表中
        points_except_currs[frame_idx] = point_downsample

    return points_except_currs


def project_uv_on_image(uv_points, image, point_size=3):
    # 将 NumPy 数组转换为 PIL 图像对象
    pil_image = Image.fromarray(np.uint8(image))

    # 创建绘图对象
    draw = ImageDraw.Draw(pil_image)

    # 将 UV 坐标点绘制在图像上
    for uv_point in uv_points:
        x, y, z = uv_point
        if z >= 0:
            # 计算圆点的位置和大小
            x1, y1 = x - point_size, y - point_size
            x2, y2 = x + point_size, y + point_size

            # 绘制一个圆点
            draw.ellipse([x1, y1, x2, y2], fill=(0, 0, 255))

    return pil_image


def project_lidar_to_depth_img(points, img_shape, cam_info, Transform_, image, K):
    # remove nan points
    points = points[:, :3]
    points = points[~np.any(np.isnan(points), axis=1)]

    extrinsic = get_camera2rfu_transform(cam_info["extrinsic"])
    T = extrinsic @ np.linalg.inv(Transform_)

    cur_camera_points = np.c_[points, np.ones(points.shape[0])]  # [N, 4]
    cur_camera_points = np.matmul(T, cur_camera_points.T).T[:, :3]  # [N, 3]

    # remove neg depth points
    cur_camera_points = cur_camera_points[cur_camera_points[:, 2] > 0, :]
    uv = np.matmul(K, cur_camera_points.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    # remove out of image size points
    mask = (uv[:, 0] < img_shape[0]) & (uv[:, 0] >= 0) & (
        uv[:, 1] >= 0) & (uv[:, 1] < img_shape[1])
    uv = uv[mask, :]
    point_size = 1  # 点的大小

    depth = cur_camera_points[mask, -1]
    depth_points = np.c_[uv, depth]
    return project_uv_on_image(depth_points, image, point_size)


def process_frame(Transform_curr, camera_info_test, images_size, frame_id, img, K, points=None):
    global POINTS

    # 如果未提供点云数据，则获取并变换点云数据
    if points is None:
        points_curr = POINTS[frame_id]
    else:
        points_curr = points
    # 将点云投影到深度图像上
    project_img = project_lidar_to_depth_img(points_curr, images_size, camera_info_test, Transform_curr, img, K)

    return project_img


def create_combined_video(args, data, calibration_data, points, points_except_currs):
    global TRANSFORM
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = args.output + "/" + args.video_path
    images_size = (args.width, args.height)
    combined_video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (args.width * 2, args.height * 2))

    # 创建四个小视频写入对象
    video_writers = []
    video_paths = [
    os.path.splitext(video_path)[0] + f"_{suffix}.mp4"
    for suffix in ["ori", "projectCur", "projectAll", "projectAllbutCur"]]
    for path in video_paths:
        video_writers.append(cv2.VideoWriter(path, fourcc, args.fps, (args.width, args.height)))

    # 设置进度条
    progress_bar = tqdm(range(args.start_frame, args.end_frame, 1), desc="Processing frames")
    for frame_idx in progress_bar:
        # 获取当前帧的点云数据和变换矩阵
        Transform_curr = TRANSFORM[frame_idx]
        # 获取原图和参数
        ori_img, camera_info_test, K = get_camera_data(data, calibration_data, frame_idx, args.camera_id, images_size)
        # 创建一个空白的图像
        blank_image = np.zeros((args.width, args.height, 3), dtype=np.uint8)

        # 将 ori_img 粘贴到空白图像的左上角
        blank_image[:ori_img.shape[0], :ori_img.shape[1]] = ori_img
        ori_img = blank_image
        image_ori = np.array(ori_img)
        
        # 获取投影图像
        project_img_1 = process_frame(Transform_curr, camera_info_test,
                                      images_size, frame_idx, ori_img, K,
                                      None)
        project_img_1 = np.array(project_img_1)

        project_img_2 = process_frame(Transform_curr, camera_info_test,
                                      images_size, frame_idx, ori_img, K,
                                      points)
        project_img_2 = np.array(project_img_2)

        project_img_3 = process_frame(Transform_curr, camera_info_test,
                                      images_size, frame_idx, ori_img, K,
                                      points_except_currs[frame_idx])
        project_img_3 = np.array(project_img_3)

        # 添加注释
        cv2.putText(image_ori, "Original Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(project_img_1, "Current Frame Point Cloud Projection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(project_img_2, "All Frames Point Cloud Projection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(project_img_3, "All But Current Frame Point Cloud Projection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 将四张图片拼接成2x2
        top_row = np.hstack((image_ori, project_img_1))
        bottom_row = np.hstack((project_img_2, project_img_3))
        combined_image = np.vstack((top_row, bottom_row))

        # 逐帧将图像写入视频
        combined_video_writer.write(combined_image)

        # 将四张图像分别写入四个小视频
        for i, img in enumerate([image_ori, project_img_1, project_img_2, project_img_3]):
            video_writers[i].write(img)

    # 释放视频写入对象
    combined_video_writer.release()
    for writer in video_writers:
        writer.release()

    # 转换视频格式
    convert_video(video_path, os.path.splitext(video_path)[0] + "_h264.mp4")
    for path in video_paths:
        convert_video(path, os.path.splitext(path)[0] + "_h264.mp4")


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
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    # 检查命令是否成功执行
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode('utf-8')}")
    else:
        print(f"Video converted successfully: {output_path}")
        # 删除原始视频文件
        os.remove(input_path)


def get_sweeprfu2keyrfu(calibration_info, key_gnss2world, sweep_gnss2world):
    lidar_rfu2ego = calibration_info["fuser_lidar_rfu2ego"]
    ins2ego = calibration_info["ins2ego"]
    sweep_rfu2keyrfu = (
        np.linalg.inv(lidar_rfu2ego)
        @ ins2ego
        @ np.linalg.inv(key_gnss2world)
        @ sweep_gnss2world
        @ np.linalg.inv(ins2ego)
        @ lidar_rfu2ego
    )
    return sweep_rfu2keyrfu


def check_cloud_status(args):
    if args.plus is not True:
        print("Cloud is old !")
    else:
        print("Cloud is new !")

    if args.random_downsample is not True:
        print("Using voxel downsampling.")
    else:
        print("Using random downsampling.")


def Project_Cloud(args):

    # 检测标志位
    check_cloud_status(args)

    # 检测output是否已存在
    prepare_output_dir(args.output)

    # 读取原始 JSON 文件
    data = load_json(args.input)

    # 读取相机名
    camera_names = filter_and_sort_keys(data['frames'][0]['sensor_data'],
                                        'cam_',
                                        args.cam_minFov, args.cam_maxFov)

    # 读取标定数据
    calibration_info = init_calibration_info(data["calibrated_sensors"],
                                             camera_names)

    # 获取要投影的相机内参
    calibration_data = extract_camera_calibration(data)

    # 第一帧位姿
    key_gnss2world = get_gnss_to_world(data["frames"][args.start_frame])

    # 获取点云
    points = get_points(data, calibration_info, key_gnss2world,
                        args.start_frame, args.end_frame,
                        args.plus,
                        args.downsample_ratio, args.random_downsample)
    points_except_currs = get_points_except_currs(args.start_frame,
                                                  args.end_frame,
                                                  args.downsample_ratio,
                                                  args.random_downsample)

    # 生成投影视频
    create_combined_video(args, data, calibration_data,
                          points, points_except_currs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--input", type=str, help="Input JSON file path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end_frame", type=int, default=100,
                        help="End frame number (default: 100)")
    parser.add_argument("--camera_id", type=str,
                        help="Additional information as string")
    parser.add_argument("--cam_minFov", type=int, default=100,
                        help="Camera min Fov (default: 100)")
    parser.add_argument("--cam_maxFov", type=int, default=200,
                        help="Camera max Fov (default: 200)")
    parser.add_argument("--plus", type=str2bool, nargs='?',
                        const=True, default=False, help="A boolean flag.")
    parser.add_argument("--height", type=int, default=1080,
                        help="Height of the video (default: 1080)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Width of the video (default: 1920)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frame rate of the video (default: 10)")
    parser.add_argument("--video_path", type=str, default="video.mp4",
                        help="Path to the output video file")
    parser.add_argument("--downsample_ratio", type=float, default=10,
                        help="Downsample ratio or voxel size (default: 10)")
    parser.add_argument("--random_downsample", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Whether to use random downsampling (default: True)")
    args = parser.parse_args()

    Project_Cloud(args)
