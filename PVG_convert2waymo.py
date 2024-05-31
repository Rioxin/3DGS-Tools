import argparse
import copy
import io
import json
import os
import shutil

import cv2
import nori2
import numpy as np
import open3d as o3d
import refile
import scipy.spatial.transform as st
import yaml
from numpy.lib import recfunctions as rfn
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def load_json(path):
    with refile.smart_open(path, "r") as f:
        return json.load(f)


def load_img_from_nid(nid):
    # img 是一个三维的 NumPy 数组，其形状是 (height, width, 3)
    nori_fetcher = nori2.Fetcher()
    img_bin = np.frombuffer(nori_fetcher.get(nid), dtype=np.uint8)
    img = cv2.imdecode(img_bin, 1)
    return img


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


def init_calibration_info(calibrated_data, camera_names):
    print("camera_names : ",camera_names)
    calibration_info = dict()
    calibration_info["ins2ego"] = get_transform_matrix(
        calibrated_data["gnss"], "gnss_ego")

    # 使用 tqdm 创建一个进度条迭代器
    with tqdm(total=len(camera_names), desc="Extracting Calibration Info") as pbar:
        for cam in camera_names:
            cam_rfu2ego = get_transform_matrix(
                calibrated_data[cam], "extrinsic")
            calibration_info[cam] = cam_rfu2ego
            pbar.update(1)  # 更新进度条
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


def extract_vehicle_poses(data, start_frame, end_frame):
    vehicle_poses = []
    frames = data.get("frames", [])
    frames = frames[start_frame:end_frame]  # 仅保留指定区间内的帧数据

    with tqdm(total=len(frames), desc="Extracting Vehicle Poses") as pbar:
        for frame in frames:
            ins_data = frame.get("ins_data", {})
            timestamp = ins_data.get("timestamp")
            localization = ins_data.get("localization", {})
            position = localization.get("position", {}) if localization else {}
            orientation = localization.get(
                "orientation", {}) if localization else {}
            if timestamp and position and orientation:
                vehicle_poses.append(
                    {"timestamp": timestamp, "position": position, "orientation": orientation})

            pbar.update(1)  # 更新进度条
    return vehicle_poses


def ins2ego(data):
    calibrated_sensors = data.get("calibrated_sensors", {})
    gnss = calibrated_sensors.get("gnss", {})
    gnss_ego = gnss.get("gnss_ego", {})
    transform = gnss_ego.get("transform", {})

    return transform


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
    trans = np.array([trans["x"], trans["y"], trans["z"]], dtype=np.float64)
    quan = np.array([quan["x"], quan["y"], quan["z"],
                    quan["w"]], dtype=np.float64)
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


# 全局变量，用于存储 map1 和 map2
MAP_CACHE = {}

def warp_and_save(wm_img_nid, intrinsic, time, save_dir, camera_name, R=None):
    global MAP_CACHE

    # 加载图像
    ori_img = load_img_from_nid(wm_img_nid)
    dim = intrinsic["resolution"]
    K, D = np.array(intrinsic["K"]), np.array(intrinsic["D"])
    
    # 修改相机内参
    new_dim = (1920, 1080)  # 新的图像尺寸
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
    if camera_name == "cam_front_120":
        cropped_img = img[:1080*3//4, :]
        img = cropped_img
    # 检查保存目录是否存在，如果不存在则创建它
    os.makedirs(save_dir, exist_ok=True)
    
    frame_name = str(time) +".png"  # Convert timestamp to string
    save_path = os.path.join(save_dir, frame_name)
    cv2.imwrite(save_path, img)
    return new_K

def _save_point_cloud(sensor_data, save_dir, timestamps):
    """
    Save point cloud data to specified directory in binary format.

    Args:
        sensor_data (dict): Sensor data containing the nori_id and timestamp
        save_dir (str): Directory to save the point cloud data
    """
    nori_id = sensor_data["nori_id"]
    # 以时间戳命名
    nori_fetcher = nori2.Fetcher()
    data = nori_fetcher.get(nori_id)
    pc_data_raw = np.load(io.BytesIO(data)).copy()

    # 确保有文件夹
    os.makedirs(save_dir, exist_ok=True)
    # 保存成bin格式
    save_path = os.path.join(save_dir, f"{timestamps}.bin")
    pc_data_raw.tofile(save_path)


def save_nori_paths(nori_paths, output_dir):
    # 指定保存的txt文件路径
    txt_file_path = os.path.join(output_dir, "nori.txt")

    # 将nori_paths写入到txt文件中
    with open(txt_file_path, 'w') as f:
        for path in nori_paths:
            f.write(path + '\n')


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


def extract_and_write_camera_pose(json_file_path, output_file, start_frame, end_frame):
    # 检测output是否已存在
    prepare_output_dir(output_file)

    # 读取原始 JSON 文件
    data = load_json(json_file_path)

    # 读取camera_id
    camera_names = filter_and_sort_keys(data["calibrated_sensors"], 'cam_', 100, 120)
    
    # 读取标定数据
    calibration_info = init_calibration_info(data["calibrated_sensors"], camera_names)

    # 获取车辆的姿态数据
    vehicle_poses = extract_vehicle_poses(data, start_frame, end_frame)
    calibration_data = extract_camera_calibration(data)

    # 第一帧作为原点
    key_gnss2world = get_gnss_to_world(data["frames"][start_frame])

    # 时间戳dict
    formatted_timestamp = {}
    # 逐相机写入内参，保存去畸变图片，写入pose
    os.makedirs(os.path.join(output_file, "calib"), exist_ok=True)
    os.makedirs(os.path.join(output_file, "pose"), exist_ok=True)
    camera_nori_paths = set()
    for count, camera_name in enumerate(camera_names):
        camera_info = data.get("calibrated_sensors", {}).get(camera_name, {})
        camera_extrinsic = camera_info.get("extrinsic", {})
        camera_intrinsic = camera_info.get("intrinsic", {})
        image_path = "image_" + str(count)
        output_image_file = os.path.join(output_file, image_path)
        if camera_extrinsic and vehicle_poses:
            for nums, vehicle_pose in enumerate(vehicle_poses, start=start_frame):
                timestamp = data['frames'][nums]['sensor_data']['cam_front_120']['timestamp']
                # timestamp = vehicle_pose["timestamp"]
                timestamp_float = float(timestamp)
                formatted_timestamp[nums] = "{:.2f}".format(timestamp_float).replace(".", "")
                # 读取nori
                nori_id = data["frames"][nums]["sensor_data"][camera_name]["nori_id"]
                nori_path = data["frames"][nums]["sensor_data"][camera_name]["nori_path"]
                camera_nori_paths.add(nori_path)

                # 去畸变
                new_K = warp_and_save(nori_id, calibration_data[camera_name], formatted_timestamp[nums], output_image_file, camera_name, None)
                new_K_list = new_K.tolist()
                new_K_str = "P{}: {:e} {:e} {:e} 0.000000e+00 {:e} {:e} {:e} 0.000000e+00 {:e} {:e} {:e} 0.000000e+00".format(count, *new_K.flatten())

                # 写相机内参
                txt_file = os.path.join(output_file, "calib", formatted_timestamp[nums] + ".txt")
                with open(txt_file, 'a') as f:
                    f.write(new_K_str + "\n")

                # 替换camera_intrinsic["K"]
                camera_intrinsic["K"] = new_K_list

                # 计算pose
                sweep_gnss2world = get_gnss_to_world(vehicle_pose)
                Transform_curr = get_sweeprfu2keyrfu(
                    calibration_info, key_gnss2world, sweep_gnss2world)

                # 写车的pose
                pose_file = os.path.join(output_file, "pose", formatted_timestamp[nums] + ".txt")
                with open(pose_file, 'w') as f:
                    for row in Transform_curr:
                        row_str = " ".join("{:.18e}".format(item) for item in row)
                        f.write(row_str + "\n")

    # 逐相机写入外参 
    for count, camera_name in enumerate(camera_names):
        camera_info = data.get("calibrated_sensors", {}).get(camera_name, {})
        camera_extrinsic = camera_info.get("extrinsic", {})
        if camera_extrinsic and vehicle_poses:
            for nums, vehicle_pose in enumerate(vehicle_poses, start=start_frame):
                # 写入外参
                lidar2camera = get_camera2rfu_transform(camera_extrinsic)
                Tr_velo_to_cam = "Tr_velo_to_cam_{}: {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e}".format(count, *lidar2camera.flatten())
                # 写相机内参
                txt_file = os.path.join(output_file, "calib", formatted_timestamp[nums] + ".txt")
                with open(txt_file, 'a') as f:
                    f.write(Tr_velo_to_cam + "\n")

    # 逐帧点云保存
    lidar_nori_paths = set()
    output_cloud_file = os.path.join(output_file, "velodyne")
    for frame_idx in range(start_frame, end_frame, 1):
        # 获取当前帧的点云数据
        fuser_lidar_curr = data["frames"][frame_idx]["sensor_data"]["fuser_lidar"]
        # 保存点云数据
        _save_point_cloud(fuser_lidar_curr, output_cloud_file, formatted_timestamp[frame_idx])
        nori_path = fuser_lidar_curr["nori_path"]
        lidar_nori_paths.add(nori_path)
    # 写nori_path到txt
    save_nori_paths(lidar_nori_paths, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--input", type=str, help="Input JSON file path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end_frame", type=int, default=100,
                        help="End frame number (default: 100)")
    args = parser.parse_args()

    extract_and_write_camera_pose(
        args.input, args.output, args.start_frame, args.end_frame
    )
