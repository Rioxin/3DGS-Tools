import argparse
import json
import os
import pickle
import refile
import numpy as np
from scipy.spatial.transform import Rotation as R


# 将旋转矩阵转换为四元数
def rotation_matrix_to_quaternion(rotation_matrix):
    rotation = R.from_matrix(rotation_matrix)
    # 将旋转对象转换为四元数
    quaternion = rotation.as_quat()
    return quaternion


def save_array_to_txt(array, folder_path, counter):
    # 构造文件名
    filename = os.path.join(folder_path, f"{counter}.txt")
    # 将 ndarray 对象保存为文本文件
    np.savetxt(filename, array, fmt='%.18e', delimiter=' ', newline='\n')


def save_as_json(obj, filename):
    # 将 NumPy ndarray 转换为列表
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # 递归地将对象中的 ndarray 转换为列表
    def convert_recursive(obj):
        if isinstance(obj, dict):
            return {k: convert_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_recursive(v) for v in obj]
        else:
            return convert_to_json_serializable(obj)

    # 转换对象
    obj = convert_recursive(obj)

    # 将对象转换为 JSON 字符串并保存到文件
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)  # 使用 json.dump() 函数保存 JSON 文件


def load_from_pickle(path):
    with refile.smart_open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with refile.smart_open(path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("--pickle_path", required=True, help="Path to the pickle file")
    parser.add_argument("--json_path", required=True, help="Path to the original JSON file")
    parser.add_argument("--output_json_path", required=True, help="Path to save the modified JSON file")
    parser.add_argument("--output_txt_folder", required=True, help="Folder path to save text files")
    parser.add_argument("--initial_counter", type=int, required=True, help="Initial counter value")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载对象
    obj = load_from_pickle(args.pickle_path)

    # 读取原始 JSON 文件
    data = load_json(args.json_path)

    for i, frame in enumerate(obj):
        # 提取旋转矩阵的前三行作为旋转部分
        rotation_matrix = frame[:3, :3]

        # 将旋转矩阵转换为四元数
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)

        # 将四元数放入 data 中对应帧的位姿信息中
        data['frames'][i]['ins_data']['localization']['orientation']['x'] = quaternion[0]
        data['frames'][i]['ins_data']['localization']['orientation']['y'] = quaternion[1]
        data['frames'][i]['ins_data']['localization']['orientation']['z'] = quaternion[2]
        data['frames'][i]['ins_data']['localization']['orientation']['w'] = quaternion[3]

        # 将位移放入 data 中对应帧的位姿信息中
        data['frames'][i]['ins_data']['localization']['position']['x'] = frame[0, 3]
        data['frames'][i]['ins_data']['localization']['position']['y'] = frame[1, 3]
        data['frames'][i]['ins_data']['localization']['position']['z'] = frame[2, 3]

    # 保存为 JSON 文件
    save_as_json(data, args.output_json_path)

    counter = args.initial_counter

    # 遍历列表中的每个 ndarray 对象，并将其保存为文本文件
    for array in obj:
        save_array_to_txt(array, args.output_txt_folder, counter)
        counter += 10  # 每次递增10


if __name__ == "__main__":
    main()
