import re
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def parse_trajectory_file(filename, num_atoms=47656, num_frames=20, data_type="coordinates"):
    """
    解析轨迹文件（坐标、速度或力）

    参数:
    filename: 文件名
    num_atoms: 原子数量
    num_frames: 帧数
    data_type: 数据类型 ("coordinates", "velocities", "forces")

    返回:
    包含所有帧DataFrame的列表
    """

    # 读取文件内容
    with open(filename, 'r') as f:
        content = f.read()

    # 跳过第一行标题
    lines = content.split('\n')[1:]

    # 用于存储所有帧的DataFrame
    all_frames_dfs = []

    # 当前帧的数据列表
    current_frame = []
    frame_count = 0

    # 正则表达式匹配浮点数（包括负数）
    float_pattern = re.compile(r'-?\d+\.\d+')

    print(f"开始解析{data_type}文件: {filename}")

    for line in tqdm(lines, desc=f"处理{data_type}文件"):
        # 跳过空行
        if not line.strip():
            continue

        # 检查是否是帧分隔符（三个数字）
        numbers = float_pattern.findall(line)
        if len(numbers) == 3:
            # 如果当前帧有数据，保存它
            if current_frame:
                # 确保当前帧有足够的数据点
                if len(current_frame) == num_atoms * 3:
                    # 将当前帧的数据转换为DataFrame
                    frame_df = create_dataframe_from_data(current_frame, num_atoms, frame_count + 1, data_type)
                    all_frames_dfs.append(frame_df)
                    frame_count += 1
                    print(f"已处理第 {frame_count} 帧{data_type}")
                else:
                    print(f"警告: 第 {frame_count + 1} 帧{data_type}数据不完整，跳过")

                # 重置当前帧
                current_frame = []

                # 如果已经达到所需的帧数，停止处理
                if frame_count >= num_frames:
                    break
            continue

        # 提取当前行中的所有浮点数
        numbers = float_pattern.findall(line)
        current_frame.extend([float(num) for num in numbers])

    # 处理最后一帧（如果没有分隔符结尾）
    if current_frame and len(current_frame) == num_atoms * 3 and frame_count < num_frames:
        frame_df = create_dataframe_from_data(current_frame, num_atoms, frame_count + 1, data_type)
        all_frames_dfs.append(frame_df)
        frame_count += 1
        print(f"已处理第 {frame_count} 帧{data_type}")

    print(f"总共提取了 {len(all_frames_dfs)} 帧{data_type}数据")
    return all_frames_dfs


def create_dataframe_from_data(data_list, num_atoms, frame_num, data_type):
    """
    从数据列表创建Pandas DataFrame

    参数:
    data_list: 包含所有原子数据的列表
    num_atoms: 原子数量
    frame_num: 帧号
    data_type: 数据类型 ("coordinates", "velocities", "forces")

    返回:
    包含数据的DataFrame
    """
    # 将数据列表重塑为 (num_atoms, 3)
    data_array = np.array(data_list).reshape(num_atoms, 3)

    # 根据数据类型设置列名
    if data_type == "coordinates":
        columns = ['X_coord', 'Y_coord', 'Z_coord']
    elif data_type == "velocities":
        columns = ['X_vel', 'Y_vel', 'Z_vel']
    elif data_type == "forces":
        columns = ['Force_X', 'Force_Y', 'Force_Z']
    else:
        columns = ['X_coord', 'Y_coord', 'Z_coord']  # 默认

    # 创建DataFrame
    df = pd.DataFrame(data_array, columns=columns)

    # 添加原子序号和帧号
    df['Center_Number'] = range(1, num_atoms + 1)
    df['Frame'] = frame_num

    # 重新排列列的顺序
    df = df[['Frame', 'Center_Number'] + columns]

    return df


def merge_all_data(coords_dfs, velocities_dfs, forces_dfs):
    """
    合并坐标、速度和力数据

    参数:
    coords_dfs: 坐标数据的DataFrame列表
    velocities_dfs: 速度数据的DataFrame列表
    forces_dfs: 力数据的DataFrame列表

    返回:
    合并后的DataFrame列表
    """

    # 确保三个列表的长度相同
    num_frames = min(len(coords_dfs), len(velocities_dfs), len(forces_dfs))
    print(f"将合并 {num_frames} 帧数据")

    merged_dfs = []

    for i in tqdm(range(num_frames), desc="合并数据"):
        # 获取当前帧的数据
        coords_df = coords_dfs[i]
        velocities_df = velocities_dfs[i]
        forces_df = forces_dfs[i]

        # 验证帧号和原子号是否匹配
        if not (coords_df['Frame'].iloc[0] == velocities_df['Frame'].iloc[0] == forces_df['Frame'].iloc[0]):
            print(f"警告: 第{i + 1}帧的帧号不匹配")
            continue

        if not (coords_df['Center_Number'].equals(velocities_df['Center_Number']) and
                coords_df['Center_Number'].equals(forces_df['Center_Number'])):
            print(f"警告: 第{i + 1}帧的原子顺序不匹配")
            continue

        # 合并数据
        # 先合并坐标和速度
        merged_df = pd.merge(coords_df, velocities_df[['Center_Number', 'X_vel', 'Y_vel', 'Z_vel']],
                             on='Center_Number', how='left')

        # 再合并力
        merged_df = pd.merge(merged_df, forces_df[['Center_Number', 'Force_X', 'Force_Y', 'Force_Z']],
                             on='Center_Number', how='left')

        # 重新排列列的顺序
        merged_df = merged_df[['Frame', 'Center_Number', 'X_coord', 'Y_coord', 'Z_coord', 'X_vel', 'Y_vel', 'Z_vel', 'Force_X', 'Force_Y', 'Force_Z']]

        merged_dfs.append(merged_df)

    return merged_dfs


def save_merged_data_to_csv(merged_dfs, output_dir="merged_output"):
    """
    将合并后的数据保存为CSV文件

    参数:
    merged_dfs: 合并后的DataFrame列表
    output_dir: 输出目录
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("开始保存合并后的CSV文件...")

    # 为每一帧创建一个CSV文件
    for i, df in enumerate(tqdm(merged_dfs, desc="保存CSV文件")):
        frame_num = i + 1
        output_filename = os.path.join(output_dir, f"frame_{frame_num:03d}_merged.csv")

        # 使用Pandas保存为CSV
        df.to_csv(output_filename, index=False)

    print(f"所有合并帧已保存到 {output_dir} 目录")


def analyze_merged_data(merged_dfs):
    """
    对合并后的数据进行基本分析

    参数:
    merged_dfs: 合并后的DataFrame列表
    """

    print("合并数据分析:")
    print("=" * 60)

    # 检查第一帧的基本信息
    first_frame = merged_dfs[0]
    print(f"第一帧数据形状: {first_frame.shape}")
    print(f"第一帧数据列名: {list(first_frame.columns)}")
    print(f"第一帧前3行数据:\n{first_frame.head(3)}")

    # 统计所有帧的基本信息
    total_rows = sum(len(df) for df in merged_dfs)
    print(f"总数据行数: {total_rows}")

    # 计算坐标的统计信息
    all_coords = pd.concat([df[['X_coord', 'Y_coord', 'Z_coord']] for df in merged_dfs])
    print(f"坐标统计信息:")
    print(f"  X范围: [{all_coords['X_coord'].min():.3f}, {all_coords['X_coord'].max():.3f}]")
    print(f"  Y范围: [{all_coords['Y_coord'].min():.3f}, {all_coords['Y_coord'].max():.3f}]")
    print(f"  Z范围: [{all_coords['Z_coord'].min():.3f}, {all_coords['Z_coord'].max():.3f}]")

    # 计算速度的统计信息
    all_velocities = pd.concat([df[['X_vel', 'Y_vel', 'Z_vel']] for df in merged_dfs])
    print(f"速度统计信息:")
    print(f"  Vx范围: [{all_velocities['X_vel'].min():.3f}, {all_velocities['X_vel'].max():.3f}]")
    print(f"  Vy范围: [{all_velocities['Y_vel'].min():.3f}, {all_velocities['Y_vel'].max():.3f}]")
    print(f"  Vz范围: [{all_velocities['Z_vel'].min():.3f}, {all_velocities['Z_vel'].max():.3f}]")

    # 计算力的统计信息
    all_forces = pd.concat([df[['Force_X', 'Force_Y', 'Force_Z']] for df in merged_dfs])
    print(f"力统计信息:")
    print(f"  Fx范围: [{all_forces['Force_X'].min():.3f}, {all_forces['Force_X'].max():.3f}]")
    print(f"  Fy范围: [{all_forces['Force_Y'].min():.3f}, {all_forces['Force_Y'].max():.3f}]")
    print(f"  Fz范围: [{all_forces['Force_Z'].min():.3f}, {all_forces['Force_Z'].max():.3f}]")

def parse_centroid_file(centroid_file):
    """
    解析质心数据文件

    参数:
    centroid_file: 质心数据文件路径

    返回:
    字典，键为帧号，值为质心坐标 (x, y, z)
    """
    centroid_data = {}

    with open(centroid_file, 'r') as f:
        for line in f:
            # 跳过注释行和空行
            if line.startswith('#') or not line.strip():
                continue

            # 使用正则表达式匹配数字（包括负数）
            numbers = re.findall(r'-?\d+\.?\d*', line)

            if len(numbers) >= 6:  # 确保有足够的数据
                frame_num = int(float(numbers[0]))  # 帧号
                x, y, z = float(numbers[1]), float(numbers[2]), float(numbers[3])

                centroid_data[frame_num] = (x, y, z)

    print(f"从 {centroid_file} 中解析了 {len(centroid_data)} 帧的质心数据")
    print(centroid_data)
    return centroid_data


def add_centroid_to_csv_files(csv_dir, centroid_data, output_dir=None):
    """
    将质心数据添加到每帧的CSV文件中

    参数:
    csv_dir: 包含CSV文件的目录
    centroid_data: 质心数据字典
    output_dir: 输出目录（如果为None，则覆盖原文件）
    """

    if output_dir is None:
        output_dir = csv_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and f.startswith('frame_')]
    csv_files.sort()  # 按文件名排序

    print(f"找到 {len(csv_files)} 个CSV文件")

    processed_files = 0

    for csv_file in tqdm(csv_files, desc="处理CSV文件"):
        # 从文件名提取帧号
        try:
            frame_num = int(csv_file.split('_')[1])
        except (IndexError, ValueError):
            print(f"无法从文件名 {csv_file} 提取帧号，跳过")
            continue

        # 检查是否有该帧的质心数据
        if frame_num not in centroid_data:
            print(f"警告: 没有第 {frame_num} 帧的质心数据，跳过")
            continue

        # 读取CSV文件
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)

        # 获取质心坐标
        centroid_x, centroid_y, centroid_z = centroid_data[frame_num]

        # 创建质心数据行
        centroid_row = {
            'Frame': frame_num,
            'Center_Number': 47657,  # 最后一个原子序号
            'X_coord': centroid_x,
            'Y_coord': centroid_y,
            'Z_coord': centroid_z,
            'X_vel': 0.0,  # 速度设为0
            'Y_vel': 0.0,
            'Z_vel': 0.0,
            'Force_X': 0.0,  # 力设为0
            'Force_Y': 0.0,
            'Force_Z': 0.0
        }

        # 将质心行添加到DataFrame
        centroid_df = pd.DataFrame([centroid_row])
        updated_df = pd.concat([df, centroid_df], ignore_index=True)

        # 保存更新后的CSV文件
        output_path = os.path.join(output_dir, csv_file)
        updated_df.to_csv(output_path, index=False)

        processed_files += 1

    print(f"成功处理 {processed_files} 个CSV文件")
    return processed_files


# 专门用于处理已有CSV文件并添加质心的函数
def add_centroid_to_existing_csv(csv_directory, centroid_file, output_directory=None):
    """
    为已存在的CSV文件添加质心数据

    参数:
    csv_directory: 包含CSV文件的目录
    centroid_file: 质心数据文件
    output_directory: 输出目录（如果为None，则覆盖原文件）
    """

    # 解析质心数据
    centroid_data = parse_centroid_file(centroid_file)

    # 添加质心到CSV文件
    processed = add_centroid_to_csv_files(csv_directory, centroid_data, output_directory)

    print(f"为 {processed} 个CSV文件添加了质心数据")
    return processed


# 主程序
if __name__ == "__main__":
    # 输入文件路径
    coords_file = "coord.crd"  # 坐标文件
    velocities_file = "vel.crd"  # 速度文件
    forces_file = "force.crd"  # 力文件
    centroid_file = "center.dat" # 转子质心文件
    output_dir = "merged_coordinates_velocities_forces"

    # 解析三个文件
    print("开始解析轨迹文件...")
    coords_dfs = parse_trajectory_file(coords_file, num_atoms=47656, num_frames=20, data_type="coordinates")
    velocities_dfs = parse_trajectory_file(velocities_file, num_atoms=47656, num_frames=20, data_type="velocities")
    forces_dfs = parse_trajectory_file(forces_file, num_atoms=47656, num_frames=20, data_type="forces")

    # 合并数据
    print("开始合并坐标、速度和力数据...")
    merged_dfs = merge_all_data(coords_dfs, velocities_dfs, forces_dfs)

    # 数据分析
    analyze_merged_data(merged_dfs)

    # 保存为多个CSV文件（每帧一个文件）
    save_merged_data_to_csv(merged_dfs, output_dir)

    # 添加转子质心坐标
    add_centroid_to_existing_csv(output_dir, centroid_file)

    print("处理完成!")