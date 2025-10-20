import pandas as pd
import numpy as np
import os


def convert_gromacs_coords(input_file, output_dir):
    """
    转换GROMACS坐标文件为每帧独立的CSV文件
    参数:
        input_file: GROMACS输出坐标文件路径
    """
    with open(input_file, 'r') as f:
        frames = f.readlines()

    # 解析每帧数据
    for frame_idx, frame_data in enumerate(frames):
        # 分割数据并移除空值
        data = list(filter(None, frame_data.strip().split()))
        frame_number = data[0]  # 提取帧号
        coords = np.array(data[1:], dtype=float)  # 提取所有原子坐标

        # 验证数据完整性 (坐标数必须是3的倍数)
        if len(coords) % 3 != 0:
            raise ValueError(f"第{frame_idx} 帧坐标数量错误: 总数据点{len(coords)}不是3的倍数")

        # 创建原子编号 (从1开始)
        num_atoms = len(coords) // 3
        atom_numbers = np.arange(1, num_atoms + 1)

        # 重组为二维数组: [原子编号, x, y, z]
        frame_coords = np.column_stack((
            atom_numbers,
            coords[0::3],  # X坐标
            coords[1::3],  # Y坐标
            coords[2::3]  # Z坐标
        ))

        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(frame_coords, columns=['Center_Number', 'X_coord', 'Y_coord', 'Z_coord'])
        output_file = os.path.join(output_dir, f"frame_{frame_number}.csv")
        df.to_csv(output_file, index=False)
        print(f"已生成: {output_file} (原子数: {num_atoms})")


def convert_gromacs_vels(input_file, output_dir):
    """
    转换GROMACS速度文件为每帧独立的CSV文件
    参数:
        input_file: GROMACS输出速度文件路径
    """
    with open(input_file, 'r') as f:
        frames = f.readlines()

    # 解析每帧数据
    for frame_idx, frame_data in enumerate(frames):
        # 分割数据并移除空值
        data = list(filter(None, frame_data.strip().split()))
        frame_number = data[0]  # 提取帧号
        vels = np.array(data[1:], dtype=float)  # 提取所有原子速度

        # 验证数据完整性 (速度数必须是3的倍数)
        if len(vels) % 3 != 0:
            raise ValueError(f"第{frame_idx} 帧速度数量错误: 总数据点{len(vels)}不是3的倍数")

        # 创建原子编号 (从1开始)
        num_atoms = len(vels) // 3
        atom_numbers = np.arange(1, num_atoms + 1)

        # 重组为二维数组: [原子编号, x, y, z]
        frame_vels = np.column_stack((
            atom_numbers,
            vels[0::3],
            vels[1::3],
            vels[2::3]
        ))

        # 创建DataFrame并融合保存至输出文件
        df_vels = pd.DataFrame(frame_vels, columns=['Center_Number', 'X_vel', 'Y_vel', 'Z_vel'])
        output_file = os.path.join(output_dir, f"frame_{frame_number}.csv")
        df_coords = pd.read_csv(output_file)

        merged_df = pd.merge(
            df_coords,
            df_vels,
            on='Center_Number',
            how='inner'
        )

        merged_df.to_csv(output_file, index=False)
        print(f"已生成: {output_file} (原子数: {num_atoms})")


def convert_gromacs_forces(input_file,output_dir):
    """
    转换GROMACS力文件为每帧独立的CSV文件
    参数:
        input_file: GROMACS输出力文件路径
    """
    with open(input_file, 'r') as f:
        frames = f.readlines()

    # 解析每帧数据
    for frame_idx, frame_data in enumerate(frames):
        # 分割数据并移除空值
        data = list(filter(None, frame_data.strip().split()))
        frame_number = data[0]  # 提取帧号
        forces = np.array(data[1:], dtype=float)  # 提取所有原子速度

        # 验证数据完整性 (力数必须是3的倍数)
        if len(forces) % 3 != 0:
            raise ValueError(f"第{frame_idx} 帧力数量错误: 总数据点{len(forces)}不是3的倍数")

        # 创建原子编号 (从1开始)
        num_atoms = len(forces) // 3
        atom_numbers = np.arange(1, num_atoms + 1)

        # 重组为二维数组: [原子编号, x, y, z]
        frame_forces = np.column_stack((
            atom_numbers,
            forces[0::3],
            forces[1::3],
            forces[2::3]
        ))

        # 创建DataFrame并融合保存至输出文件
        df_forces = pd.DataFrame(frame_forces, columns=['Center_Number', 'Force_X', 'Force_Y', 'Force_Z'])
        output_file = os.path.join(output_dir, f"frame_{frame_number}.csv")
        df = pd.read_csv(output_file)

        merged_df = pd.merge(
            df,
            df_forces,
            on='Center_Number',
            how='inner'
        )

        merged_df.to_csv(output_file, index=False)
        print(f"已生成: {output_file} (原子数: {num_atoms})")

def convert_com_coords(input_file,output_dir):
    """
    添加质心坐标至csv文件
    """
    with open(input_file, 'r') as f:
        frames = f.readlines()

    # 解析每帧数据
    for frame_idx, frame_data in enumerate(frames):
        # 分割数据并移除空值
        data = list(filter(None, frame_data.strip().split()))
        frame_number = data[0]  # 提取帧号
        coords = np.array(data[1:], dtype=float)

        output_file = os.path.join(output_dir, f"frame_{frame_number}.csv")
        df = pd.read_csv(output_file)

        df.loc[df.shape[0]] = [47657, coords[0], coords[1], coords[2], 0, 0, 0, 0, 0, 0]
        df.to_csv(output_file, index=False)


# 使用示例
if __name__ == "__main__":
    coords_file = "coords.xvg" # 替换为实际文件路径
    vels_file = "velocities.xvg"
    forces_file = "forces.xvg"
    com_file = "com_coords.xvg"
    output_dir = "./"
    convert_gromacs_coords(coords_file, output_dir)
    convert_gromacs_vels(vels_file, output_dir)
    convert_gromacs_forces(forces_file, output_dir)
    convert_com_coords(com_file, output_dir)
