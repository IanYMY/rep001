import pandas as pd
import numpy as np

def extract_time_series(input_file):
    with open(input_file, 'r') as f:
        frames = f.readlines()

    times = []

    for frame_data in frames:
        # 分割数据并移除空值
        data = list(filter(None, frame_data.strip().split()))
        time = data[0]
        times.append(time)

    return times


def calculate_axis_torque(input_csv, output_csv, center_numberA, center_numberB):

    # 获取确定轴的两个原子坐标
    df = pd.read_csv(input_csv)
    A = df[df['Center_Number'] == center_numberA][['X_coord', 'Y_coord', 'Z_coord']].values[0]
    B = df[df['Center_Number'] == center_numberB][['X_coord', 'Y_coord', 'Z_coord']].values[0]
    O = df[df['Center_Number'] == 47657][['X_coord', 'Y_coord', 'Z_coord']].values[0]
    AB = B - A
    OA = A - O

    # 归一化轴向量
    axis_norm = AB / np.linalg.norm(AB)
    distance = np.linalg.norm(np.cross(OA, axis_norm))


    print(f"质心坐标: [{O[0]}, {O[1]}, {O[2]}]")
    print(f"轴向量由原子 {center_numberA} 和 {center_numberB} 确定")
    print(f"轴方向向量: [{axis_norm[0]:.10f}, {axis_norm[1]:.10f}, {axis_norm[2]:.10f}]")
    print(f"质心与轴的距离: {distance}")
    # 计算每个原子的位置向量（相对于轴上的点）
    r_vectors = df[['X_coord', 'Y_coord', 'Z_coord']].values - O

    # 计算每个原子的力矩向量
    torque_vectors = np.cross(r_vectors, df[['Force_X', 'Force_Y', 'Force_Z']].values) * 1.6605

    # 计算每个原子相对于轴的力矩（标量，力矩在轴上的投影）
    df['Torque_Axis'] = np.dot(torque_vectors, axis_norm)

    # 计算总力矩
    total_torque_axis = df['Torque_Axis'].sum()

    # 计算特定原子范围的力矩和
    rotator_torque_axis = df[df['Center_Number'].between(43171, 47656)]['Torque_Axis'].sum()

    # 保存结果
    df.to_csv(output_csv, index=False)

    # 打印摘要信息
    print(f"轴总力矩: {total_torque_axis}")
    print(f"转子部分轴总力矩: {rotator_torque_axis}")

    # 返回计算结果
    return df, total_torque_axis, rotator_torque_axis


if __name__ == "__main__":

    Center_Number_A = 47605
    Center_Number_B = 44603

    times = extract_time_series('coords.xvg')
    print(f"time series: {times}")

    total_torques = []
    rotator_torques = []

    for time in times:
        # 输入和输出文件路径
        input_csv = "frame_" + str(time) + ".csv" # 替换为您的输入文件路径
        output_csv = "frame_" + str(time) + "_torque.csv" # 输出文件路径

        # 执行计算
        result_df, total_torque, rotator_torque = calculate_axis_torque(input_csv, output_csv, Center_Number_A, Center_Number_B)
        total_torques.append(total_torque)
        rotator_torques.append(rotator_torque)

        # 打印前几个原子的力矩值
        print("\n前5个原子的力矩值:")
        print(result_df[['Center_Number', 'Torque_Axis']].head())

    # 将个时间点的力矩保存至一个csv文件
    torque_data = np.column_stack((
            times,
            total_torques,
            rotator_torques
        ))
    df_torque = pd.DataFrame(torque_data, columns=['time', 'protein_torque', 'rotator_torque'])
    df_torque.to_csv("torque_summary.csv", index=False)

