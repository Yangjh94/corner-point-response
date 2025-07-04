'''
本程序用于计算建筑表明的等效静力风荷载。
'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt  # 添加导入
import os
import pandas as pd
from pathlib import Path

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def read_wind_load_files(folder_path):
    """
    读取指定文件夹中的所有CSV风荷载文件
    
    参数:
        folder_path: Path对象或字符串，风荷载文件所在的文件夹路径
        
    返回:
        wind_loads: 字典，键为文件名，值为DataFrame格式的风荷载数据
    """
    wind_loads = {}
    folder_path = Path(folder_path) if not isinstance(folder_path, Path) else folder_path
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"错误：{folder_path} 不是有效的文件夹路径")
        return {}
    
    # 获取文件夹中所有的CSV文件
    csv_files = [f for f in folder_path.glob("*.csv") if f.is_file()]
    
    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return {}
    
    # 读取所有CSV文件
    for file_path in csv_files:
        try:
            print(f"正在读取文件：{file_path.name}")
            data = pd.read_csv(file_path)
            wind_loads[file_path.name] = data
            print(f"成功读取文件：{file_path.name}，包含 {len(data)} 行数据")
        except Exception as e:
            print(f"读取文件 {file_path.name} 时发生错误：{e}")
            continue
    
    return wind_loads

def process_wind_load_data(wind_loads):
    """
    处理风荷载数据，使其符合SAP2000模型所需的格式
    
    参数:
        wind_loads: 字典，键为文件名，值为DataFrame格式的风荷载数据
        
    返回:
        processed_loads: 字典，处理后的风荷载数据，包含时程信息和结构信息
    """
    processed_loads = {}
    
    for file_name, data in wind_loads.items():
        print(f"正在处理文件：{file_name}")
        try:
            # 检查数据格式，假设CSV文件包含以下列：
            # 时间步、楼层/节点编号、X方向力、Y方向力、Z方向力等
            required_columns = ['时间', '楼层', 'Fx', 'Fy', 'Fz']
            
            # 尝试标准化列名（处理不同命名方式）
            column_mapping = {}
            for col in data.columns:
                col_lower = col.lower()
                if '时间' in col_lower or 'time' in col_lower:
                    column_mapping[col] = '时间'
                elif '楼层' in col_lower or 'floor' in col_lower or '节点' in col_lower or 'node' in col_lower:
                    column_mapping[col] = '楼层'
                elif 'fx' in col_lower or 'x' in col_lower and '力' in col_lower or 'force_x' in col_lower:
                    column_mapping[col] = 'Fx'
                elif 'fy' in col_lower or 'y' in col_lower and '力' in col_lower or 'force_y' in col_lower:
                    column_mapping[col] = 'Fy'
                elif 'fz' in col_lower or 'z' in col_lower and '力' in col_lower or 'force_z' in col_lower:
                    column_mapping[col] = 'Fz'
            
            # 重命名列
            if column_mapping:
                data = data.rename(columns=column_mapping)
            
            # 检查是否包含必要的列
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"警告：文件 {file_name} 缺少必要的列：{missing_columns}")
                # 尝试基于列的位置进行映射
                if len(data.columns) >= 5:  # 至少应该有5列
                    print("尝试基于列的位置进行映射...")
                    data.columns = required_columns + list(data.columns)[5:]
                else:
                    print(f"无法处理文件 {file_name}，列数不足")
                    continue
            
            # 提取时程信息
            time_steps = data['时间'].unique()
            floors = data['楼层'].unique()
            
            # 创建处理后的数据结构
            processed_data = {
                'time_steps': time_steps,
                'floors': floors,
                'forces': {}
            }
            
            # 按楼层组织力数据
            for floor in floors:
                floor_data = data[data['楼层'] == floor]
                forces = {
                    'Fx': floor_data['Fx'].values,
                    'Fy': floor_data['Fy'].values,
                    'Fz': floor_data['Fz'].values if 'Fz' in data.columns else np.zeros(len(floor_data))
                }
                processed_data['forces'][floor] = forces
            
            processed_loads[file_name] = processed_data
            print(f"成功处理文件：{file_name}，包含 {len(floors)} 个楼层，{len(time_steps)} 个时间步")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误：{e}")
            continue
    
    return processed_loads

def calculate_wind_load(analysis_type, node_coor=None, load_path=None, keyword=None):
    '''
    计算风荷载的时程数据、均值、标准差和极值

    参数:
    load_path: str, .mat文件的路径
    keyword: str, .mat文件中荷载数据的关键字

    返回:
    wind_load_time: np.ndarray, 风荷载时程数据
    wind_load_mean: np.ndarray, 风荷载均值
    wind_load_std: np.ndarray, 风荷载标准差
    wind_load_peak: np.ndarray, 风荷载极值 (均值加上3.5倍的标准差)
    '''
    # 设置分析类型：1为静力计算，2为动力计算, 3表示其它类型，可根据需要扩展
    if analysis_type == 1:

        # 静力计算中直接根据高度计算荷载
        # 这里假设节点坐标为三维坐标[x, y, z]，z为高度
        load_values = []
        for i, z in enumerate(node_coor):
            if z[2] < 10:
                load_value = [0, 0, 0, 0, 0, 0]
            else:
                load_value = [245.7 * (z[2] / 10000) ** 0.30, 379.08 * (z[2] / 10000) ** 0.30, 0, 0, 0, 0]
            load_values.append(load_value)
        return load_values
    
    elif analysis_type == 2:

        # 读取.mat文件中的数据
        wind_load_data = scipy.io.loadmat(load_path)
        # print(wind_load_data.keys())

        wind_load_time = wind_load_data[keyword]

        # 计算均值和标准差
        wind_load_mean = np.mean(wind_load_time, axis=1) # axis=1表示按行计算均值，axis=0表示按列计算均值
        wind_load_std = np.std(wind_load_time, axis=1)

        wind_load_peak = np.abs(wind_load_mean) + 3.5 * wind_load_std

        return wind_load_time, wind_load_mean, wind_load_std, wind_load_peak

    else:
        raise ValueError("无效的计算类型。请使用1(静力)或2(动力)。")  # 显示错误信息，并终止程序

if __name__ == '__main__':
    
    # 设置分析类型：1为静力计算，2为动力计算
    analysis_type = 1  # 默认为静力计算(1)
    print(f"使用分析类型: {analysis_type} ({'静力分析' if analysis_type == 1 else '动力分析'})")
    
    # 测试计算节点荷载值
    # 创建简单的测试节点坐标
    test_node_coor = [
        [0, 0, 0],         # 地面节点
        [0, 0, 5000],      # 5m高度节点
        [0, 0, 10000],     # 10m高度节点
        [0, 0, 50000],     # 50m高度节点
        [0, 0, 100000]     # 100m高度节点
    ]
    if analysis_type == 1:
        # 根据分析类型计算荷载值
        test_load_values = calculate_wind_load(analysis_type, test_node_coor)
        print("\n根据分析类型计算的节点荷载值:")
        for i, load in enumerate(test_load_values):
            print(f"节点 {i+1} (高度: {test_node_coor[i][2]/1000}m): {load}")
    elif analysis_type == 2:
        # 测试计算风荷载
        load_path = r'd:\MyFiles\PhD\00模型与荷载\01CAARC\CAARC_F50.mat'
        keyword = 'F50_00'
        wind_load_time, wind_load_mean, wind_load_std, wind_load_peak = calculate_wind_load(analysis_type, test_node_coor, load_path, keyword)

        print(f"Wind Load shape: {wind_load_time.shape}")
        print(f"Wind Load Mean shape: {wind_load_mean.shape}")
        print(f"Wind Load Std shape: {wind_load_std.shape}")
        print(f"Wind Load Peak shape: {wind_load_peak.shape}")
    
        # 选择一个代表性节点进行时程可视化 (假设是第一个节点)
        node_index = 180-3
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(wind_load_time[node_index, :], 'k-', label=f'节点 {node_index//3 + 1} X方向')
        plt.plot(wind_load_time[node_index+1, :], 'b-', label=f'节点 {node_index//3 + 1} Y方向')
        plt.plot(wind_load_time[node_index+2, :], 'g-', label=f'节点 {node_index//3 + 1} Z方向')
        plt.axhline(y=wind_load_mean[node_index], color='r', linestyle='--', label='均值 X')
        plt.axhline(y=wind_load_mean[node_index+1], color='r', linestyle='--', label='均值 Y')
        plt.axhline(y=wind_load_mean[node_index+2], color='r', linestyle='--', label='均值 Z')
        plt.xlabel('时间步')
        plt.ylabel('风荷载 (kN)')
        plt.legend()
        plt.title('风荷载时程')
        plt.grid()
        
        # 极值分布
        plt.subplot(2, 2, 2)
        plt.plot(wind_load_peak[0::3], 'r', label='X')
        plt.plot(wind_load_peak[1::3], 'g', label='Y')
        plt.plot(wind_load_peak[2::3], 'b', label='Z')
        plt.xlabel('节点编号')
        plt.ylabel('风荷载极值 (kN)')
        plt.legend()
        plt.title('风荷载极值 (均值+3.5×标准差)')
        plt.grid()
        
        # 均值分布
        plt.subplot(2, 2, 3)
        plt.plot(wind_load_mean[0::3], 'r', label='X')
        plt.plot(wind_load_mean[1::3], 'g', label='Y')
        plt.plot(wind_load_mean[2::3], 'b', label='Z')
        plt.xlabel('节点编号')
        plt.ylabel('风荷载均值 (kN)')
        plt.legend()
        plt.title('风荷载均值分布')
        plt.grid()
        
        # 标准差分布
        plt.subplot(2, 2, 4)
        plt.plot(wind_load_std[0::3], 'r', label='X')
        plt.plot(wind_load_std[1::3], 'g', label='Y')
        plt.plot(wind_load_std[2::3], 'b', label='Z')
        plt.xlabel('节点编号')
        plt.ylabel('风荷载标准差 (kN)')
        plt.legend()
        plt.title('风荷载标准差分布')
        plt.grid()
        
        plt.tight_layout()
        plt.show()

    # 保存数据示例
    # np.savetxt(r'd:\MyFiles\PhD\00模型与荷载\01CAARC\CAARC_F50_peak.txt', wind_load_peak, fmt='%s')
    # np.savetxt(r'd:\MyFiles\PhD\00模型与荷载\01CAARC\CAARC_F50_mean.txt', wind_load_mean, fmt='%s')
    # np.savetxt(r'd:\MyFiles\PhD\00模型与荷载\01CAARC\CAARC_F50_std.txt', wind_load_std, fmt='%s')