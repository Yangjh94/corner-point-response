"""
简化版风振分析数据处理程序 - 专门处理Ux、Uy、Rz三个方向的响应数据
"""
# 导入所需的库
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入编写的函数工具
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
from utils.io_utils.utils import *
    
# 设置文件路径和目标列
building_name = "1-1"  # 模型名称
analysis_type = "Acceleration"  # 分析类型
path_Now = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(os.getcwd(), "data", "output", "Timehistory_modal", building_name, analysis_type)
target_columns = ['UX', 'UY', 'RZ']  # 目标分析列

# 获取所有文件夹名称
folder_names = os.listdir(output_folder)

# 存储结果的列表
results_data = []
angles = []
plot_data = {'mean': {col: [] for col in target_columns}, 
             'std': {col: [] for col in target_columns}, 
             'extreme': {col: [] for col in target_columns}}

# 处理每个文件夹（代表不同角度）
for folder_name in folder_names:
    print(f"\n处理文件夹: {folder_name}")
    
    # 提取角度信息
    angle = int(folder_name.split('_')[-1])
    angles.append(angle)
    
    # 获取文件夹下的CSV文件
    folder_path = os.path.join(output_folder, folder_name)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    last_5_files = csv_files[-5:]  # 取最后5个文件
    
    # 为每个文件创建一行数据，包含所有指标的统计量
    file_result = {
        'folder':folder_name,
        'angle':angle,
    }
    points_name = [] # 存储点名
    # 处理每个CSV文件
    for i, csv_file in enumerate(last_5_files, 1):
        print(f"  处理第{i}个文件: {csv_file}")
        
        file_without_ext = csv_file.replace('.csv', "")  # 去掉.csv
        parts = file_without_ext.split('_')
        last_number = int(parts[-2])
        points_name.append(last_number)

        # 读取数据并筛选目标列
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        
        # 检查目标列是否存在
        available_cols = [col for col in target_columns if col in data.columns]
        if not available_cols:
            print(f"    警告：文件中没有找到目标列，跳过")
            continue
        
        # 计算统计值并保存结果
        file_data = data[available_cols]

        # 为每个可用列添加mean、std和extreme值
        for col in available_cols:
            mean_val = file_data[col].mean()
            std_val = file_data[col].std()
            
            g = g_D(file_data[col], dt=1/8.3227)
            extreme_val = mean_val + g * std_val
            
            # 保存详细结果
            # file_result[f'P{last_number}_{col}_mean'] = round(mean_val, 2)/1000  # 转换为m
            # file_result[f'P{last_number}_{col}_std'] = round(std_val, 2)/1000  # 转换为m
            file_result[f'P{last_number}_{col}_extreme'] = round(extreme_val, 2)/1000  # 转换为m

        range_2D = CDC(file_data['UX'].values, file_data['UY'].values, tDlt=1/8.3227)
        file_result[f'P{last_number}_range_2D_extreme'] = round(range_2D, 2)/1000  # 转换为m
    # 添加完整的文件结果到列表
    results_data.append(file_result)
        
# 打印结果预览
results_df = pd.DataFrame(results_data)
print("\n结果预览:")
print(results_df.head())

# 结果处理，获得中心点极值和角点极值

# 创建可视化图表
if results_data:

    print(f"\n开始绘图...")
    for point in points_name:
        print(f"  绘制点: P{point}")

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False    
        
        # 创建三个子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        
        # # 绘制均值图
        # for col in target_columns:
        #     ax1.plot(angles, results_df[f'P{point}_{col}_mean'], marker='o', label=f'{col}_均值', linewidth=2)
        # ax1.set_xlabel('角度 (度)')
        # ax1.set_ylabel('均值')
        # ax1.set_title(f'节点P{point}各角度下响应均值变化')
        # ax1.legend()
        # ax1.grid(True, alpha=0.3)
        
        # # 绘制标准差图
        # for col in target_columns:
        #     ax2.plot(angles, results_df[f'P{point}_{col}_std'], marker='s', label=f'{col}_标准差', linewidth=2)
        # ax2.set_xlabel('角度 (度)')
        # ax2.set_ylabel('标准差')
        # ax2.set_title(f'节点P{point}各角度下响应标准差变化')
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)
        
        # 绘制极值图
        for col in target_columns:
            ax3.plot(angles, results_df[f'P{point}_{col}_extreme'], marker='^', label=f'{col}_极值', linewidth=2, markersize=8)
        ax3.set_xlabel('角度 (度)')
        ax3.set_ylabel('极值 (均值+3.5×标准差)')
        ax3.set_title(f'节点P{point}各角度下响应极值变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_parent = os.path.dirname(output_folder)

        plot_path = os.path.join(output_parent, f"节点P{point}响应分析图.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {plot_path}")
    # plt.show()

# 保存结果到CSV文件
if results_data:
    # 转换为DataFrame
    results_df = pd.DataFrame(results_data)
    
    print("\n结果预览:")
    print(results_df.head())
    
    # 保存Excel文件
    output_parent = os.path.dirname(output_folder)
    excel_path = os.path.join(output_parent, "响应统计结果.xlsx")

    # 指定要保存的sheet名称
    sheet_name = f"{analysis_type}"
    if os.path.exists(excel_path): # 如果文件已存在，则追加数据
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            results_df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:  # 如果文件不存在，则创建新文件
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"结果已保存: {excel_path}")

    # 打印统计信息
    folder_count = len(set(results_df['folder']))
    print(f"\n处理完成!")
    print(f"共处理了 {folder_count} 个角度的数据")
    # print(f"处理的指标: {', '.join(processed_metrics)}")
    
else:
    print("没有生成有效的结果数据")

