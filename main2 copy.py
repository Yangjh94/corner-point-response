"""
此程序用于连接到当前打开的SAP2000模型,读取风荷载时程数据，使用模态法进行动力时程分析。
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import comtypes.client
import time
from datetime import datetime
from collections import defaultdict # 用于存储楼层信息, 方便创建刚性隔板

# 导入编写的函数工具
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
# Commented out as this import path appears to be invalid
from utils.io_utils.sap_model import SAP2000Model

def main():
    building_name = input("请输入模型名称（例如：3-1）: ").strip()  # 模型名称,
    number_modes = 15  # 模态数
    target_elevations = [6000, 10500, 15000, 19500, 23100, 26700, 30300, 33900, 37500, 41100, 44700, 48300, 51900, 55500, 
                         59100, 62700, 66300, 69900, 73500, 77100, 80700, 84300, 87900, 91500, 95100, 98700, 102300, 
                         105900, 109500, 113100, 116700, 120300, 123900, 127500, 131100, 134700, 138300, 141900, 145500, 
                         149100, 152700, 156300, 159900, 163500, 167100, 170700, 174300, 177900, 181500, 185100, 188700, 
                         192300, 196150, 200000]
    
    # 添加风荷载时程曲线，使用自定义风荷载时程文件
    wind_load_filepath = os.path.join("data", "raw", "WindloadTimes", building_name)

    # 自动读取building_name文件夹下的所有CSV文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wind_file_paths = os.path.join(script_dir, wind_load_filepath)
    # 获取文件夹下的风荷载时程数据文件
    wind_file = [f for f in os.listdir(wind_file_paths) if f.endswith('.csv')]
    
    # wind_file = ["Model2_10yr_000.csv", "Model2_10yr_005.csv"]  # 测试时可以只使用一个文件
    Type = "acceleration" # acceleration or displacement
    damp = 0.02 # 阻尼比

    # 记录程序开始时间
    start_time = time.time() # 记录开始时间
    start_datetime = datetime.now() # 获取当前时间
  
    print("=" * 80)
    print("SAP2000模型连接程序")
    print(f"程序开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # [1] 连接到SAP2000实例
    sapmodel = SAP2000Model(building_name)
    sapmodel.connect()

    # 连接到当前打开的SAP2000实例
    model = sapmodel.model
    if model is None:
        print("无法连接到SAP2000，程序终止")
        return
    
    # [2] 锁定/解锁模型
    locked = model.GetModelIsLocked()
    if locked:
        model.SetModelIsLocked(False)
        print("模型已解锁")
    else:
        print("模型未锁定")

    # 创建刚性隔板
    print("\n[步骤2] 创建刚性隔板...")
    # 指定要创建刚性隔板的楼层标高列表
    diaphragm_constraints, node_z_coords = sapmodel.add_diaphragms(target_elevations=target_elevations, tolerance=10)
    if diaphragm_constraints:
        print(f"成功创建刚性隔板: {diaphragm_constraints[0:3]} 等 {len(diaphragm_constraints)} 个隔板")
    else:
        print("创建刚性隔板失败")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    wind_file_paths = os.path.join(script_dir, wind_load_filepath)

    # 初始化结果存储列表
    all_results = []
    for wind_file_name in wind_file[:7]:
        # 解锁模型，以便修改荷载
        model.SetModelIsLocked(False)
        wind_file_path = os.path.join(wind_file_paths, wind_file_name)
        wind_load_count, diaphragm_centers = sapmodel.add_wind_time_history_load(
            number_modes,
            diaphragm_constraints,
            node_z_coords,
            wind_time_history_file=wind_file_path,
            num_rows=33000,
            damp=damp
        )
        
        if wind_load_count > 0:
            print(f"成功添加 {wind_load_count} 个风荷载时程曲线")
        else:
            print("添加风荷载时程曲线失败")
        
        # [4] 运行分析
        print("开启多线程求解器...")
        ret = model.Analyze.SetSolverOption_1(2,0,True)
        print(f"正在运行分析工况{wind_file_name}")
        ret = model.Analyze.RunAnalysis()
        if ret == 0:
            print("分析已成功完成")
        else:
            print(f"分析失败，返回代码: {ret}")

        # [5] 获取节点位移响应时程
        # 获取最高楼层的隔板名称
        top_diaphragm_center_name = max(diaphragm_centers.keys(), key=lambda x: float(x.split('_')[-1]))
        print(f"最高楼层的隔板名称: {top_diaphragm_center_name}")
        node_top_center_name = diaphragm_centers[top_diaphragm_center_name]["point_name"]
        if building_name == "1-1":
            target_nodes = [node_top_center_name, "54000062", "54000070", "54000071", "54000079"]
        elif building_name == "2-1":
            target_nodes = [node_top_center_name, "54000050", "54000059", "54000047", "54000058"]
        elif building_name == "3-1":
            target_nodes = [node_top_center_name, "54000060", "54000067", "54000059", "54000066"]
        
        # 在输出文件时包含 wind_file_path 中的文件名部分
        wind_file_name = os.path.basename(wind_file_path)
        results_dir = "data\\output\\Timehistory_modal"
        output_path = os.path.join(script_dir, results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for target_node in target_nodes:
            print(f"获取节点 {target_node} 的位移和加速度响应...")

            # 获取位移响应时程
            times, responses = sapmodel.get_node_response_history(
                building_name,
                target_node,
                Type=Type,
                damp=damp,
                load_case="Wind_time_history",
                load_history_file=wind_file_name,
                output_file=output_path
            )
            if times and responses:
                print(f"成功获取顶层角点 {target_node} 的 {len(times)} 个时间步的位移数据")
            else:
                print("获取位移响应失败")

    # 计算程序总耗时
    end_time = time.time()
    end_datetime = datetime.now()
    total_time = end_time - start_time
    
    print("=" * 80)
    print("程序执行完成")
    print(f"共运行了{len(wind_file)}个风荷载时程文件，分别为: {wind_file}")
    print(f"程序结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"程序总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print("=" * 80)

if __name__ == "__main__":
    main()
