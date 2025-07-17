"""
此程序用于连接到当前打开的SAP2000模型,读取风荷载时程数据，进行动力时程分析。
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
    building_name = input("请输入建筑名称: 例如 3-1: ").strip()  # 模型名称,
    number_modes = 15  # 模态数
    SapModel = SAP2000Model(building_name)
    # 连接到SAP2000
    SapModel.connect()
    ret = SapModel.model.Analyze.SetRunCaseFlag("MODAL", True)
    ret = SapModel.model.Analyze.SetRunCaseFlag("DEAD", False)
    ret = SapModel.model.Analyze.SetRunCaseFlag("LIVE", False)
    ret = SapModel.model.Analyze.SetRunCaseFlag("Wind_time_history", False)
    ret = SapModel.model.Analyze.RunAnalysis()

    # 获取模型的所有节点坐标信息
    node_coords, diaphragm_constraint_coords = SapModel.get_node_coordinates()
    # =================打印最后10个节点的坐标信息=========================
    # print("\n最后10个节点的坐标信息:")
    # for point_name in list(node_coords.keys())[-10:]:
    #     x, y, z = node_coords[point_name]
    #     print(f"节点 {point_name}: 坐标 = ({x:.3f}, {y:.3f}, {z:.3f})")

    # 获取节点质量
    node_masses = SapModel.get_node_mass()
    # =================打印最后20个节点的质量信息=========================
    print("\n最后20个节点的质量信息:")
    for point_name in list(node_masses.keys())[-20:]:
        print(f"节点 {point_name}: 质量 = {node_masses[point_name]}")

    # 将node_coords和node_masses中标高一直的节点质量进行求和
    Floor_Masses = {}
    target_Floor_z = [6000, 10500, 15000, 19500, 23100, 26700, 30300, 33900, 37500, 41100, 44700, 48300, 51900, 55500, 
                         59100, 62700, 66300, 69900, 73500, 77100, 80700, 84300, 87900, 91500, 95100, 98700, 102300, 
                         105900, 109500, 113100, 116700, 120300, 123900, 127500, 131100, 134700, 138300, 141900, 145500, 
                         149100, 152700, 156300, 159900, 163500, 167100, 170700, 174300, 177900, 181500, 185100, 188700, 
                         192300, 196150, 200000]
    
    for z in target_Floor_z:
        Floor_Masses[z] = [0, 0, 0, 0, 0, 0]

    Center_coords = (sum(coord[0] for coord in node_coords.values()) / len(node_coords),
                     sum(coord[1] for coord in node_coords.values()) / len(node_coords))
    for point_name, mass in list(node_masses.items())[2000:]:
        # 获取节点坐标
        x, y, z = node_coords[point_name] # 保留0位小数
        z = round(z, 0)
        if z in Floor_Masses:
            # 将元组转化为列表，以便修改
            mass = list(mass)
            mass[5] = mass[0]*((Center_coords[0]-x)/1000)**2 + mass[1]*((Center_coords[1]-y)/1000)**2
            Floor_Masses[z] = [x + y for x, y in zip(Floor_Masses[z], mass)]
    # ======================打印每层的质量信息============================
    print(f"\n模型中共有 {len(Floor_Masses)} 层，每层的质量信息如下:")
    for z, mass in Floor_Masses.items():
        print(f"层 {z}: 质量 = {mass}")
    # 直接保存每层质量信息到CSV文件
    df_floor_masses = pd.DataFrame.from_dict(Floor_Masses, orient='index', columns=["MASS_X", "MASS_Y", "MASS_Z", "MASS_R1", "MASS_R2", "MASS_R3"])
    df_floor_masses.index.name = 'FLOOR_LEVEL'

    output_dir = os.path.join("data", "output", "parameter", building_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df_floor_masses.to_csv(os.path.join(output_dir, "Floor_Masses.csv"))

    modal_periods, modal_freqs, df_shapes = SapModel.get_modal_results(number_modes)
    # df_shapes.to_csv("modal_shapes.csv", index=False)
    print(f"df_shapes的尺寸为: {df_shapes.shape}\n前5行振型数据:")
    print(df_shapes.head())

    # 获取指定节点的模态位移并构建矩阵
    all_nodes_modal_matrix = []  # 存储所有节点的模态数据
    node_names_list = []  # 存储节点名称，用于后续标识
    
    for point_name in list(diaphragm_constraint_coords.keys()):
        # 从df_shapes中获取该节点的模态位移
        node_shapes = df_shapes[df_shapes["Obj"] == point_name]
        if node_shapes.empty:
            print(f"警告: 节点 {point_name} 没有模态数据，跳过")
            continue
            
        # 确保按模态编号排序
        node_shapes = node_shapes.sort_values('ModeNum')
        # 构建该节点的模态矩阵：每列为一个模态的[U1, U2, R3]
        node_modal_matrix = []
        max_modes = len(node_shapes)
        for mode_idx in range(max_modes):
            row = node_shapes.iloc[mode_idx]
            # 提取[U1, U2, R3]作为一列
            mode_column = [row['U1'], row['U2'], row['R3']]
            node_modal_matrix.append(mode_column)
        
        # 转置矩阵，使每列对应一个模态
        node_modal_matrix = np.array(node_modal_matrix).T  # 转置后：3行×15列
        # 将该节点的数据添加到总矩阵中
        if len(all_nodes_modal_matrix) == 0:
            all_nodes_modal_matrix = node_modal_matrix
        else:
            all_nodes_modal_matrix = np.vstack([all_nodes_modal_matrix, node_modal_matrix])
        
        node_names_list.append(point_name)
        # print(f"节点 {point_name}: 模态矩阵形状 = {node_modal_matrix.shape}")
    
    print(f"\n所有节点的模态矩阵构建完成:")
    print(f"总矩阵形状: {all_nodes_modal_matrix.shape}")
    # print(f"包含 {len(node_names_list)} 个节点")
    # print(f"每个节点有 3 行数据 (U1, U2, R3)")
    # print(f"共有 {all_nodes_modal_matrix.shape[1]} 列模态数据")
    
    # 创建DataFrame便于查看和保存
    # 创建列名
    mode_columns = [f"Mode_{i+1}" for i in range(all_nodes_modal_matrix.shape[1])]
    
    # # 创建行索引
    row_indices = []
    for node_name in node_names_list:
        row_indices.extend([f"{node_name}_U1", f"{node_name}_U2", f"{node_name}_R3"])
    
    # 创建DataFrame
    modal_matrix_df = pd.DataFrame(all_nodes_modal_matrix, 
                                   index=row_indices, 
                                   columns=mode_columns)
    
    print(f"\n模态矩阵前5行5列:")
    print(modal_matrix_df.iloc[:5, :5])
    
    # 保存到CSV文件
    modal_matrix_df.to_csv(os.path.join(output_dir, "modal_matrix.csv"), index=False)
    print(f"\n模态矩阵已保存到: {os.path.join(output_dir, 'modal_matrix.csv')}")

if __name__ == "__main__":
    main()

