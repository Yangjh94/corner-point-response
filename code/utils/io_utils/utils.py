"""
工具函数模块
包含各种辅助函数
"""

import os
from datetime import datetime
import numpy as np

def get_timestamp():
    """
    获取当前时间戳，用于文件命名
    
    返回:
        格式化的时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_unique_filename(base_path, type, timestamp=None):
    """
    创建带时间戳的唯一文件名
    
    参数:
        base_path: 基础文件路径（可能包含扩展名）
        timestamp: 时间戳，如果为None则使用当前时间
    
    返回:
        带时间戳的完整文件路径
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    # 分离目录、文件名和扩展名
    directory = os.path.dirname(base_path) # 获取目录部分，如果没有目录则为当前目录
    filename = os.path.basename(base_path) # 获取文件名部分，不包含目录
    
    # 如果base_path包含扩展名，分离出来
    if '.' in filename:
        name_part, ext_part = os.path.splitext(filename)
        timestamped_filename = f"{timestamp}_{name_part}_{type}{ext_part}"
    else:
        name_part = filename
        timestamped_filename = f"{timestamp}_{name_part}_{type}.csv"

    return os.path.join(directory, timestamped_filename)

def g_D(response_data, dt, tao=600, gama=0.5772):
    """求解Davenport峰值因子"""
    response_data = np.array(response_data)
    std_q = np.std(response_data)
    
    # 计算一阶导数
    response_dot = np.gradient(response_data, dt)
    std_q_dian = np.std(response_dot)
    
    # 计算零穿越率和峰值因子
    v0i = std_q_dian / (2 * np.pi * std_q)
    k = np.sqrt(2 * np.log(v0i * tao))
    gi = k + gama / k
    
    return gi