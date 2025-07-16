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

import numpy as np

def dA(data, dt):
    """
    计算数据的一阶导数（使用数值微分）
    
    参数:
        data: 输入数据数组
        dt: 时间步长
    
    返回:
        一阶导数数组
    """
    return np.gradient(data, dt)

def g_CartandLong(data, std_val, std_dot, T, fs):
    """
    Cart and Long峰值因子计算方法
    
    参数:
        data: 响应时程数据
        std_val: 响应标准差
        std_dot: 响应一阶导标准差
        T: 持续时间
        fs: 采样频率
    
    返回:
        峰值因子
    """
    # 简化实现，可以根据需要替换为更精确的Cart-Long方法
    v0 = std_dot / (2 * np.pi * std_val)
    k = np.sqrt(2 * np.log(v0 * T))
    return k + 0.5772 / k

def CDC(Xt, Yt, tDlt):
    """
    CPF方法二维响应极值计算
    
    参数:
        Xt: X方向响应时程，零均值
        Yt: Y方向响应时程，零均值
        tDlt: 采样时间间隔
    
    返回:
        R_CDC: CPF方法二维响应极值
    """
    # 转换为numpy数组
    Xt = np.array(Xt)
    Yt = np.array(Yt)
    
    # 去除均值，确保零均值
    Xt = Xt - np.mean(Xt)
    Yt = Yt - np.mean(Yt)
    
    # 计算X方向统计量
    std_X = np.std(Xt)
    std_X_dian = np.std(dA(Xt, tDlt))
    gf_X = g_CartandLong(Xt, std_X, std_X_dian, 600, 1/tDlt)
    
    # 计算Y方向统计量
    std_Y = np.std(Yt)
    std_Y_dian = np.std(dA(Yt, tDlt))
    gf_Y = g_CartandLong(Yt, std_Y, std_Y_dian, 600, 1/tDlt)
    
    # 计算相关系数
    correlation_matrix = np.corrcoef(Xt, Yt)
    rouxy = correlation_matrix[0, 1]
    
    # 计算极值响应
    # R1公式
    term1 = (gf_X * std_X)**2 + (gf_Y * std_Y)**2
    term2_inner = ((gf_X * std_X)**2 - (gf_Y * std_Y)**2)**2 / 4
    term2_corr = (rouxy * gf_X * std_X * gf_Y * std_Y)**2
    term2 = np.sqrt(term2_inner + term2_corr)
    R1 = np.sqrt(term1/2 + term2)
    
    # R2公式
    R2 = 0.8 * np.sqrt((gf_X * std_X)**2 + (gf_Y * std_Y)**2)
    
    # 取最大值
    R_CDC = max(R1, R2)
    
    return R_CDC

    