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
# from comtypes import gen  # 用于SAP2000 API的类型定义
# # 确保comtypes可以找到SAP2000的类型库
# if sys.platform.startswith('win'):
#     # 添加SAP2000的类型库路径
#     sap2000_tlb_path = Path("C:/Program Files/Computers and Structures/SAP2000 23/SAP2000v1.tlb")
#     if sap2000_tlb_path.exists():
#         comtypes.client.GetModule(str(sap2000_tlb_path))
#     else:
#         print(f"错误: 找不到SAP2000类型库文件: {sap2000_tlb_path}")
#         sys.exit(1)
# import comtypes.gen.SAP2000v1  # 导入SAP2000的类型定义

def connect_to_sap2000():
    """
    连接到当前打开的SAP2000实例
    
    返回:
        成功连接时返回SAP2000模型对象，否则返回None
    """
    try:
        print("尝试连接到SAP2000...")
        
        # 创建SAP2000 API帮助对象
        helper = comtypes.client.CreateObject('SAP2000v1.Helper')
        helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
        
        # 获取当前打开的SAP2000实例
        mySapObject = helper.GetObject("CSI.SAP2000.API.SapObject")
        if mySapObject is None:
            print("找不到打开的SAP2000实例，尝试启动新实例...")
            
            # 启动新的SAP2000实例
            mySapObject = helper.CreateObject("CSI.SAP2000.API.SapObject")
            mySapObject.ApplicationStart()
            
        # 获取激活的模型
        model = mySapObject.SapModel
        
        # 检查是否已打开模型
        file_path = model.GetModelFilename()
        if not file_path:
            print("警告: SAP2000中未打开模型。请先打开一个模型文件。")
        else:    
            print(f"成功连接到SAP2000！当前模型文件: {file_path}")
        
        return model
    
    except Exception as e:
        print(f"连接到SAP2000时出错: {e}")
        return None

def get_model_info(model):
    """
    获取并打印SAP2000模型的详细信息
    
    参数:
        model: SAP2000模型对象
    """
    try:
        print("\n" + "=" * 40)
        print("SAP2000模型详细信息:")
        print("=" * 40)
        
        # 获取所有节点信息
        _, point_names, _, _, _, _, _ = model.PointObj.GetAllPoints()
        print(f"\n节点总数: {len(point_names)}")
        print(f"前10个节点: {point_names[:10] if len(point_names) > 10 else point_names}")
        
        # 获取所有框架元素信息
        _, frame_names, _, _, _ = model.FrameObj.GetAllFrames()
        print(f"\n框架元素总数: {len(frame_names)}")
        print(f"前10个框架元素: {frame_names[:10] if len(frame_names) > 10 else frame_names}")
        
        # 获取所有载荷模式
        _, load_patterns = model.LoadPatterns.GetNameList()
        print(f"\n载荷模式总数: {len(load_patterns)}")
        print(f"载荷模式: {load_patterns}")
        
        # 获取所有载荷组合
        _, load_combos = model.RespCombo.GetNameList()
        print(f"\n载荷组合总数: {len(load_combos)}")
        print(f"载荷组合: {load_combos}")
        
        # 获取所有隔板信息
        _, diaphragm_names = model.AreaObj.GetNameListDiaphragm()
        print(f"\n隔板总数: {len(diaphragm_names)}")
        print(f"隔板: {diaphragm_names}")
        
        print("\n" + "=" * 40)
        
        return True
    except Exception as e:
        print(f"获取模型信息时出错: {e}")
        return False
    
def get_timestamp():
    """
    获取当前时间戳，用于文件命名
    
    返回:
        格式化的时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def add_diaphragms(model, target_elevations=None, tolerance=0.01):
    """
    在SAP2000模型中创建刚性隔板
    
    参数:
        model: SAP2000模型对象
        target_elevations: 指定的楼层标高列表，如果为None则使用所有标高（可选）
        tolerance: 楼层标高容差（可选）

    返回:
        成功创建的隔板名称列表，如果失败则返回空列表
    """
    try:        
        # 获取所有节点的Z坐标（即楼层标高）
        node_info = []  # 用于保存所有节点名称和坐标
        node_z_coords = defaultdict(list)
        constraint_names = []  # 用于存储创建的约束名称

        # 获取模型中的楼层信息
        number_of_points, point_names, ret = model.PointObj.GetNameList()
       
        for point_name in point_names:
            [x, y, z, ret] = model.PointObj.GetCoordCartesian(point_name)
            node_info.append({"name": point_name, "x": x, "y": y, "z": z})
        
        # 如果指定了目标标高，则筛选节点
        if target_elevations is not None:
            print(f"按照指定标高筛选节点：{target_elevations}")
            filtered_nodes = []
            for node in node_info:
                # 检查节点是否在任意目标标高附近
                for elevation in target_elevations:
                    if abs(node["z"] -elevation) <= tolerance:
                        filtered_nodes.append(node)
                        # 将节点按照最接近的标高值进行分组
                        closest_z = min(target_elevations, key=lambda e: abs(e - node["z"]))
                        node_z_coords[closest_z].append(node["name"])
                        break
            print(f"筛选后剩余{len(filtered_nodes)}个节点")
        else:
            print("未指定目标标高，使用所有节点")
            for node in node_info:
                node_z_coords[round(node["z"], 3)].append(node["name"])  # 使用Z坐标分组，保留3位小数

        # 保存节点信息到CVS文件
        df = pd.DataFrame(node_info)
        csv_path = os.path.join(os.getcwd(), "node_coordinates.csv")
        df.to_csv(csv_path, index=False)
        print(f"节点坐标已保存到: {csv_path}")            

        # 为每一组节点设置刚性隔板约束
        print(f"将为以下Z坐标创建刚性隔板约束: {list(node_z_coords.keys())}")

        for z_value, nodes in node_z_coords.items():
            constraint_name = f"Diaphragm_Z_{z_value}"  # 根据Z值生成约束名称
            
            # 首先定义刚性隔板的名称
            ret = model.ConstraintDef.SetDiaphragm(constraint_name, 3, "Global")  # 设置刚性隔板约束

            # print(f"创建刚性隔板约束: {constraint_name}，包含节点: {nodes}")
            constraint_names.append(constraint_name)  # 保存约束名称

            # 设置刚性隔板约束
            # model.ConstraintDef.SetDiaphragm(constraint_name, nodes, "Global")  # 设置刚性隔板约束
            for i in nodes:
                model.PointObj.SetConstraint(i,constraint_name)

        print("刚性隔板约束设置完成！")
        return constraint_names, node_z_coords  # 返回创建的约束名称列表

    except Exception as e:
        print(f"创建刚性隔板时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def add_wind_time_history_load(model, diaphragm_constraints, node_z_coords, wind_time_history_file=None,num_rows=None):
    """
    在每个刚性隔板的中心点添加风荷载时程曲线
    
    参数:
        model: SAP2000模型对象
        diaphragm_constraints: 刚性隔板约束名称列表
        node_z_coords: 节点Z坐标字典
        wind_time_history_file: 风荷载时程函数文件路径（如果为None，则使用默认函数）

    返回:
        成功添加荷载的数量
    """
    print("\n开始添加风荷载时程曲线...")
    
    # 步骤1：为每个刚性隔板找到中心点
    diaphragm_centers = {}
    success_count = 0
    
    for constraint_name in diaphragm_constraints:
        # 获取约束中的所有节点
        constraint_points = []
        point_names = node_z_coords.get(round(float(constraint_name.split('_')[-1]), 3), [])
        
        # 遍历所有节点，找出使用该约束的节点
        # _, point_names, _ = model.PointObj.GetNameList()
        for point_name in point_names:
            # 获取节点坐标
            [x, y, z, _] = model.PointObj.GetCoordCartesian(point_name)
            constraint_points.append({"name": point_name, "x": x, "y": y, "z": z})
        if not constraint_points:
            print(f"警告：隔板 {constraint_name} 没有关联节点，跳过")
            continue
        print(f"隔板 {constraint_name} 包含 {len(constraint_points)} 个节点")

        # 计算中心点坐标
        avg_x = sum(point["x"] for point in constraint_points) / len(constraint_points)
        avg_y = sum(point["y"] for point in constraint_points) / len(constraint_points)
        avg_z = sum(point["z"] for point in constraint_points) / len(constraint_points)
        
        # 步骤2：在中心点创建新节点（或使用最接近中心的现有节点）
        # 查找最接近中心的节点
        closest_point = min(constraint_points, 
                            key=lambda p: ((p["x"]-avg_x)**2 + (p["y"]-avg_y)**2)**0.5)
                            
        center_point_name = f"WIND_CENTER_{constraint_name}"
        
        # 检查是否需要创建新节点（如果中心点附近没有现有节点）
        if ((closest_point["x"]-avg_x)**2 + (closest_point["y"]-avg_y)**2)**0.5 > 1:  # 如果最近点距离中心超过1mm
            # 创建新节点
            ret = model.PointObj.AddCartesian(avg_x, avg_y, avg_z, center_point_name)
            if ret[-1] != 0:
                print(f"创建中心点失败：{constraint_name}，使用最近节点")
                center_point_name = closest_point["name"]
            else:
                # 将新节点添加到隔板约束中
                model.PointObj.SetConstraint(center_point_name, constraint_name)
                print(f"在隔板 {constraint_name} 中创建了中心点: {center_point_name}")
        else:
            # 使用最近的现有节点
            center_point_name = closest_point["name"]
            print(f"使用隔板 {constraint_name} 中的最近点作为中心: {center_point_name}")
            
        diaphragm_centers[constraint_name] = {
            "point_name": center_point_name, 
            "x": avg_x, 
            "y": avg_y, 
            "z": avg_z
        }

    # 读取CSV文件中的数据
    df = pd.read_csv(wind_time_history_file, header=None)
    fs = 8.3227

    # 生成时间序列
    MyTime  = [i/fs for i in range(0, len(df))]
    print(f"时程数据共有 {len(df)} 行，采样频率 {fs}Hz")
    print(f"时间序列前5个值: {MyTime[:5]}")
    
    # 限值缝隙行数以加快测试速度（实际分析请移除次限制）
    num_columns = df.shape[1]
    if num_rows is None or num_rows > df.shape[0]:
        num_rows = df.shape[0]
    MyTime = MyTime[:num_rows]

    # 删除可能存在的旧荷载模式和时程函数
    try:
        # 删除旧的荷载模式
        existing_patterns = []
        try:
            _, existing_patterns = model.LoadPatterns.GetNameList()
        except:
            pass
            
        for pattern in existing_patterns:
            if pattern.startswith("Wind_"):
                try:
                    model.LoadPatterns.Delete(pattern)
                    print(f"已删除荷载模式: {pattern}")
                except:
                    pass
        
        # 删除旧的时程工况
        try:
            model.LoadCases.Delete("WIND_TIME_HISTORY")
            print("已删除旧的时程工况")
        except:
            pass
            
        # 删除旧的时程函数
        try:
            _, func_names = model.Func.GetNameList()
            for func in func_names:
                if func.startswith("WIND_"):
                    model.Func.Delete(func)
            print("已删除旧的时程函数")
        except:
            pass
    except Exception as e:
        print(f"清理旧数据时出错: {e}")

    # 创建统一的时程工况
    unified_case_name = "Wind_time_history"
    ret = model.LoadCases.DirHistLinear.SetCase(unified_case_name)
    if ret != 0:
        print(f"创建时程工况失败: {unified_case_name}")
        return 0
    
    print(f"创建统一的时程工况: {unified_case_name}")

    # 存储所有荷载参数，用于后续批量添加到时程工况
    load_types = []      # 荷载类型
    load_patterns = []   # 荷载模式名称
    func_names = []      # 时程函数名称
    scales = []          # 比例系数
    tfactors = []        # 时间比例因子
    delays = []          # 时间延迟
    coord_systems = []   # 坐标系
    angles = []          # 角度

    # 步骤5：为每个隔板中心添加风荷载
    col_idx = 0
    # 创建响应组合
    ret = model.RespCombo.Delete("Combo2")
    ret = model.RespCombo.Add("Combo2", 0)
    for constraint_name, center_info in diaphragm_centers.items():
        point_name = center_info["point_name"]
        
        print(f"首先删除隔板 {constraint_name} 中心点 {point_name} 的风荷载")
        ret = model.PointObj.DeleteLoadForce(point_name, "Wind")

        col_data = df[col_idx].values.tolist() # 提取当前列的风荷载数据
        Wind_func_name = f"Wind_x_{col_idx + 1}" # 生成风荷载时程函数名称
        ret = model.Func.FuncTH.SetUser(Wind_func_name, len(MyTime), MyTime, col_data)  # 从文件创建风荷载时程函数
        
        LoadPatternName = Wind_func_name # 创建荷载模式
        ret = model.LoadPatterns.Add(LoadPatternName, 6, 0, True)

        # 在节点上施加荷载
        x_force = 1000  # X方向风荷载
        ret = model.PointObj.SetLoadForce(point_name, LoadPatternName, [x_force, 0, 0, 0, 0, 0], True, "Global", 0)
        print(f"在隔板 {constraint_name} 中心点 {point_name} 添加风荷载X方向风荷载时程函数: {Wind_func_name}")

        # 将荷载参数添加到列表
        # X方向
        load_types.append("Load")
        load_patterns.append(LoadPatternName)
        func_names.append(Wind_func_name)
        scales.append(1.0)
        tfactors.append(1.0)
        delays.append(0.0)
        coord_systems.append("Global")
        angles.append(0.0)

        col_idx += 1
        # 提取当前列的风荷载数据
        col_data = df[col_idx].values.tolist()
        Wind_func_name = f"Wind_y_{col_idx + 1}"
        ret = model.Func.FuncTH.SetUser(Wind_func_name, len(MyTime), MyTime, col_data)  # 从文件创建风荷载时程函数
        # 创建荷载模式
        LoadPatternName = Wind_func_name
        ret = model.LoadPatterns.Add(LoadPatternName, 6, 0, True)
        # 在节点上施加荷载
        y_force = 1000  # Y方向风荷载
        ret = model.PointObj.SetLoadForce(point_name, LoadPatternName, [0, y_force, 0, 0, 0, 0], True, "Global", 0)
        print(f"在隔板 {constraint_name} 中心点 {point_name} 添加风荷载Y方向风荷载时程函数: {Wind_func_name}")

        # 将荷载参数添加到列表
        # Y方向
        load_types.append("Load")
        load_patterns.append(LoadPatternName)
        func_names.append(Wind_func_name)
        scales.append(1.0)
        tfactors.append(1.0)
        delays.append(0.0)
        coord_systems.append("Global")
        angles.append(0.0)


        col_idx += 1
        # 提取当前列的风荷载数据
        col_data = df[col_idx].values.tolist()
        Wind_func_name = f"Wind_z_{col_idx + 1}"
        ret = model.Func.FuncTH.SetUser(Wind_func_name, len(MyTime), MyTime, col_data)  # 从文件创建风荷载时程函数
        # 创建荷载模式
        LoadPatternName = Wind_func_name
        ret = model.LoadPatterns.Add(LoadPatternName, 6, 0, True)
        # 在节点上施加荷载
        z_force = 1000  # Z方向风荷载
        ret = model.PointObj.SetLoadForce(point_name, LoadPatternName, [0, 0, 0, 0, 0, z_force], True, "Global", 0)
        print(f"在隔板 {constraint_name} 中心点 {point_name} 添加风荷载Z方向风荷载时程函数: {Wind_func_name}")
        
        # 将荷载参数添加到列表
        # Z方向
        load_types.append("Load")
        load_patterns.append(LoadPatternName)
        func_names.append(Wind_func_name)
        scales.append(1.0)
        tfactors.append(1.0)
        delays.append(0.0)
        coord_systems.append("Global")
        angles.append(0.0)

        # 将工况添加到响应组合
        ret = model.RespCombo.SetCaseList("Combo2", 0, LoadPatternName, 1)
        col_idx += 1

    # 步骤7：将所有荷载关联到统一时程工况
    num_loads = len(load_patterns)
    print(f"将 {num_loads} 个荷载关联到统一时程工况 {unified_case_name}")

    # 提交所有荷载到统一时程工况
    ret = model.LoadCases.DirHistLinear.SetLoads(
        unified_case_name,
        num_loads,
        load_types,
        load_patterns,
        func_names,
        scales,
        tfactors,
        delays,
        coord_systems,
        angles
    )
    ret = model.LoadCases.DirHistLinear.SetTimeIntegration(unified_case_name, 1, 0, 0.5, 0.25, 0, 0)
    ret = model.LoadCases.DirHistLinear.SetTimeStep(unified_case_name, num_rows, 1/fs)

    if ret != 0:
        print(f"将荷载关联到时程工况失败, 错误码: {ret}")
    else:
        print(f"成功将所有荷载关联到时程工况 {unified_case_name}")

    # 设置运行工况
    ret = model.Analyze.SetRunCaseFlag(unified_case_name, True)
    
    print(f"风荷载时程曲线添加完成，共添加了 {col_idx} 个荷载")

    return col_idx, diaphragm_centers
        
def get_node_response_history(model, node_name, load_case="Wind_time_history", output_file=None, timestamp=None):
    """
    获取指定节点在指定荷载工况下的位移响应时程
    
    参数:
        model: SAP2000模型对象
        node_name: 要获取位移的节点名称
        load_case: 荷载工况名称，默认为"Wind_time_history"
        output_file: 输出CSV文件路径，如果指定则将结果保存到CSV文件
        timestamp: 时间戳，用于文件命名（可选）

    返回:
        成功时返回元组(时间列表, [X位移列表, Y位移列表, Z位移列表, X旋转列表, Y旋转列表, Z旋转列表])
        失败时返回(None, None)
    """
    try:
        print(f"\n获取节点 {node_name} 在 {load_case} 工况下的位移响应时程...")
        
        # 检查节点是否存在
        ret = model.PointObj.GetNameList()
        if ret[0] == 0 and node_name not in ret[1]:
            print(f"错误: 节点 {node_name} 不存在")
            return None, None
            
        # 检查荷载工况是否存在
        ret = model.LoadCases.GetNameList()
        if ret[0] == 0 and load_case not in ret[1]:
            print(f"错误: 荷载工况 {load_case} 不存在")
            return None, None
            
        # 获取时间步长信息
        ret = model.LoadCases.DirHistLinear.GetTimeStep(load_case)
        if ret[-1] != 0:
            print(f"获取时间步长失败，错误码: {ret[0]}")
            return None, None
            
        num_steps = ret[0]
        time_step = ret[1]
        print(f"时程分析包含 {num_steps} 个时间步，步长为 {time_step} 秒")

        ret = model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        ret = model.Results.Setup.SetCaseSelectedForOutput(load_case)

        # 获取位移结果（注意正确的方法名）
        GroupElm = 0
        NumberResults = 0
        Obj = []
        Elm = []
        LoadCase = [load_case]
        StepType = ["Time"]
        StepNum = []
        U1, U2, U3, R1, R2, R3 = [], [], [], [], [], []
        
        [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = \
        model.Results.JointDispl(
            node_name, 
            GroupElm, 
            NumberResults, 
            Obj, 
            Elm, 
            LoadCase, 
            StepType, 
            StepNum,
            U1, U2, U3, R1, R2, R3 
        )
        
        ux_list = U1  # X方向位移
        uy_list = U2  # Y方向位移
        uz_list = U3  # Z方向位移
        rx_list = R1  # X方向旋转
        ry_list = R2  # Y方向旋转
        rz_list = R3  # Z方向旋转

        # 创建位移DataFrame
        df_disp = pd.DataFrame({
            # "time":time_points,
            "UX": U1,
            "UY": U2,
            "UZ": U3,
            "RX": R1,
            "RY": R2,
            "RZ": R3
        })

        # 输出简单统计信息
        print("\n位移响应统计:")
        print(f"X方向最大位移: {max(ux_list, key=abs):.6f} mm")
        print(f"Y方向最大位移: {max(uy_list, key=abs):.6f} mm")
        print(f"Z方向最大位移: {max(uz_list, key=abs):.6f} mm")
        
        # 汇总位移结果
        displacement_results = [ux_list, uy_list, uz_list, rx_list, ry_list, rz_list]
        time_points = [i * time_step for i in range(num_steps+1)]  # 生成时间点列表
        print(f"displacement_results的尺寸为: {len(displacement_results[0])}")
        print(f"time_points的尺寸为: {len(time_points)}")

        # 获取节点加速度
        GroupElm = 0
        NumberResults = 0
        Obj = []
        Elm = []
        LoadCase = [load_case]
        StepType = ["Time"]
        StepNum = []
        U1, U2, U3, R1, R2, R3 = [], [], [], [], [], []
        
        [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = \
        model.Results.JointAcc(
            node_name, 
            GroupElm, 
            NumberResults, 
            Obj, 
            Elm, 
            LoadCase, 
            StepType, 
            StepNum,
            U1, U2, U3, R1, R2, R3 
        )
        
        ux_list = U1  # X方向加速度
        uy_list = U2  # Y方向加速度
        uz_list = U3  # Z方向加速度
        rx_list = R1  # X方向加速度
        ry_list = R2  # Y方向加速度
        rz_list = R3  # Z方向加速度

        # 创建加速度DataFrame
        df_acc = pd.DataFrame({
            # "time":time_points,
            "UX": ux_list,
            "UY": uy_list,
            "UZ": uz_list,
            "RX": rx_list,
            "RY": ry_list,
            "RZ": rz_list
        })
        
        # 输出简单统计信息
        print("\n加速度响应统计:")
        print(f"X方向最大加速度: {max(ux_list, key=abs):.6f} mm")
        print(f"Y方向最大加速度: {max(uy_list, key=abs):.6f} mm")
        print(f"Z方向最大加速度: {max(uz_list, key=abs):.6f} mm")

        # 汇总加速度结果
        acceleration_results = [ux_list, uy_list, uz_list, rx_list, ry_list, rz_list]
        print(f"acceleration_results的尺寸为: {len(acceleration_results[0])}")

        # 如果指定了输出文件，保存结果到CSV
        if output_file:
            # 创建带时间戳的文件名
            timestamp = get_timestamp()
            output_disp_file_with_timestamp = create_unique_filename(output_file, "disp", timestamp)
            output_acce_file_with_timestamp = create_unique_filename(output_file, "acce", timestamp)

            # 确保输出目录存在
            output_dir = os.path.dirname(output_disp_file_with_timestamp) # 获取输出文件的目录
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存到CSV文件
            df_disp.to_csv(output_disp_file_with_timestamp, index=False)
            print(f"位移响应时程已保存至: {output_disp_file_with_timestamp}")

            df_acc.to_csv(output_acce_file_with_timestamp, index=False)
            print(f"加速度响应时程已保存至: {output_acce_file_with_timestamp}")

        return time_points, displacement_results, acceleration_results
        
    except Exception as e:
        print(f"获取节点位移响应时程时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
        # extension = ext_part
    else:
        name_part = filename

    # 创建带时间戳的文件名：文件名_时间戳
    # timestamped_filename = f"{timestamp}_{name_part}_{type}{ext_part}"
    timestamped_filename = f"{timestamp}_{name_part}_{type}{ext_part}"

    return os.path.join(directory, timestamped_filename)

def summarize_results(all_results):
    """
    根据 all_results 统计 wind_file 的种类，并输出表格
    
    参数:
        all_results: 包含所有节点响应结果的列表
    
    返回:
        pandas.DataFrame 表格，包含 wind_file 的统计信息
    """
    import pandas as pd

    # 提取 wind_file 和节点信息
    summary_data = []
    for result in all_results:
        summary_data.append({
            "wind_file": result["wind_file"],
            "node": result["node"],
            "time_steps": len(result["times"]),
            "max_displacement": max(max(map(abs, result["displacements"][0]), default=0),  # X方向最大位移
                                    max(map(abs, result["displacements"][1]), default=0),  # Y方向最大位移
                                    max(map(abs, result["displacements"][2]), default=0)), # Z方向最大位移
            "max_acceleration": max(max(map(abs, result["accelerations"][0]), default=0),  # X方向最大加速度
                                    max(map(abs, result["accelerations"][1]), default=0),  # Y方向最大加速度
                                    max(map(abs, result["accelerations"][2]), default=0))  # Z方向最大加速度
        })

    # 转换为 DataFrame
    df_summary = pd.DataFrame(summary_data)

    # 按 wind_file 分组统计
    grouped_summary = df_summary.groupby("wind_file").agg({
        "node": "count",  # 节点数量
        "time_steps": "sum",  # 总时间步数
        "max_displacement": "max",  # 最大位移
        "max_acceleration": "max"  # 最大加速度
    }).reset_index()

    # 输出表格
    print("\n统计结果表格:")
    print(grouped_summary)

    return grouped_summary

def main():
    # 记录程序开始时间
    start_time = time.time() # 记录开始时间
    start_datetime = datetime.now() # 获取当前时间
  
    print("=" * 80)
    print("SAP2000模型连接程序")
    print(f"程序开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 连接到当前打开的SAP2000实例
    model = connect_to_sap2000()
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
    target_elevations = [6000, 10500, 15000, 19500, 23100, 26700, 30300, 33900, 37500, 41100, 44700, 48300, 51900, 55500, 
                         59100, 62700, 66300, 69900, 73500, 77100, 80700, 84300, 87900, 91500, 95100, 98700, 102300, 
                         105900, 109500, 113100, 116700, 120300, 123900, 127500, 131100, 134700, 138300, 141900, 145500, 
                         149100, 152700, 156300, 159900, 163500, 167100, 170700, 174300, 177900, 181500, 185100, 188700, 
                         192300, 196150, 200000]

    diaphragm_constraints, node_z_coords = add_diaphragms(model, target_elevations=target_elevations, tolerance=10)
    if diaphragm_constraints:
        print(f"成功创建刚性隔板: {diaphragm_constraints}")
    else:
        print("创建刚性隔板失败")

    # 添加风荷载时程曲线，使用自定义风荷载时程文件
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本目录
    wind_file_path = os.path.join(script_dir, "WindloadTimes", "Model2_10yr_000.csv")
    wind_load_count, diaphragm_centers = add_wind_time_history_load(model, diaphragm_constraints, node_z_coords, wind_time_history_file=wind_file_path)
    if wind_load_count > 0:
        print(f"成功添加 {wind_load_count} 个风荷载时程曲线")
    else:
        print("添加风荷载时程曲线失败")
        
        # [4] 运行分析
        print("开启多线程求解器...")
        ret = model.Analyze.SetSolverOption_1(2,0,True)
        print("正在运行分析...")
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

        target_nodes = [node_top_center_name, "54000062", "54000070", "54000071", "54000079"]  # 示例节点名称列表

        # 在输出文件时包含 wind_file_path 中的文件名部分
        wind_file_name = os.path.basename(wind_file_path)
        wind_file_base_name = os.path.splitext(wind_file_name)[0]

        results_dir = os.path.join(script_dir, "output",f"{wind_file_base_name}") # 确保结果目录存在
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        
        for target_node in target_nodes:
            print(f"\n获取节点 {target_node} 的位移和加速度响应...")

            output_path = os.path.join(results_dir, f"{target_node}.csv")

            # 获取位移响应时程
            times, displacements, accelerations = get_node_response_history(
                model,
                target_node,
                load_case="Wind_time_history", 
                output_file=output_path
            )
            if times and displacements and accelerations:
                print(f"成功获取顶层角点 {target_node} 的 {len(times)} 个时间步的位移数据")
            else:
                print("获取位移响应失败")

            # 可以在此处将本次计算得到的加速度和位移结果保存到一个变量中，方便后续可视化展示
            all_results.append({
                "wind_file": wind_file_name,
                "node": target_node,
                "times": times,
                "displacements": displacements,
                "accelerations": accelerations
            })
    
    # [6] 结果统计分析
    summary_table = summarize_results(all_results)


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
