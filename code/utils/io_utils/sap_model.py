"""
SAP2000模型操作类
负责所有与SAP2000模型相关的操作
"""

import os
import pandas as pd
import comtypes.client
from collections import defaultdict
from .utils import get_timestamp, create_unique_filename

class SAP2000Model:
    """SAP2000模型管理器 - 负责所有SAP2000操作"""
    
    def __init__(self, building_name):
        self.model = None
        self.is_connected = False
        self.building_name = building_name

    def connect(self):
        """连接到SAP2000 - 您的原始connect_to_sap2000代码"""
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
            self.model = mySapObject.SapModel
            
            # 解锁模型
            if self.model.GetModelIsLocked():
                self.model.SetModelIsLocked(False)
                print("模型已解锁")
            
            # 检查是否已打开模型
            file_path = self.model.GetModelFilename()
            if not file_path:
                print("警告: SAP2000中未打开模型。请先打开一个模型文件。")
            else:    
                print(f"成功连接到SAP2000！当前模型文件: {file_path}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"连接到SAP2000时出错: {e}")
            return False
    
    def get_model_info(self):
        """获取模型信息 - 您的原始get_model_info代码"""
        try:
            print("\n" + "=" * 40)
            print("SAP2000模型详细信息:")
            print("=" * 40)
            
            # 获取所有节点信息
            _, point_names, _, _, _, _, _ = self.model.PointObj.GetAllPoints()
            print(f"\n节点总数: {len(point_names)}")
            print(f"前10个节点: {point_names[:10] if len(point_names) > 10 else point_names}")
            
            # 获取所有框架元素信息
            _, frame_names, _, _, _ = self.model.FrameObj.GetAllFrames()
            print(f"\n框架元素总数: {len(frame_names)}")
            print(f"前10个框架元素: {frame_names[:10] if len(frame_names) > 10 else frame_names}")
            
            # 获取所有载荷模式
            _, load_patterns = self.model.LoadPatterns.GetNameList()
            print(f"\n载荷模式总数: {len(load_patterns)}")
            print(f"载荷模式: {load_patterns}")
            
            # 获取所有载荷组合
            _, load_combos = self.model.RespCombo.GetNameList()
            print(f"\n载荷组合总数: {len(load_combos)}")
            print(f"载荷组合: {load_combos}")
            
            # 获取所有隔板信息
            _, diaphragm_names = self.model.AreaObj.GetNameListDiaphragm()
            print(f"\n隔板总数: {len(diaphragm_names)}")
            print(f"隔板: {diaphragm_names}")
            
            print("\n" + "=" * 40)
            
            return True
        except Exception as e:
            print(f"获取模型信息时出错: {e}")
            return False
    
    def add_diaphragms(self, target_elevations=None, tolerance=0.01):
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
            number_of_points, point_names, ret = self.model.PointObj.GetNameList()
        
            for point_name in point_names:
                [x, y, z, ret] = self.model.PointObj.GetCoordCartesian(point_name)
                node_info.append({"name": point_name, "x": x, "y": y, "z": z})
            
            # 如果指定了目标标高，则筛选节点
            if target_elevations is not None:
                print(f"按照指定标高筛选节点：{target_elevations[0:5]} 等 {len(target_elevations)} 个标高")
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
            csv_path = os.path.join(os.getcwd(), "data", "output", "parameter",self.building_name, "node_coordinates.csv")
            if not os.path.exists(os.path.dirname(csv_path)):
                os.makedirs(os.path.dirname(csv_path))
            df.to_csv(csv_path, index=False)
            print(f"节点坐标已保存到: {csv_path}") 
            # 为每一组节点设置刚性隔板约束
            print(f"将为以下Z坐标创建刚性隔板约束: {list(node_z_coords.keys())[0:5]} 等 {len(node_z_coords)} 个")

            for z_value, nodes in node_z_coords.items():
                constraint_name = f"Diaphragm_Z_{z_value}"  # 根据Z值生成约束名称
                
                # 首先定义刚性隔板的名称
                ret = self.model.ConstraintDef.SetDiaphragm(constraint_name, 3, "Global")  # 设置刚性隔板约束

                # print(f"创建刚性隔板约束: {constraint_name}，包含节点: {nodes}")
                constraint_names.append(constraint_name)  # 保存约束名称

                # 设置刚性隔板约束
                # self.model.ConstraintDef.SetDiaphragm(constraint_name, nodes, "Global")  # 设置刚性隔板约束
                for i in nodes:
                    self.model.PointObj.SetConstraint(i,constraint_name)

            print("刚性隔板约束设置完成！")
            return constraint_names, node_z_coords  # 返回创建的约束名称列表

        except Exception as e:
            print(f"创建刚性隔板时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def add_wind_time_history_load(self, number_modes, diaphragm_constraints, node_z_coords, wind_time_history_file=None,num_rows=None,damp=None):
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
        print("开始添加风荷载时程曲线...")
        
        # 步骤1：为每个刚性隔板找到中心点
        diaphragm_centers = {}
        success_count = 0
        
        for constraint_name in diaphragm_constraints:
            # 获取约束中的所有节点
            constraint_points = []
            point_names = node_z_coords.get(round(float(constraint_name.split('_')[-1]), 3), [])
            
            # 遍历所有节点，找出使用该约束的节点
            for point_name in point_names:
                # 获取节点坐标
                [x, y, z, _] = self.model.PointObj.GetCoordCartesian(point_name)
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
            if ((closest_point["x"]-avg_x)**2 + (closest_point["y"]-avg_y)**2)**0.5 > 10:  # 如果最近点距离中心超过100mm
                # 创建新节点
                ret = self.model.PointObj.AddCartesian(avg_x, avg_y, avg_z, center_point_name)
                if ret[-1] != 0:
                    print(f"创建中心点失败：{constraint_name}，使用最近节点")
                    center_point_name = closest_point["name"]
                else:
                    # 将新节点添加到隔板约束中
                    self.model.PointObj.SetConstraint(center_point_name, constraint_name)
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
        
        # 限值缝隙行数以加快测试速度（实际分析请移除次限制）
        num_columns = df.shape[1]
        if num_rows is None or num_rows > df.shape[0]:
            num_rows = df.shape[0]
        MyTime = MyTime[:num_rows]

        # 设置模态分析工况
        print("创建模态分析工况...")
        # ret = self.model.LoadCases.Delete("MODAL")  # 先删除可能存在的旧模态工况
        ret = self.model.LoadCases.ModalEigen.SetCase("MODAL")
        ret = self.model.LoadCases.ModalEigen.SetNumberModes("MODAL", number_modes, 1)  # 设置计算前30阶模态
        ret = self.model.LoadCases.ModalEigen.SetParameters("MODAL", 0, 0, 1E-10, 1)

        unified_case_name = "Wind_time_history"
        ret = self.model.LoadCases.Delete(unified_case_name)  # 删除可能存在的旧工况
        ret = self.model.LoadCases.ModHistLinear.SetCase(unified_case_name)  # 创建模态时程分析工况
        if ret != 0:
            print(f"创建时程工况失败: {unified_case_name}")
        # 设置模态时程分析的参数
        ret = self.model.LoadCases.ModHistLinear.SetModalCase(unified_case_name, "MODAL")  # 指定模态分析工况
        # ret = self.model.LoadCases.ModHistLinear.SetNumberModes(unified_case_name, 30)  # 使用前30阶模态
        ret = self.model.LoadCases.ModHistLinear.SetTimeStep(unified_case_name, num_rows, 1/fs)
        ret = self.model.LoadCases.ModHistLinear.SetDampConstant(unified_case_name, damp)
        print(f"创建模态时程分析工况: {unified_case_name}")

        # 删除可能存在的旧荷载模式和时程函数
        existing_patterns = []
        ret = self.model.LoadPatterns.GetNameList()
        existing_patterns = ret[1]
            
        for pattern in existing_patterns:
            if pattern.startswith("Wind_"):
                ret = self.model.LoadCases.Delete(pattern)
                ret = self.model.LoadPatterns.Delete(pattern) # 无法删除荷载模式是什么原因
                if ret != 0:
                    print(f"删除荷载模式 {pattern} 失败，错误码: {ret}")
                else:
                    print(f"已删除荷载模式: {pattern}")

        # 删除旧的时程函数
        func_names = []
        ret = self.model.Func.GetNameList()
        func_names = ret[1]
        for func in func_names:
            if func.startswith("Wind_"):
                ret = self.model.Func.Delete(func)
                if ret != 0:
                    print(f"删除时程函数 {func} 失败，错误码: {ret}")
                else:
                    print(f"已删除时程函数: {func}")

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
        for constraint_name, center_info in diaphragm_centers.items():
            point_name = center_info["point_name"]
            
            # print(f"首先删除隔板 {constraint_name} 中心点 {point_name} 的风荷载")
            # ret = model.PointObj.DeleteLoadForce(point_name, "Wind")

            col_data = df[col_idx].values.tolist() # 提取当前列的风荷载数据
            Wind_func_name = f"Wind_x_{col_idx + 1}" # 生成风荷载时程函数名称
            ret = self.model.Func.FuncTH.SetUser(Wind_func_name, len(MyTime), MyTime, col_data)  # 从文件创建风荷载时程函数

            LoadPatternName = Wind_func_name # 创建荷载模式
            ret = self.model.LoadPatterns.Add(LoadPatternName, 6, 0, True)

            # 在节点上施加荷载
            x_force = 1000  # X方向风荷载
            ret = self.model.PointObj.SetLoadForce(point_name, LoadPatternName, [0, x_force, 0, 0, 0, 0], True, "Global", 0)
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
            ret = self.model.Func.FuncTH.SetUser(Wind_func_name, len(MyTime), MyTime, col_data)  # 从文件创建风荷载时程函数
            # 创建荷载模式
            LoadPatternName = Wind_func_name
            ret = self.model.LoadPatterns.Add(LoadPatternName, 6, 0, True)
            # 在节点上施加荷载
            y_force = 1000  # Y方向风荷载
            ret = self.model.PointObj.SetLoadForce(point_name, LoadPatternName, [-1*y_force, 0, 0, 0, 0, 0], True, "Global", 0)
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
            ret = self.model.Func.FuncTH.SetUser(Wind_func_name, len(MyTime), MyTime, col_data)  # 从文件创建风荷载时程函数
            # 创建荷载模式
            LoadPatternName = Wind_func_name
            ret = self.model.LoadPatterns.Add(LoadPatternName, 6, 0, True)
            # 在节点上施加荷载
            z_force = 1000  # Z方向风荷载
            ret = self.model.PointObj.SetLoadForce(point_name, LoadPatternName, [0, 0, 0, 0, 0, z_force], True, "Global", 0)
            print(f"在隔板 {constraint_name} 中心点 {point_name} 添加风荷载Z方向风荷载时程函数: {Wind_func_name}")
            
            # 将荷载参数添加到列表
            load_types.append("Load")
            load_patterns.append(LoadPatternName)
            func_names.append(Wind_func_name)
            scales.append(1.0)
            tfactors.append(1.0)
            delays.append(0.0)
            coord_systems.append("Global")
            angles.append(0.0)

            # 将工况添加到响应组合
            # ret = model.RespCombo.SetCaseList("Combo2", 0, LoadPatternName, 1)
            col_idx += 1

        # 步骤7：将所有荷载关联到统一时程工况
        num_loads = len(load_patterns)
        print(f"将 {num_loads} 个荷载关联到统一时程工况 {unified_case_name}")

        # 提交所有荷载到统一时程工况
        ret = self.model.LoadCases.ModHistLinear.SetLoads(
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
        if ret == 0:
            print(f"成功将 {num_loads} 个荷载关联到工况 {unified_case_name}")
        # 获取运行工况的状态
        existing_LoadCases = []
        ret = self.model.LoadCases.GetNameList()
        existing_LoadCases = ret[1]
        for case in existing_LoadCases:
            ret = self.model.Analyze.SetRunCaseFlag(case, False)  # 确保所有工况都未选中
        # 设置统一时程工况为运行状态
        ret = self.model.Analyze.SetRunCaseFlag("MODAL", True)
        ret = self.model.Analyze.SetRunCaseFlag(unified_case_name, True)
        print(f"风荷载时程曲线添加完成，共添加了 {col_idx} 个荷载")

        return col_idx, diaphragm_centers

    def get_node_response_history(self, building_name, node_name, Type, damp, load_case="Wind_time_history", load_history_file=None, output_file=None, timestamp=None):
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
        print(f"\n获取节点 {node_name} 在 {load_case} 工况下的位移响应时程...")
        
        # 检查节点是否存在
        ret = self.model.PointObj.GetNameList()
        if ret[0] == 0 and node_name not in ret[1]:
            print(f"错误: 节点 {node_name} 不存在")
            return None, None
            
        # 检查荷载工况是否存在
        ret = self.model.LoadCases.GetNameList()
        if ret[0] == 0 and load_case not in ret[1]:
            print(f"错误: 荷载工况 {load_case} 不存在")
            return None, None
            
        # 获取时间步长信息
        ret = self.model.LoadCases.ModHistLinear.GetTimeStep(load_case)
        if ret[-1] != 0:
            print(f"获取时间步长失败，错误码: {ret[0]}")
            return None, None
            
        num_steps = ret[0]
        time_step = ret[1]
        time_points = [i * time_step for i in range(num_steps+1)]  # 生成时间点列表
        print(f"时程分析包含 {num_steps} 个时间步，步长为 {time_step} 秒")

        ret = self.model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        ret = self.model.Results.Setup.SetCaseSelectedForOutput(load_case)
        ret = self.model.Results.Setup.SetOptionModalHist(2)  # 使用绝对值输出

        if Type == "displacement":
            # 获取位移结果（注意正确的方法名）
            GroupElm = 0
            NumberResults = 0
            Obj = []
            Elm = []
            LoadCase = [load_case]  # 空列表，让函数填充
            StepType = []  # 空列表，让函数填充
            StepNum = []   # 空列表，让函数填充
            U1, U2, U3, R1, R2, R3 = [], [], [], [], [], []

            # 正确的调用方式，接收所有返回值
            [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = self.model.Results.JointDispl(
                node_name, 
                GroupElm,  # 使用0表示按对象获取结果
                NumberResults, 
                Obj, 
                Elm, 
                LoadCase,
                StepType, 
                StepNum,
                U1, U2, U3, R1, R2, R3 
            )

            # 创建位移DataFrame
            df = pd.DataFrame({
                # "time":time_points,
                "UX": U1,
                "UY": U2,
                "UZ": U3,
                "RX": R1,
                "RY": R2,
                "RZ": R3
            })

            # 输出简单统计信息
            print(f"获取节点 {node_name} 的位移响应时程成功！")
            print(f"当前荷载工况为: {load_case}")
            print(f"当前NumberResults为: {NumberResults}")
            # print(f"当前Obj为: {Obj}")
            # print(f"当前Elm为: {Elm}")
            # print(f"当前ACase为: {ACase}")
            # print(f"当前StepType为: {StepType}")
            # print(f"当前StepNum为: {StepNum}")

            print("\n位移响应统计:")
            print(f"X方向最大位移: {max(U1, key=abs):.6f} mm")
            print(f"Y方向最大位移: {max(U2, key=abs):.6f} mm")
            print(f"Z方向最大位移: {max(U3, key=abs):.6f} mm")
            print(f"RX方向最大旋转: {max(R1, key=abs):.6f} rad")
            print(f"RY方向最大旋转: {max(R2, key=abs):.6f} rad")
            print(f"RZ方向最大旋转: {max(R3, key=abs):.6f} rad")
            
            # 汇总位移结果
            results = [U1, U2, U3, R1, R2, R3]
            print(f"displacement_results的尺寸为: {len(results[0])}")

            load_history_file = os.path.splitext(load_history_file)[0]
            output_file = os.path.join(output_file,"1-1",Type,load_history_file)

        elif Type == "acceleration":
            # 获取节点加速度
            GroupElm = 0
            NumberResults = 0
            Obj = []
            Elm = []
            LoadCase = [load_case]
            StepType = []
            StepNum = []
            U1, U2, U3, R1, R2, R3 = [], [], [], [], [], []

            [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = self.model.Results.JointAccAbs(
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

            # 创建加速度DataFrame
            df = pd.DataFrame({
                # "time":time_points,
                "UX": U1,
                "UY": U2,
                "UZ": U3,
                "RX": R1,
                "RY": R2,
                "RZ": R3
            })
            
            # 输出简单统计信息
            print("\n加速度响应统计:")
            print(f"X方向最大加速度: {max(U1, key=abs):.6f} mm")
            print(f"Y方向最大加速度: {max(U2, key=abs):.6f} mm")
            print(f"Z方向最大加速度: {max(U3, key=abs):.6f} mm")
            print(f"RX方向最大加速度: {max(R1, key=abs):.6f} rad")
            print(f"RY方向最大加速度: {max(R2, key=abs):.6f} rad")
            print(f"RZ方向最大加速度: {max(R3, key=abs):.6f} rad")

            # 汇总加速度结果
            results = [U1, U2, U3, R1, R2, R3]
            print(f"acceleration_results的尺寸为: {len(results[0])}")

            load_history_file = os.path.splitext(load_history_file)[0]
            output_file = os.path.join(output_file,building_name,Type,load_history_file, node_name)

        # 如果指定了输出文件，保存结果到CSV
        if output_file:
            # 创建带时间戳的文件名
            timestamp = get_timestamp()
            output_file_with_timestamp = create_unique_filename(output_file, Type, timestamp)

            # 确保输出目录存在
            output_dir = os.path.dirname(output_file_with_timestamp) # 获取输出文件的目录
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存到CSV文件
            df.to_csv(output_file_with_timestamp, index=False)
            print(f"响应时程已保存至: {output_file_with_timestamp}")

        return time_points, results
    
    def get_modal_results(self, num_modes=15, group_name="ALL"):
        """
        获取SAP2000模型的模态频率、周期和所有节点振型（模态位移）
        返回：
            modal_periods: 周期列表
            modal_freqs: 频率列表
            df_shapes: DataFrame，包含所有节点的振型数据
        """
        # 1. 获取模态周期和频率
        self.model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        self.model.Results.Setup.SetCaseSelectedForOutput("MODAL")

        NumberResults = 0
        LoadCase = []
        StepType = []
        StepNum = []
        Period = []
        Frequency = []
        CircFreq = []
        EigenValue = []

        ret = self.model.Results.ModalPeriod(NumberResults, LoadCase, StepType, StepNum, Period, Frequency, CircFreq, EigenValue)
        if ret[-1] != 0:
            print(f"获取模态分析结果失败，错误码: {ret}")
            return None, None, None
        NumberResults = ret[0]
        LoadCase = ret[1]
        StepType = ret[2]
        StepNum = ret[3]
        Period = ret[4]
        Frequency = ret[5]
        CircFreq = ret[6]
        EigenValue = ret[7]

        modal_periods = []
        modal_freqs = []
        # =============================打印信息================================
        # print(f"模态分析结果: 共 {len(Period)} 个模态")
        # for i in range(min(num_modes, len(Period))):
        #     modal_periods.append(Period[i])
        #     modal_freqs.append(Frequency[i])
        #     print(f"模态 {i+1}: 周期 = {Period[i]:.6f} 秒, 频率 = {Frequency[i]:.6f} Hz")

        # 2. 获取所有节点的振型（模态位移）
        # eItemTypeElm 枚举，GroupElm=2
        GroupElm = 2
        NumberResults2 = 0
        Obj = []
        Elm = []
        LoadCase2 = []
        StepType2 = []
        StepNum2 = []
        U1 = []
        U2 = []
        U3 = []
        R1 = []
        R2 = []
        R3 = []

        ret = self.model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        ret = self.model.Results.Setup.SetCaseSelectedForOutput("MODAL")

        ret2 = self.model.Results.ModeShape(group_name, GroupElm, NumberResults2, Obj, Elm, LoadCase2, StepType2, StepNum2, U1, U2, U3, R1, R2, R3)
        print("ret2的长度为:", len(ret2))
        # print(f"ret2的关键字：{ret2[1]}...")  # 打印前10个字符
        print(f"ret2的类型为: {type(ret2)}")
        if ret2[-1] != 0:
            print(f"获取振型结果失败，错误码: {ret2}")
            return modal_periods, modal_freqs, None
        
        NumberResults2 = ret2[0]
        Obj = ret2[1]
        Elm = ret2[2]
        LoadCase2 = ret2[3]
        StepType2 = ret2[4]
        StepNum2 = ret2[5]
        U1 = ret2[6]
        U2 = ret2[7]
        U3 = ret2[8]
        R1 = ret2[9]
        R2 = ret2[10]
        R3 = ret2[11]
        temp = ret2[12]  # 可能是多余的返回值，通常为0或None
        print(f"获取模态振型数据成功，共 {NumberResults2} 个节点的振型数据。")

        df_shapes = pd.DataFrame({
            "Obj": Obj,
            "Elm": Elm,
            "LoadCase": LoadCase2,
            "StepType": StepType2,
            "ModeNum": StepNum2,
            "U1": U1,
            "U2": U2,
            "U3": U3,
            "R1": R1,
            "R2": R2,
            "R3": R3,
            "temp": temp  # 如果U12是多余的，可以删除这一列
        })
        print(f"已获取{len(df_shapes)}个节点的振型数据。前5行：")
        print(df_shapes.head())
        
        return modal_periods, modal_freqs, df_shapes

    def get_node_mass(self, node_names="ALL"):
        """
        获取节点的质量信息，如果 node_names 为 "ALL"，则获取所有节点的质量。
        参数:
            model: SAP2000模型对象
            node_names: 节点名称列表，默认为"ALL"表示获取所有节点的质量信息

        返回:
            mass_info: 包含节点名称和对应质量的字典
        """
        mass_info = {}
        # 获取所有节点的质量
        m = [1, 1, 1, 1, 1, 1]  # 获取质量的参数，1表示获取所有质量分量

        if node_names == "ALL":
            ret = self.model.PointObj.GetNameList()
            if ret[-1] != 0:
                print(f"获取所有节点名称失败，错误代码: {ret[-1]}")
                return mass_info

            # number_points = ret[0]
            # print(f"模型中共有 {number_points} 个节点")
            point_names = ret[1]

            for point_name in point_names:
                # 调用GetMass方法获取所有节点的质量
                ret = self.model.PointObj.GetMass(point_name, m)
                if ret[-1] != 0:
                    print(f"获取节点 {point_name} 的质量失败，错误代码: {ret[-1]}")
                    continue
                masses = ret[0]
                mass_info[point_name] = masses

        return mass_info

    def get_node_coordinates(self, node_names="ALL"):
        """
        获取SAP2000模型的节点坐标信息,如果 node_names 为 "ALL"，则获取所有节点的坐标。
        参数:
            model: SAP2000模型对象
            node_names: 节点名称列表，默认为"ALL"表示获取所有节点的坐标
        返回：
            node_coords: 节点坐标的字典，包含节点ID和对应的坐标（x, y, z）
        """
        node_coords = {}
        if node_names == "ALL":
            # 获取所有节点的ID
            ret = self.model.PointObj.GetNameList()
            if ret[-1] != 0:
                print(f"获取节点列表失败，错误代码: {ret[-1]}")
                return node_coords

            number_points = ret[0]
            print(f"模型中共有 {number_points} 个节点")
            point_names = ret[1]

            for point_name in point_names:
                # 获取每个节点的坐标
                ret = self.model.PointObj.GetCoordCartesian(point_name)
                if ret[-1] != 0:
                    print(f"获取节点 {point_name} 的坐标失败，错误代码: {ret[-1]}")
                    continue
                x, y, z = ret[0], ret[1], ret[2]
                node_coords[point_name] = (x, y, z)

        # 获取刚性隔板约束中心节点名称
        ret = self.model.ConstraintDef.GetNameList()
        if ret[-1] == 0:
            print(f"获取刚性隔板约束中心节点名称成功，共{ret[0]}个约束")
            constraint_names_list = ret[1]  # 这是一个元组，包含约束名称
            diaphragm_constraint_coords = {}  # 创建一个新的字典来存储坐标
            # 遍历所有刚性隔板约束中心，获取他们的坐标
            for constraint_name in constraint_names_list:
                # 获取坐标
                ret = self.model.PointObj.GetCoordCartesian(constraint_name)
                x, y, z = ret[0], ret[1], ret[2]
                diaphragm_constraint_coords[constraint_name] = (x, y, z)  # 使用新的字典
                # print(f"刚性隔板约束 {constraint_name} 的中心坐标为: ({x:.3f}, {y:.3f}, {z:.3f})")

        return node_coords, diaphragm_constraint_coords
