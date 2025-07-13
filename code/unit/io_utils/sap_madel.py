"""
SAP2000模型操作类
负责所有与SAP2000模型相关的操作
"""

import os
import pandas as pd
import comtypes.client
from collections import defaultdict
from .utils import get_timestamp

class SAP2000Model:
    """SAP2000模型管理器 - 负责所有SAP2000操作"""
    
    def __init__(self):
        self.model = None
        self.is_connected = False
    
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
        """添加刚性隔板 - 您的原始add_diaphragms代码"""
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
                print(f"按照指定标高筛选节点：{target_elevations}")
                filtered_nodes = []
                for node in node_info:
                    # 检查节点是否在任意目标标高附近
                    for elevation in target_elevations:
                        if abs(node["z"] - elevation) <= tolerance:
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

            # 保存节点信息到CSV文件
            df = pd.DataFrame(node_info)
            csv_path = os.path.join(os.getcwd(), "node_coordinates.csv")
            df.to_csv(csv_path, index=False)
            print(f"节点坐标已保存到: {csv_path}")            

            # 为每一组节点设置刚性隔板约束
            print(f"将为以下Z坐标创建刚性隔板约束: {list(node_z_coords.keys())}")

            for z_value, nodes in node_z_coords.items():
                constraint_name = f"Diaphragm_Z_{z_value}"  # 根据Z值生成约束名称
                
                # 首先定义刚性隔板的名称
                ret = self.model.ConstraintDef.SetDiaphragm(constraint_name, 3, "Global")  # 设置刚性隔板约束

                constraint_names.append(constraint_name)  # 保存约束名称

                # 设置刚性隔板约束
                for i in nodes:
                    self.model.PointObj.SetConstraint(i, constraint_name)

            print("刚性隔板约束设置完成！")
            return constraint_names, node_z_coords  # 返回创建的约束名称列表

        except Exception as e:
            print(f"创建刚性隔板时出错: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def add_wind_time_history_load(self, diaphragm_constraints, node_z_coords, wind_time_history_file=None, num_rows=None):
        """添加风荷载 - 您的原始add_wind_time_history_load代码"""
        # 这里放入您的完整add_wind_time_history_load函数代码
        # 将model参数改为self.model
        # 其他代码保持不变
        print("\n开始添加风荷载时程曲线...")
        
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
            
            # 查找最接近中心的节点
            closest_point = min(constraint_points, 
                                key=lambda p: ((p["x"]-avg_x)**2 + (p["y"]-avg_y)**2)**0.5)
                                
            center_point_name = f"WIND_CENTER_{constraint_name}"
            
            # 检查是否需要创建新节点
            if ((closest_point["x"]-avg_x)**2 + (closest_point["y"]-avg_y)**2)**0.5 > 1:
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
        MyTime = [i/fs for i in range(0, len(df))]
        print(f"时程数据共有 {len(df)} 行，采样频率 {fs}Hz")
        
        # 限制行数以加快测试速度
        num_columns = df.shape[1]
        if num_rows is None or num_rows > df.shape[0]:
            num_rows = df.shape[0]
        MyTime = MyTime[:num_rows]

        # 删除可能存在的旧荷载模式和时程函数
        # ... 其余代码保持不变，将model改为self.model
        
        return len(diaphragm_centers) * 3, diaphragm_centers  # 返回示例值
    
    def get_node_response_history(self, node_name, load_case="Wind_time_history", output_file=None, timestamp=None):
        """获取节点响应 - 您的原始get_node_response_history代码"""
        # 这里放入您的完整get_node_response_history函数代码
        # 将model参数改为self.model
        # 其他代码保持不变
        print(f"\n获取节点 {node_name} 在 {load_case} 工况下的位移响应时程...")
        
        # 简化示例，您可以替换为完整代码
        return [], [], []  # 返回时间、位移、加速度列表