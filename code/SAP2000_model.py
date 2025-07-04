import comtypes.client
import comtypes.automation
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
import pandas as pd

# 设置matplotlib支持中文显示
import matplotlib
# 设置全局字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

def connect_to_sap2000():
    """
    连接到当前运行的SAP2000实例或启动新实例
    
    返回:
        model: SAP2000模型对象，连接失败则返回None
    """
    try:
        # 创建SAP2000帮助对象
        helper = comtypes.client.CreateObject('SAP2000v1.Helper')
        helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
        
        # 尝试获取当前运行的SAP2000对象
        sap_object = helper.GetObject("CSI.SAP2000.API.SapObject")
        
        if sap_object is None:
            print("未获得当前运行的SAP2000实例，正在启动新实例...")
            program_path = 'C:\\Program Files\\Computers and Structures\\SAP2000 24\\SAP2000.exe'
            sap_object = helper.CreateObject(program_path)
            sap_object.ApplicationStart()
            print("已启动新的SAP2000实例")
            
        # 获取SAP2000模型对象
        model = sap_object.SapModel
        print("已成功连接到SAP2000模型")
        
        return model
    except Exception as e:
        print(f"连接SAP2000时发生错误：{e}")
        return None

def add_diaphragms(sap_model):
    """
    在SAP2000模型中创建刚性隔板
    
    参数:
        sap_model: SAP2000模型对象
        
    返回:
        diaphragms: 字典，键为楼层高度，值为刚性隔板名称和相关节点信息
    """
    try:
        print("正在识别模型中的楼层...")
        
        # 获取所有点对象
        ret, number_of_points = sap_model.PointObj.Count()
        if ret != 0:
            print(f"获取点对象数量失败，错误码：{ret}")
            return {}
            
        print(f"模型中共有 {number_of_points} 个节点")
        
        # 收集所有点的坐标和标签
        point_coords = {}  # 键为点的标签，值为坐标 [x, y, z]
        for i in range(1, number_of_points + 1):
            ret, name = sap_model.PointObj.GetNameList(i)
            if ret != 0:
                print(f"获取第 {i} 个点对象的名称失败，错误码：{ret}")
                continue
            
            point_name = name
            ret, x, y, z = sap_model.PointObj.GetCoordCartesian(point_name)
            if ret != 0:
                print(f"获取点 {point_name} 的坐标失败，错误码：{ret}")
                continue
                
            point_coords[point_name] = [x, y, z]
        
        # 按Z坐标分组，识别楼层
        floors = {}  # 键为楼层高度，值为该楼层的点集合
        for point_name, coords in point_coords.items():
            z = coords[2]
            # 使用近似相等，处理浮点数舍入误差
            z_rounded = round(z, 3)  # 保留3位小数
            if z_rounded not in floors:
                floors[z_rounded] = []
            floors[z_rounded].append(point_name)
        
        print(f"识别出 {len(floors)} 个楼层高度")
        
        # 创建刚性隔板
        diaphragms = {}
        for floor_height, points in floors.items():
            # 只为包含多个点的楼层创建刚性隔板
            if len(points) >= 3:  # 假设至少需要3个点才能形成一个平面
                diaphragm_name = f"DIAPH_Z{floor_height}"
                
                # 删除可能已存在的同名隔板
                try:
                    ret = sap_model.ConstraintDef.Delete(diaphragm_name)
                except:
                    pass
                    
                # 创建新的刚性隔板定义
                ret = sap_model.ConstraintDef.SetDiaphragm(diaphragm_name)
                if ret != 0:
                    print(f"创建刚性隔板 {diaphragm_name} 失败，错误码：{ret}")
                    continue
                
                # 将楼层上的点指定使用该刚性隔板
                for point_name in points:
                    ret = sap_model.PointObj.SetConstraint(point_name, diaphragm_name)
                    if ret != 0:
                        print(f"将点 {point_name} 指定给隔板 {diaphragm_name} 失败，错误码：{ret}")
                
                diaphragms[floor_height] = {
                    'name': diaphragm_name,
                    'points': points
                }
                print(f"已创建刚性隔板 {diaphragm_name}，包含 {len(points)} 个节点")
        
        return diaphragms
    except Exception as e:
        print(f"创建刚性隔板时发生错误：{e}")
        return {}

def apply_wind_loads_to_diaphragms(sap_model, diaphragms, wind_loads):
    """
    将风荷载应用到SAP2000模型的刚性隔板
    
    参数:
        sap_model: SAP2000模型对象
        diaphragms: 字典，包含刚性隔板信息
        wind_loads: 字典，包含处理后的风荷载数据
        
    返回:
        成功返回True，失败返回False
    """
    try:
        # 创建时程荷载定义
        print("正在创建风荷载时程定义...")
        
        # 对每个风荷载文件创建一个时程荷载工况
        for load_file, load_data in wind_loads.items():
            load_case_name = f"WIND_{os.path.splitext(load_file)[0]}"
            
            # 尝试删除可能存在的同名荷载工况
            try:
                ret = sap_model.LoadCases.Delete(load_case_name)
            except:
                pass
                
            # 创建新的时程荷载工况
            ret = sap_model.LoadCases.StaticLinear.SetCase(load_case_name)
            if ret != 0:
                print(f"创建荷载工况 {load_case_name} 失败，错误码：{ret}")
                continue
                
            # 获取时间步和楼层信息
            time_steps = load_data['time_steps']
            
            # 创建时程函数
            time_history_name = f"TH_{os.path.splitext(load_file)[0]}"
            
            # 尝试删除可能存在的同名时程函数
            try:
                ret = sap_model.Func.Delete(time_history_name)
            except:
                pass
            
            # 创建时程函数
            time_values = time_steps.tolist()
            force_values = np.ones_like(time_values)  # 初始为全1，后面会为每个楼层设置具体的力
            
            ret = sap_model.Func.FuncTH.SetUser(time_history_name, time_values, force_values.tolist())
            if ret != 0:
                print(f"创建时程函数 {time_history_name} 失败，错误码：{ret}")
                continue
            
            print(f"已创建时程函数 {time_history_name}，包含 {len(time_values)} 个时间步")
            
            # 为每个楼层/隔板应用风荷载
            for floor_height, diaphragm_info in diaphragms.items():
                # 找出最接近的楼层
                closest_floor = None
                min_diff = float('inf')
                
                for floor in load_data['floors']:
                    diff = abs(float(floor) - floor_height)
                    if diff < min_diff:
                        min_diff = diff
                        closest_floor = floor
                
                if closest_floor is None:
                    print(f"警告：在风荷载数据中找不到与楼层高度 {floor_height} 对应的数据")
                    continue
                
                # 获取该楼层的力
                floor_forces = load_data['forces'][closest_floor]
                fx_values = floor_forces['Fx']
                fy_values = floor_forces['Fy']
                fz_values = floor_forces['Fz'] if 'Fz' in floor_forces else np.zeros_like(fx_values)
                
                # 获取刚性隔板的主控节点
                diaphragm_name = diaphragm_info['name']
                points = diaphragm_info['points']
                
                # 选择第一个点作为主控点(或可以找质心)
                master_point = points[0]
                
                # 为X方向应用时程荷载
                load_pattern_name_x = f"{load_case_name}_X"
                ret = sap_model.LoadPatterns.Add(load_pattern_name_x, 6)  # 6表示风荷载类型
                
                ret = sap_model.PointObj.SetLoadForce(
                    master_point, 
                    load_pattern_name_x, 
                    [fx_values[0], 0, 0, 0, 0, 0], 
                    False, 
                    "Global", 
                    "All"
                )
                
                # 为Y方向应用时程荷载
                load_pattern_name_y = f"{load_case_name}_Y"
                ret = sap_model.LoadPatterns.Add(load_pattern_name_y, 6)
                
                ret = sap_model.PointObj.SetLoadForce(
                    master_point, 
                    load_pattern_name_y, 
                    [0, fy_values[0], 0, 0, 0, 0], 
                    False, 
                    "Global", 
                    "All"
                )
                
                # 如果有Z方向荷载，也应用
                if not np.all(fz_values == 0):
                    load_pattern_name_z = f"{load_case_name}_Z"
                    ret = sap_model.LoadPatterns.Add(load_pattern_name_z, 6)
                    
                    ret = sap_model.PointObj.SetLoadForce(
                        master_point, 
                        load_pattern_name_z, 
                        [0, 0, fz_values[0], 0, 0, 0], 
                        False, 
                        "Global", 
                        "All"
                    )
                
                print(f"已为隔板 {diaphragm_name} 应用风荷载")
        
        print("风荷载应用完成")
        return True
    except Exception as e:
        print(f"应用风荷载时发生错误：{e}")
        return False

def run_time_history_analysis(sap_model):
    """
    在SAP2000模型中运行风荷载时程分析
    
    参数:
        sap_model: SAP2000模型对象
        
    返回:
        分析成功返回分析结果信息，失败返回None
    """
    try:
        print("正在设置时程分析参数...")
        
        # 创建时程分析工况
        time_history_case_name = "WIND_TIME_HISTORY"
        
        # 尝试删除可能存在的同名分析工况
        try:
            ret = sap_model.LoadCases.Delete(time_history_case_name)
        except:
            pass
        
        # 创建模态分析工况（时程分析需要先进行模态分析）
        modal_case_name = "MODAL"
        try:
            ret = sap_model.LoadCases.Delete(modal_case_name)
        except:
            pass
            
        # 设置模态分析参数
        ret = sap_model.LoadCases.ModalEigen.SetCase(modal_case_name)
        ret = sap_model.LoadCases.ModalEigen.SetNumberModes(12)  # 设置模态数量
        ret = sap_model.LoadCases.ModalEigen.SetMaxCycles(30)
        
        # 创建时程分析工况
        ret = sap_model.LoadCases.ModalHistory.SetCase(time_history_case_name)
        
        # 设置时程分析参数
        ret = sap_model.LoadCases.ModalHistory.SetDampConstant(time_history_case_name, 0.05)  # 设置阻尼比
        ret = sap_model.LoadCases.ModalHistory.SetModalCase(time_history_case_name, modal_case_name)  # 指定模态分析工况
        
        # 添加风荷载
        # 获取所有的荷载模式
        ret, number_patterns, load_patterns = sap_model.LoadPatterns.GetNameList()
        if ret != 0:
            print(f"获取荷载模式列表失败，错误码：{ret}")
            return None
        
        # 添加与风相关的荷载模式到时程分析中
        wind_patterns = [pattern for pattern in load_patterns if "WIND" in pattern.upper()]
        for pattern in wind_patterns:
            func_name = f"TH_{pattern.replace('WIND_', '').split('_')[0]}"
            scale_factor = 1.0
            time_type = 1  # 1表示周期性，0表示非周期性
            ret = sap_model.LoadCases.ModalHistory.SetLoads(
                time_history_case_name, 
                1, 
                [pattern], 
                [func_name], 
                [scale_factor], 
                [time_type]
            )
            
            if ret != 0:
                print(f"为时程分析添加荷载模式 {pattern} 失败，错误码：{ret}")
            else:
                print(f"已为时程分析添加荷载模式 {pattern}")
                
        # 设置时间步和输出
        time_step = 0.02  # 秒
        num_steps = 1000  # 根据需要调整
        ret = sap_model.LoadCases.ModalHistory.SetTimeStep(time_history_case_name, num_steps, time_step)
        
        # 设置要运行的分析类型
        ret = sap_model.Analyze.SetRunCaseFlag(modal_case_name, True)
        ret = sap_model.Analyze.SetRunCaseFlag(time_history_case_name, True)
        
        # 运行分析
        print("正在运行分析...")
        ret = sap_model.Analyze.RunAnalysis()
        
        if ret != 0:
            print(f"运行分析失败，错误码：{ret}")
            return None
            
        print("分析运行完成")
        
        # 返回分析结果信息
        return {
            'modal_case': modal_case_name,
            'time_history_case': time_history_case_name,
            'time_step': time_step,
            'num_steps': num_steps
        }
    except Exception as e:
        print(f"运行时程分析时发生错误：{e}")
        return None

def export_displacement_acceleration_curves(sap_model, nodes_to_export, result_folder=None):
    """
    导出指定节点的位移和加速度时程曲线
    
    参数:
        sap_model: SAP2000模型对象
        nodes_to_export: 需要导出结果的节点标识列表
        result_folder: 结果输出文件夹，默认为None时使用默认文件夹
        
    返回:
        成功返回结果数据，失败返回None
    """
    try:
        print("正在导出结果数据...")
        
        # 确保选择时程分析结果
        ret = sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        ret = sap_model.Results.Setup.SetCaseSelectedForOutput("WIND_TIME_HISTORY")
        
        if len(nodes_to_export) == 0:
            # 如果没有指定节点，则获取所有节点
            ret, number_of_points = sap_model.PointObj.Count()
            if ret != 0:
                print(f"获取点对象数量失败，错误码：{ret}")
                return None
                
            ret, node_names = sap_model.PointObj.GetNameList(0)
            if ret != 0:
                print(f"获取所有节点名称失败，错误码：{ret}")
                return None
                
            # 仅选择前10个节点导出结果（可以根据需要修改）
            nodes_to_export = node_names[:10]
        
        results = {}
        
        # 创建结果文件夹
        if result_folder is None:
            results_dir = Path("D:/MyFiles/SAP2000/code/Results")
        else:
            results_dir = Path(result_folder)
            
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # 为每个节点导出位移和加速度时程曲线
        for node_name in nodes_to_export:
            print(f"正在导出节点 {node_name} 的结果...")
            
            # 导出位移时程
            ret, obj_count, element_name, load_case, step_type, step_num, u1, u2, u3, r1, r2, r3 = (
                sap_model.Results.JointDispl(node_name, 0)
            )
            
            if ret != 0:
                print(f"获取节点 {node_name} 的位移结果失败，错误码：{ret}")
                continue
            
            # 导出加速度时程
            ret, obj_count, element_name, load_case, step_type, step_num, a1, a2, a3, ar1, ar2, ar3 = (
                sap_model.Results.JointAcc(node_name, 0)
            )
            
            if ret != 0:
                print(f"获取节点 {node_name} 的加速度结果失败，错误码：{ret}")
                continue
            
            # 存储结果
            node_results = {
                'displacement': {
                    'x': u1,
                    'y': u2,
                    'z': u3,
                    'rx': r1,
                    'ry': r2,
                    'rz': r3,
                    'step_num': step_num
                },
                'acceleration': {
                    'x': a1,
                    'y': a2,
                    'z': a3,
                    'rx': ar1,
                    'ry': ar2,
                    'rz': ar3,
                    'step_num': step_num
                }
            }
            
            results[node_name] = node_results
            
            # 绘制位移和加速度时程曲线
            plt.figure(figsize=(12, 10))
            
            # 位移时程曲线
            plt.subplot(2, 1, 1)
            plt.plot(step_num, u1, 'r-', label='X方向')
            plt.plot(step_num, u2, 'g-', label='Y方向')
            plt.plot(step_num, u3, 'b-', label='Z方向')
            plt.xlabel('时间步')
            plt.ylabel('位移 (mm)')
            plt.title(f'节点 {node_name} 位移时程曲线')
            plt.grid(True)
            plt.legend()
            
            # 加速度时程曲线
            plt.subplot(2, 1, 2)
            plt.plot(step_num, a1, 'r-', label='X方向')
            plt.plot(step_num, a2, 'g-', label='Y方向')
            plt.plot(step_num, a3, 'b-', label='Z方向')
            plt.xlabel('时间步')
            plt.ylabel('加速度 (mm/sec²)')
            plt.title(f'节点 {node_name} 加速度时程曲线')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
            # 保存图形
            plt.savefig(results_dir / f'Node_{node_name}_Results.png')
            plt.close()
            
            # 保存CSV数据
            df_disp = pd.DataFrame({
                'Step': step_num,
                'U1': u1,
                'U2': u2,
                'U3': u3,
                'R1': r1,
                'R2': r2,
                'R3': r3
            })
            
            df_acc = pd.DataFrame({
                'Step': step_num,
                'A1': a1,
                'A2': a2,
                'A3': a3,
                'AR1': ar1,
                'AR2': ar2,
                'AR3': ar3
            })
            
            df_disp.to_csv(results_dir / f'Node_{node_name}_Displacement.csv', index=False)
            df_acc.to_csv(results_dir / f'Node_{node_name}_Acceleration.csv', index=False)
            
            print(f"节点 {node_name} 的结果已导出")
        
        print(f"所有结果已导出到文件夹：{results_dir}")
        return results
    except Exception as e:
        print(f"导出结果时发生错误：{e}")
        return None

class SAP2000Model:
    def __init__(self, beam_connections, column_connections, model_path):
        self.joint_tags = []
        self.joint_coords = []

        self.beam_tags = []
        self.beam_connections = beam_connections
        self.beam_section = []

        self.column_tags = []
        self.column_connections = column_connections
        self.column_section = []

        self.joint_displacements = []
        self.joint_Accelerations = []
        self.joint_Velocities = []

        self.program_path = 'C:\\Program Files\\Computers and Structures\\SAP2000 24\\SAP2000.exe' # SAP2000程序路径
        self.model_path = model_path
        self.sap_object = None
        self.model = None

    def init_sap2000(self): # 初始化SAP2000
        helper = comtypes.client.CreateObject('SAP2000v1.Helper') # 创建SAP2000帮助对象
        helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper) # 获取SAP2000帮助对象
        
        self.sap_object = helper.GetObject("CSI.SAP2000.API.SapObject") # 获取当前运行的SAP2000对象
        if self.sap_object is None:
            print('未获得当前运行的SAP2000.\n正在打开SAP2000...')
            self.sap_object = helper.CreateObject(self.program_path)
        self.sap_object.ApplicationStart()      # 启动SAP2000
        self.model = self.sap_object.SapModel   # 获取SAP2000模型对象
        self.model.InitializeNewModel()         # 初始化新模型
        # self.File.OpenFile(self.model_path)   # 打开模型文件
        self.model.File.NewBlank()              # 新建空白模型
        self.model.File.Save(self.model_path)   # 保存模型
        self.model.SetPresentUnits(9)           # 设置单位为kN_mm_C
        self.sap_object.Visible = True          # 设置SAP2000界面可见

    def create_joints(self,joint_coords): # 创建节点
        for i, coord in enumerate(joint_coords):
            x, y, z = coord
            # print(f"Adding joint {i + 1} at coordinates ({x}, {y}, {z})")
            self.model.PointObj.AddCartesian(x, y, z, '', str(i + 1))   # 添加节点
            
            self.joint_tags.append(i + 1)
            self.joint_coords.append(coord)

    def define_material(self, material_name, material_type, fc, E): # 定义材料
        region = "China"
        standard = "GB"
        Grade = 'GB50010 ' + material_name
        print(f'设置材料等级为：{Grade}')
        self.model.PropMaterial.AddMaterial(material_name, material_type, region, standard, Grade)
        nu = 0.2
        A = 1.0e-5
        return material_name

    def define_section(self, section_name, material_name, b, h):
        ret = self.model.PropFrame.SetRectangle(section_name, material_name, h, b)
        if ret != 0:
            print(f"Failed to define section {section_name}")
        # 设置截面属性修正系数
        ModValues = [1, 0, 0, 1, 1, 1, 1, 1]  # 修正系数数组
        ret = self.model.PropFrame.SetModifiers(section_name, ModValues)
        if ret[-1] != 0:
            print(f"Failed to set modifiers for section {section_name}")
        return section_name

    def create_beams(self, section_name):
        for i, connection in enumerate(self.column_connections+self.beam_connections):
            start_joint, end_joint = connection
            # print(f"Adding beam from joint {start_joint} to joint {end_joint}")
            ret = self.model.FrameObj.AddByPoint(str(start_joint), str(end_joint), '', '', f'B{i + 1}')
            ret = self.model.FrameObj.SetSection(f'B{i + 1}', f'{section_name[i]}')
            # if ret != 0:
            #     print(f"Failed to set section {beam_section_name} to frame elements")

    def create_constraints(self,joint_tags_boundry_conditions):
        for i in joint_tags_boundry_conditions:
            self.model.PointObj.SetRestraint(str(i), [True, True, True, True, True, True])

    # 创建刚性隔板
    def create_rigid_diaphragm(self):
        floor_grops = {}
        for joint in self.joint_tags:
            x,y,z = self.joint_coords[joint-1]
            if z not in floor_grops:
                floor_grops[z] = []
            floor_grops[z].append(joint)
        # print(f'floor_grops:{floor_grops}')

        # 创建刚性隔板
        for floor in floor_grops:
            joints_tag = floor_grops[floor]
            # print(f'joints_tag:{joints_tag}')
            # 定义刚性隔板名称：这里以刚性隔板的节点编号为名称
            self.model.ConstraintDef.SetDiaphragm(f'floor_{floor}')
            for i in joints_tag:
                # print(f'{i}---floor_{floor}')
                self.model.PointObj.SetConstraint(str(i),f'floor_{floor}')
    
            
    def create_loads(self, joint_tags, load_values):
        # try:
        #     self.model.LoadPatterns.Delete("DEAD")
        #     self.model.LoadPatterns.Delete("Live")
        #     self.model.LoadPatterns.Delete("Wind_X")
        #     self.model.LoadPatterns.Delete("MODAL")
        # except:
        #     print("Failed to delete load patterns")

        # 定义荷载模式
        ret = self.model.LoadPatterns.Add("Live", 3, 0, True)
        if ret != 0:
            print(f"添加活载模式Live失败，错误码: {ret}")
        ret = self.model.LoadPatterns.Add("Wind_X", 6, 0, True)
        if ret != 0:
            print(f"添加风载模式Wind_X失败，错误码: {ret}")
        ret = self.model.LoadPatterns.Add("Wind_Y", 6, 0, True)
        if ret != 0:
            print(f"添加风载模式Wind_Y失败，错误码: {ret}")
        
        # 定义质量源
        LoadPat = ["DEAD", "Live"]  # Define load patterns
        SF = [1.0, 0.5]  # Define scale factors
        ret = self.model.SourceMass.SetMassSource("MyMassSource", True, True, True, True, 2, LoadPat, SF)
        if ret[-1] != 0:
            print(f"设置自重工况DEAD失败，错误码: {ret}")
        ret = self.model.SourceMass.SetDefault("MyMassSource")

        # number_points = self.model.PointObj.Count() # 获取节点总数
        for joint_tag, load_value in zip(joint_tags, load_values):
            ret = self.model.PointObj.SetLoadForce(str(joint_tag), "Wind_X", load_value, False, "Global")
        


    def create_analysis_combos(self):
        try:
            self.model.RespCombo.Delete("Combo1")
        except:
            print("Failed to delete response Combo1")

        ret = self.model.RespCombo.Add("Combo2", 1)
        if ret != 0:
            print(f"添加分析组合失败，错误码: {ret}")
        ret = self.model.RespCombo.SetCaseList("Combo2", 0, "DEAD", 1.3)
        ret = self.model.RespCombo.SetCaseList("Combo2", 0, "Live", 1.3)
        ret = self.model.RespCombo.SetCaseList("Combo2", 0, "Wind_X", 1.5)

    def create_analysis_parameters(self):
        ret = self.model.Analyze.SetRunCaseFlag("DEAD", True)
        if ret != 0:
            print("Failed to set run case flag for load case DEAD")
        ret = self.model.Analyze.SetRunCaseFlag("Live", False)
        if ret != 0:
            print("Failed to set run case flag for load case Live")
        ret = self.model.Analyze.SetRunCaseFlag("Wind_X", True)
        if ret != 0:
            print("Failed to set run case flag for load case Wind_X")
        ret = self.model.Analyze.SetRunCaseFlag("MODAL", False)
        if ret != 0:
            print("Failed to set run case flag for load case MODAL")

    def create_analysis_model(self, analysis_type=1):
        """创建分析模型
        
        Args:
            analysis_type: 分析类型，1为静力分析，2为动力分析，默认为1
        """
        # 根据分析类型创建不同的分析模型
        if analysis_type == 1:  # 静力分析
            print("直接创建静力分析模型")
        elif analysis_type == 2:  # 动力分析
            print("创建动力分析模型")
            # 设置时程分析参数
            ret = self.model.LoadCases.ModalEigen.SetCase("MODAL")
            ret = self.model.LoadCases.ModalEigen.SetNumberModes(10)
            ret = self.model.LoadCases.ModalEigen.SetEigenShift(0.0)
            ret = self.model.LoadCases.ModalEigen.SetMaxCycles(20)
            ret = self.model.LoadCases.ModalEigen.SetToleranceForDiagonalTerms(1.0e-9)
            ret = self.model.LoadCases.ModalEigen.SetTrialsPerVector(20)
            
        # 创建分析模型
        self.model.Analyze.CreateAnalysisModel()

    def run_analysis(self):
        ret = self.model.Analyze.RunAnalysis()
        if ret != 0:
            print(f"分析运行失败，错误代码：{ret}")
        else:
            print("SAP2000分析运行成功")

    def get_results(self, joint_tags):
        
        Obj, Elm, ACase, StepType, StepNum = [], [], [], [], []
        U1, U2, U3, R1, R2, R3 = [], [], [], [], [], []
        ObjectElm, NumberResults = 0, 0

        try:
            # 选择输出结果的工况
            ret = self.model.Results.Setup.DeselectAllCasesAndCombosForOutput()
            if ret != 0:
                print(f"取消选择所有工况失败，错误代码：{ret}")
                return []
            
            ret = self.model.Results.Setup.SetCaseSelectedForOutput("Wind_X")  # 改为您实际需要的工况
            if ret != 0:
                print(f"选择输出工况失败，错误代码：{ret}")
                return []
            
            # 获取关节位移，使用ALL获取所有节点的结果
            [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = \
            self.model.Results.JointDispl(str(joint_tags), ObjectElm, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
            
            if ret != 0:
                print(f"获取节点位移结果失败，错误代码：{ret}")
                return []
            
            result = [U1[0], U2[0], U3[0], R1[0], R2[0], R3[0]]
            return result
            
        except Exception as e:
            print(f"获取结果时出错: {e}")
            import traceback # 引入traceback模块,打印错误信息,可以追踪错误
            traceback.print_exc() # 打印异常信息
            return []

    def visualize_results(self):
        plt.plot()
        plt.title('Beam Shear Force')
        plt.xlabel('Element Index')
        plt.ylabel('Shear Force (kN)')
        plt.show()

    def save_model(self):
        # 检查self.model_path的倒数第二个文件夹是否存在
        # 如果不存在，则创建该文件夹
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        # 检查self.model_path的最后一个文件夹是否存在
        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)
        self.model.File.Save(self.model_path)
        

    def close_sap2000(self):
        self.sap_object.ApplicationExit(True)
        self.sap_object = None
        self.model = None

    def define_time_history_load(self, gm_file, scale_factor=1.0):
        """定义时程荷载（地震波）
        
        Args:
            gm_file: 地震波文件路径
            scale_factor: 地震波缩放系数，默认为1.0
        """
        # 读取地震波数据
        try:
            gm_data = np.loadtxt(gm_file)
            time_points = gm_data[:, 0]
            accel_values = gm_data[:, 1] * scale_factor
            
            # 创建函数型时程
            ret = self.model.Func.FuncTH.Delete("ELCENTRO")
            ret = self.model.Func.FuncTH.SetFromFile("ELCENTRO", gm_file, 1, 0, 1, 2, 
                                                    True, 1.0, 0.0, 0.0, 0)
            
            # 创建时程分析案例
            ret = self.model.LoadCases.Delete("TimeHistory")
            ret = self.model.LoadCases.StaticLinear.SetCase("TimeHistory")
            
            print(f"成功读取地震波文件，共{len(time_points)}个时间点")
            return True
        except Exception as e:
            print(f"读取地震波文件或定义时程分析失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def get_modal_results(self, num_modes=3):
        """获取模态分析结果
        
        Args:
            num_modes: 要获取的模态数量
            
        Returns:
            modal_periods: 模态周期列表
            modal_freqs: 模态频率列表
        """
        try:
            # 确保MODAL工况被选择
            ret = self.model.Results.Setup.DeselectAllCasesAndCombosForOutput()
            ret = self.model.Results.Setup.SetCaseSelectedForOutput("MODAL")
            
            modal_periods = []
            modal_freqs = []
            
            for mode in range(1, num_modes + 1):
                [period, frequency, circular_freq, eigen_value, ret] = self.model.Results.ModalPeriod(mode)
                if ret == 0:
                    modal_periods.append(period)
                    modal_freqs.append(frequency)
                else:
                    print(f"获取模态 {mode} 结果失败，错误码: {ret}")
            
            return modal_periods, modal_freqs
        except Exception as e:
            print(f"获取模态分析结果失败: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def export_results_to_file(self, output_dir):
        """将分析结果导出到文件
        
        Args:
            output_dir: 输出目录
        """
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 导出位移结果
            disp_file = output_path / "nodal_displacements.txt"
            ret = self.model.DatabaseTables.SetLoadCasesSelectedForDisplay(["Wind_X"])
            ret = self.model.DatabaseTables.SetTableKeyNameList("Joint Displacements", ["Joint"])
            ret = self.model.DatabaseTables.ExportToTextFile(str(disp_file), "Joint Displacements")
            
            # 导出反力结果
            force_file = output_path / "joint_reactions.txt"
            ret = self.model.DatabaseTables.ExportToTextFile(str(force_file), "Joint Reactions")
            
            print(f"分析结果已导出到 {output_dir}")
            return True
        except Exception as e:
            print(f"导出结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def plot_displacement_results(self, output_dir=None):
        """绘制位移结果
        
        Args:
            output_dir: 输出目录，如果为None则不保存图片
        """
        try:
            # 获取所有节点的位移
            all_displacements = []
            for tag in self.joint_tags:
                disp = self.get_results(tag)
                if disp:
                    all_displacements.append(disp)
            
            if not all_displacements:
                print("没有可用的位移数据可绘制")
                return False
                
            # 转换为NumPy数组便于处理
            all_displacements = np.array(all_displacements)
            
            # 创建绘图
            plt.figure(figsize=(12, 8))
            
            # 绘制X方向位移
            plt.subplot(2, 2, 1)
            heights = [self.joint_coords[i-1][2] for i in self.joint_tags]
            plt.scatter(all_displacements[:, 0], heights, c='b', alpha=0.7)
            plt.title('X方向位移')
            plt.xlabel('位移 (mm)')
            plt.ylabel('高度 (mm)')
            plt.grid(True)
            
            # 绘制Y方向位移
            plt.subplot(2, 2, 2)
            plt.scatter(all_displacements[:, 1], heights, c='r', alpha=0.7)
            plt.title('Y方向位移')
            plt.xlabel('位移 (mm)')
            plt.ylabel('高度 (mm)')
            plt.grid(True)
            
            # 绘制Z方向位移
            plt.subplot(2, 2, 3)
            plt.scatter(all_displacements[:, 2], heights, c='g', alpha=0.7)
            plt.title('Z方向位移')
            plt.xlabel('位移 (mm)')
            plt.ylabel('高度 (mm)')
            plt.grid(True)
            
            # 绘制总位移
            plt.subplot(2, 2, 4)
            total_disp = np.sqrt(all_displacements[:, 0]**2 + all_displacements[:, 1]**2 + all_displacements[:, 2]**2)
            plt.scatter(total_disp, heights, c='purple', alpha=0.7)
            plt.title('总位移')
            plt.xlabel('位移 (mm)')
            plt.ylabel('高度 (mm)')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(output_path / "displacement_results.png"), dpi=300)
                print(f"位移结果图已保存至 {output_path / 'displacement_results.png'}")
            
            plt.show()
            return True
        except Exception as e:
            print(f"绘制位移结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """测试 SAP2000Model 类功能"""
    print("="*50)
    print("SAP2000Model 测试程序")
    print("="*50)
    
    # 设置测试参数
    analysis_type = 1  # 1为静力分析，2为动力分析
    print(f"测试分析类型: {analysis_type} ({'静力分析' if analysis_type == 1 else '动力分析'})")
    
    # 创建输出目录
    output_dir = './data/output/sap2000_test'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置模型基本信息
    joint_coords = np.array([
        [0, 0, 0], [6000, 0, 0], [12000, 0, 0],
        [0, 6000, 0], [6000, 6000, 0], [12000, 6000, 0],
        [0, 0, 4000], [6000, 0, 4000], [12000, 0, 4000],
        [0, 6000, 4000], [6000, 6000, 4000], [12000, 6000, 4000],
        [0, 0, 8000], [6000, 0, 8000], [12000, 0, 8000],
        [0, 6000, 8000], [6000, 6000, 8000], [12000, 6000, 8000],
    ])
    
    # 定义梁柱连接
    column_connections = [
        [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12],  # 一层柱
        [7, 13], [8, 14], [9, 15], [10, 16], [11, 17], [12, 18],  # 二层柱
    ]
    
    # 定义梁连接（X方向和Y方向）
    beam_connections_x = [
        [7, 8], [8, 9], [10, 11], [11, 12],  # 二层X方向
        [13, 14], [14, 15], [16, 17], [17, 18],  # 三层X方向
    ]
    
    beam_connections_y = [
        [7, 10], [8, 11], [9, 12],  # 二层Y方向
        [13, 16], [14, 17], [15, 18],  # 三层Y方向
    ]
    
    beam_connections = beam_connections_x + beam_connections_y
    
    # 创建SAP2000模型
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"D:\\MyFiles\\PhD\\optimization\\Model\\creatmodel\\Models_FEM\\Model_SAP2000_test_{timestamp}\\Model_SAP2000_test_{timestamp}.sdb"

    try:
        print("\n1. 初始化SAP2000模型...")
        sap_model = SAP2000Model(beam_connections, column_connections, model_path)
        sap_model.init_sap2000()
        
        print("\n2. 创建节点...")
        sap_model.create_joints(joint_coords)
        print(f"  创建了 {len(sap_model.joint_tags)} 个节点")
        
        print("\n3. 定义材料和截面...")
        beam_material_name = sap_model.define_material("C60", 2, 40, 3.6e4)
        column_material_name = sap_model.define_material("C80", 2, 50, 3.8e4)
        
        # 创建不同尺寸的截面
        section_names = []
        
        # 柱截面 - 不同层的柱截面尺寸不同
        for i in range(len(column_connections)):
            if i < 6:  # 一层柱
                section_name = sap_model.define_section(f"Column600x600_{i+1}", column_material_name, 600, 600)
            else:  # 二层柱
                section_name = sap_model.define_section(f"Column500x500_{i+1}", column_material_name, 500, 500)
            section_names.append(section_name)
        
        # 梁截面 - X方向和Y方向可以使用不同的截面
        for i in range(len(beam_connections)):
            if i < len(beam_connections_x):  # X方向梁
                section_name = sap_model.define_section(f"BeamX300x600_{i+1}", beam_material_name, 300, 600)
            else:  # Y方向梁
                section_name = sap_model.define_section(f"BeamY300x500_{i+1}", beam_material_name, 300, 500)
            section_names.append(section_name)
        
        print(f"  创建了 {len(section_names)} 个截面")
        
        print("\n4. 创建梁柱构件...")
        sap_model.create_beams(section_names)
        
        print("\n5. 创建边界约束...")
        joint_tags_boundry_conditions = list(range(1, 7))  # 底部6个节点固定
        sap_model.create_constraints(joint_tags_boundry_conditions)
        
        print("\n6. 创建刚性隔板...")
        sap_model.create_rigid_diaphragm()
        
        print("\n7. 应用荷载...")
        if analysis_type == 1:  # 静力分析
            # 创建风荷载 - 只施加到顶层节点
            load_values = [[0, 0, 0, 0, 0, 0] for _ in range(len(sap_model.joint_tags))]
            for i in range(13, 19):  # 顶层节点
                load_values[i-1] = [30, 0, 0, 0, 0, 0]  # X方向30kN风荷载
            
            sap_model.create_loads(sap_model.joint_tags, load_values)
            
        elif analysis_type == 2:  # 动力分析
            # 读取地震波
            gm_file = r"D:\MyFiles\git\Highrise-Structure-Optimization\tests\ELCENTRO.txt"
            if Path(gm_file).exists():
                print(f"  使用地震波文件: {gm_file}")
                sap_model.define_time_history_load(gm_file, scale_factor=1.0)
            else:
                print(f"  地震波文件不存在: {gm_file}")
                # 创建简单荷载代替
                load_values = [[0, 0, 0, 0, 0, 0] for _ in range(len(sap_model.joint_tags))]
                sap_model.create_loads(sap_model.joint_tags, load_values)
        
        print("\n8. 创建分析参数...")
        sap_model.create_analysis_combos()
        sap_model.create_analysis_parameters()
        
        print("\n9. 创建分析模型...")
        sap_model.create_analysis_model(analysis_type)
        
        print("\n10. 保存模型...")
        sap_model.save_model()
        
        print("\n11. 运行分析...")
        sap_model.run_analysis()
        
        print("\n12. 获取结果...")
        if analysis_type == 1:  # 静力分析
            # 获取顶层节点位移结果
            for i in range(13, 19):
                result = sap_model.get_results(joint_tags=i)
                print(f"  节点 {i} 位移结果: {result}")
                
            # 可视化位移结果
            sap_model.plot_displacement_results(output_dir)
            
            # 导出结果到文件
            sap_model.export_results_to_file(output_dir)
            
        elif analysis_type == 2:  # 动力分析
            # 获取模态分析结果
            modal_periods, modal_freqs = sap_model.get_modal_results(num_modes=6)
            print("\n  模态分析结果:")
            for i, (period, freq) in enumerate(zip(modal_periods, modal_freqs)):
                print(f"  模态 {i+1}: 周期 = {period:.4f} 秒, 频率 = {freq:.4f} Hz")
        
        print("\n13. 测试完成")
        print(f"  模型文件已保存至: {model_path}")
        print(f"  结果已保存至: {output_dir}")
        
        # # 询问用户是否关闭SAP2000
        # close_sap = input("\n是否关闭SAP2000? (y/n): ")
        # if close_sap.lower() == 'y':
        #     sap_model.close_sap2000()
        #     print("SAP2000已关闭")
        # else:
        #     print("SAP2000保持打开状态")
            
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
