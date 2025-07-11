import os
import sys
import comtypes.client
import tkinter as tk # 用于创建GUI窗口
from tkinter import filedialog # 用于文件选择对话框
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def connect_to_sap2000():
    """
    连接到当前打开的SAP2000应用程序
    返回: SAP2000对象, 当前模型对象
    """
    try:
        # 连接到SAP2000
        mySapObject = comtypes.client.GetActiveObject("CSI.SAP2000.API.SapObject")
        print("成功连接到SAP2000应用程序")
        
        # 获取当前模型
        SapModel = mySapObject.SapModel
        print(f"当前模型文件: {SapModel.GetModelFilename()}")
        
        return mySapObject, SapModel
    except Exception as e:
        print(f"连接SAP2000失败: {e}")
        return None, None

def select_folder(default_folder=None, use_gui=True):
    """
    获取文件夹路径
    参数:
        default_folder: 默认文件夹路径
        use_gui: 是否使用图形界面选择文件夹
    返回: 所选文件夹路径
    """
    if default_folder and not use_gui:
        print(f"使用默认文件夹: {default_folder}")
        return default_folder
        
    if use_gui:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 如果有默认文件夹，就在对话框中初始化为该路径
        folder_path = filedialog.askdirectory(
            title="选择包含.TXM/.TXC/.TXK文件的文件夹",
            initialdir=default_folder if default_folder else None
        )
        
        if folder_path:
            print(f"已选择文件夹: {folder_path}")
            return folder_path
        else:
            # 如果用户取消选择但存在默认文件夹，则使用默认文件夹
            if default_folder:
                print(f"用户取消选择，使用默认文件夹: {default_folder}")
                return default_folder
            else:
                print("未选择任何文件夹")
                return None
    
    print("未指定文件夹路径")
    return None

def find_tx_files(folder_path):
    """
    在指定文件夹中查找.TXM、.TXC和.TXK文件
    返回: 文件列表（按类型分组）
    """
    txm_files = []
    txc_files = []
    txk_files = []
    
    if not folder_path:
        return txm_files, txc_files, txk_files
        
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        print(file)
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            lower_file = file.lower()
            if lower_file.endswith('.txm'):
                txm_files.append(file_path)
            elif lower_file.endswith('.txc'):
                txc_files.append(file_path)
            elif lower_file.endswith('.txk'):
                txk_files.append(file_path)

    print(f"找到 {len(txm_files)} 个.txm文件")
    print(f"找到 {len(txc_files)} 个.txc文件")
    print(f"找到 {len(txk_files)} 个.txk文件")

    return txm_files, txc_files, txk_files

def read_tx_file(file_path):
    """
    读取.TXM/.TXK文件中的数据
    这些文件通常包含模态形状数据或刚度矩阵数据，格式相同
    """
    # 确定文件类型
    file_type = "未知"
    if file_path.lower().endswith('.txm'):
        file_type = "TXM"
    elif file_path.lower().endswith('.txk'):
        file_type = "TXK"
    
    # 读取文件内容
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # 处理文件数据
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Note'):  # 忽略空行和注释
                values = line.split()
                data.append(values)
    
    # 转换为矩阵数据结构
    if data:
        # 将数据转换为数值类型
        numeric_data = []
        for row in data:
            if len(row) >= 3:  # 确保至少有三列数据
                try:
                    row_idx = int(row[0])
                    col_idx = int(row[1])
                    value = float(row[2])
                    numeric_data.append((row_idx, col_idx, value))
                except (ValueError, IndexError):
                    continue
        
        if numeric_data:
            # 确定矩阵的大小
            max_row = max(item[0] for item in numeric_data)
            max_col = max(item[1] for item in numeric_data)
            
            # 创建全零矩阵
            matrix = np.zeros((max_row + 1, max_col + 1))  # +1 因为索引从0开始
            
            # 填充矩阵
            for row_idx, col_idx, value in numeric_data:
                matrix[row_idx, col_idx] = value
            
            # 输出前10行和10列作为示意
            print(f"成功读取{file_type}文件: {os.path.basename(file_path)}")
            print("矩阵数据示例 (前10行10列):")
            display_rows = min(10, matrix.shape[0])
            display_cols = min(10, matrix.shape[1])
            print(matrix[:display_rows, :display_cols])
            
            # 返回完整矩阵和原始DataFrame
            df = pd.DataFrame(data[1:], columns=data[0] if len(data) > 0 else None)
            return {"matrix": matrix, "df": df, "file_type": file_type}
        else:
            print(f"无法从{file_type}文件创建矩阵: {file_path}")
            return pd.DataFrame()
    else:
        print(f"{file_type}文件为空: {file_path}")
        return pd.DataFrame()
            
def process_tx_data(txm_data, txk_data, output_folder):
    """
    处理读取到的TX文件数据
    根据具体需求进行调整
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 示例：将处理后的数据导出到CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理数据的通用函数
    def process_data(data, default_type="UNKNOWN"):
        if isinstance(data, dict) and 'matrix' in data:
            file_type = data.get('file_type', default_type)
            
            # 原始矩阵太大，不再保存完整矩阵数据
            print(f"跳过保存完整{file_type}矩阵数据（矩阵尺寸: {data['matrix'].shape[0]}×{data['matrix'].shape[1]}）")
            
            # 2. 保存为系数矩阵格式（行、列、值）
            coef_matrix_filename = f"{timestamp}_{file_type.lower()}_coef_matrix.csv"
            
            # 获取非零元素的位置和值
            non_zero_indices = np.nonzero(data['matrix'])
            rows = non_zero_indices[0]
            cols = non_zero_indices[1]
            values = data['matrix'][rows, cols]
            
            # 创建系数矩阵数据框并保存
            coef_df = pd.DataFrame({
                'row': rows,
                'column': cols,
                'value': values
            })
            coef_df.to_csv(os.path.join(output_folder, coef_matrix_filename), index=False)
            print(f"{file_type}系数矩阵已保存至 {coef_matrix_filename}")
            
            # 3. 保存矩阵统计信息
            stats_filename = f"{timestamp}_{file_type.lower()}_matrix_stats.txt"
            with open(os.path.join(output_folder, stats_filename), 'w') as f:
                matrix = data['matrix']
                non_zero_count = np.count_nonzero(matrix)
                total_elements = matrix.size
                sparsity = 1.0 - (non_zero_count / total_elements)
                
                f.write(f"矩阵维度: {matrix.shape[0]} × {matrix.shape[1]}\n")
                f.write(f"总元素数: {total_elements}\n")
                f.write(f"非零元素数: {non_zero_count}\n")
                f.write(f"稀疏度: {sparsity:.6f} ({sparsity*100:.2f}%)\n")
                f.write(f"矩阵最大值: {np.max(matrix)}\n")
                f.write(f"矩阵最小值: {np.min(matrix[np.nonzero(matrix)])}\n")
                f.write(f"矩阵平均值: {np.mean(matrix[np.nonzero(matrix)])}\n")
            print(f"{file_type}矩阵统计信息已保存至 {stats_filename}")
            
            # 4. 使用scipy保存稀疏矩阵格式（如果scipy可用）
            sparse_filename = f"{timestamp}_{file_type.lower()}_sparse.npz"
            sparse_matrix = sparse.csr_matrix(data['matrix'])
            sparse.save_npz(os.path.join(output_folder, sparse_filename), sparse_matrix)
            print(f"{file_type}稀疏矩阵已保存至 {sparse_filename}")
            
            # 5. 保存原始DataFrame
            if 'df' in data and not data['df'].empty:
                df_filename = f"{timestamp}_{file_type.lower()}_processed.csv"
                data['df'].to_csv(os.path.join(output_folder, df_filename), index=False)
                print(f"{file_type}原始数据已保存至 {df_filename}")
        elif not isinstance(data, dict) and hasattr(data, 'empty') and not data.empty:
            df_filename = f"{timestamp}_{default_type.lower()}_processed.csv"
            data.to_csv(os.path.join(output_folder, df_filename), index=False)
            print(f"{default_type}处理结果已保存至 {df_filename}")
    
    # 处理并保存TXM数据
    process_data(txm_data, "TXM")
    
    # 处理并保存TXK数据
    process_data(txk_data, "TXK")

def main():
    """
    主函数 - 连接SAP2000，读取TX文件并处理数据
    """
    # 连接到SAP2000
    sap_object, sap_model = connect_to_sap2000()
    if not sap_object or not sap_model:
        print("无法连接到SAP2000，程序终止")
        return
    
    # 获取当前打开的SAP2000模型文件路径
    model_path = sap_model.GetModelFilename()
    model_folder = os.path.dirname(model_path) if model_path else None
    
    # 询问用户是否使用默认文件夹
    use_default = True
    if model_folder:
        print(f"当前SAP2000模型所在文件夹: {model_folder}")
        # response = input("是否使用此文件夹作为默认路径? (y/n, 默认y): ").strip().lower()
        # use_default = response != 'n'
    
    # 如果使用默认文件夹，则直接使用模型文件夹；否则让用户选择
    if use_default and model_folder:
        folder_path = model_folder
    else:
        folder_path = select_folder(default_folder=model_folder)
        
    if not folder_path:
        print("未指定文件夹，程序终止")
        return
        
    # 查找TX文件
    txm_files, txc_files, txk_files = find_tx_files(folder_path)
    
    # 如果没有找到任何文件
    if not txm_files and not txc_files and not txk_files:
        print(f"在所选文件夹中未找到任何.TXM、.TXC或.TXK文件")
        return
    
    # 读取并处理第一个找到的各类型文件（示例）
    # 根据实际需求可以修改为处理全部文件
    txm_data = pd.DataFrame() # 初始化为空DataFrame
    txk_data = pd.DataFrame() # 初始化为空DataFrame

    # 只显示矩阵的一部分
    display_size = 100

    if txm_files:
        txm_data = read_tx_file(txm_files[0])
        plt.figure(figsize=(10, 8))
        plt.imshow(txm_data['matrix'][:display_size, :display_size], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("TXM Matrix Heatmap")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        

    if txk_files:
        txk_data = read_tx_file(txk_files[0])
        plt.figure(figsize=(10, 8))
        plt.imshow(txk_data['matrix'][:display_size, :display_size], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("TXK Matrix Heatmap")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        

    # 处理和保存数据
    # 获取当前文件夹路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "output","MCK_results")
    process_tx_data(txm_data, txk_data, output_folder)
    
    print("处理完成")
    plt.show()

if __name__ == "__main__":
    main()
