import zipfile
import os

def unzip_file(zip_file_path, extract_to):
    """
    解压缩文件并存储到指定路径
    :param zip_file_path: 压缩文件路径
    :param extract_to: 解压后文件保存的目标文件夹路径
    """
    # 检查指定目录是否存在，如果不存在则创建
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # 打开并解压 zip 文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"文件已成功解压到: {extract_to}")

# 示例用法
zip_file = 'cloud.tsinghua.edu.cn/f/1ca72189fc4646368892/replica.zip'  # 替换为你的压缩文件路径
output_folder = 'data/experiment_data/split100'  # 替换为你想保存解压内容的文件夹路径
unzip_file(zip_file, output_folder)