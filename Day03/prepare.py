### prepare.py

import os

# 创建保存路径的函数
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    # 使用绝对路径写入文件
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")

# 使用原始字符串（r''）处理路径中的空格
create_txt_file(r'D:\ShiXiDaiMa\Day03\image2\train', 'train.txt')
create_txt_file(r'D:\ShiXiDaiMa\Day03\image2\val', 'val.txt')
