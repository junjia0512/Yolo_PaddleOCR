import os
import shutil
from pathlib import Path

# 1. 设置路径 (请根据实际路径修改)
base_path = "dataset/custom_data_1"
train_images_dir = os.path.join(base_path, "train_pic")
train_labels_dir = os.path.join(base_path, "train_plab")
val_images_dir = os.path.join(base_path, "val_pic")
val_labels_dir = os.path.join(base_path, "val_lab")

# 2. 创建文件夹
Path(train_images_dir).mkdir(exist_ok=True)
Path(train_labels_dir).mkdir(exist_ok=True)
Path(val_images_dir).mkdir(exist_ok=True)
Path(val_labels_dir).mkdir(exist_ok=True)

# 3. 准备路径列表文件
train_images_list = []
train_labels_list = []
val_images_list = []
val_labels_list = []

# 4. 处理文件函数
def organize_files(file_type, files):
    for file in files:
        # 获取完整路径
        src_path = os.path.join(base_path, file)
        
        # 根据文件名判断类型
        if file.startswith("train_"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                dest = os.path.join(train_images_dir, file)
                train_images_list.append(dest)
            elif file.lower().endswith('.txt'):
                dest = os.path.join(train_labels_dir, file)
                train_labels_list.append(dest)
        elif file.startswith("val_"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                dest = os.path.join(val_images_dir, file)
                val_images_list.append(dest)
            elif file.lower().endswith('.txt'):
                dest = os.path.join(val_labels_dir, file)
                val_labels_list.append(dest)
        
        # 移动文件
        shutil.move(src_path, dest)
        print(f"Moved: {file} → {os.path.basename(os.path.dirname(dest))}")

# 5. 列出所有文件
all_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

# 6. 组织文件
organize_files("所有文件", all_files)

# 7. 创建路径列表文件
def save_path_list(path_list, filename):
    with open(os.path.join(base_path, filename), "w") as f:
        f.write("\n".join(path_list))

save_path_list(train_images_list, "train_images.txt")
save_path_list(train_labels_list, "train_labels.txt")
save_path_list(val_images_list, "val_images.txt")
save_path_list(val_labels_list, "val_labels.txt")

print("\n整理完成!")
print(f"训练图片数量: {len(train_images_list)}")
print(f"训练标签数量: {len(train_labels_list)}")
print(f"验证图片数量: {len(val_images_list)}")
print(f"验证标签数量: {len(val_labels_list)}")
print(f"路径列表已保存至 {base_path} 下的txt文件中")