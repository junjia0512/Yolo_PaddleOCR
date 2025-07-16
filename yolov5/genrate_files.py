import os

def list_files(root_dir):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            # 拼接相对路径（以dataset为根目录）
            rel_path = os.path.relpath(os.path.join(root, file), start='.')
            file_paths.append(rel_path.replace('\\', '/'))  # 兼容Windows
    return file_paths

def save_list(file_list, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in file_list:
            f.write(item + '\n')

if __name__ == "__main__":
    images_dir = 'dataset/images'
    labels_dir = 'dataset/labels'

    images_list = list_files(images_dir)
    labels_list = list_files(labels_dir)

    save_list(images_list, 'images.txt')
    save_list(labels_list, 'labels.txt')

    print(f"共找到 {len(images_list)} 张图片，{len(labels_list)} 个标签。")
    print("已分别保存到 images.txt 和 labels.txt")