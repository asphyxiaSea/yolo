import os

# 配置路径
label_dir = 'datasets/VisDrone/orign_labels/val'  # 你的 label 文件夹路径
save_dir = 'datasets/VisDrone/labels/val' # 过滤后的保存路径

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def filter_labels(target_class):
    for file_name in os.listdir(label_dir):
        if not file_name.endswith('.txt'):
            continue
            
        with open(os.path.join(label_dir, file_name), 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            data = line.split()
            if not data:
                continue

            if int(data[0]) == target_class:
                data[0] = '0' 
                filtered_lines.append(" ".join(data) + "\n")

        # 只有当文件中包含目标类别时才保存（可选，也可以保存空文件）
        if filtered_lines:
            with open(os.path.join(save_dir, file_name), 'w') as f:
                f.writelines(filtered_lines)

    print(f"处理完成！过滤后的文件已保存在: {save_dir}")

if __name__ == "__main__":
    filter_labels(target_class=0)