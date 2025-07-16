# 信封识别程序

# 🎯 Yolo_PaddleOCR 项目指南

本项目结合了两大经典模型框架——**YOLOv5**（目标检测）和**PaddleOCR**（文本识别），实现对图像中的文本目标的高效定位与识别。以下是详细的使用步骤：

---

## 1. 创建并激活虚拟环境

推荐使用 Conda 来隔离依赖，确保环境干净、复现性强。

在终端执行以下命令，新建一个名为 `yolocr` 的环境并指定 Python 3.10 版本：

```bash
conda create -n olocr python=3.10
```

系统会提示安装必要的依赖，输入 `y` 确认即可。环境创建完成后，激活该环境：

```bash
conda activate yolocr
```

激活后你会在命令行提示符中看到 `(yolocr)`，表示你已经切入该环境。

---

## 2. 克隆项目仓库

选择一个合适的目录位置（比如 `~/projects` 或当前工作目录），然后拉取项目代码到本地。以下命令为示例，你需要替换为实际项目的仓库地址：

```bash
git clone https://github.com/junjia0512/Yolo_PaddleOCR.git
```

克隆完成后，会生成一个名为 `Yolo_Paddleocr` 或者你自定义的文件夹。

---

## 3. 安装依赖

进入项目目录：

```bash
cd Yolo_Paddleocr
```

通常，项目会提供 `requirements.txt`（或 `environment.yml`、`setup.py`）来记录所有依赖包和确切版本。如果是 `requirements.txt`，可运行：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

---

## 4.运行程序 🚀

在项目根目录下运行主脚本：

```bash
python yolo-pad.py
```

该脚本将调用 YOLOv5 进行目标检测，并通过 PaddleOCR 对检测到的文本区域执行识别操作。确保图像、视频或摄像头输入路径在脚本中已正确配置。