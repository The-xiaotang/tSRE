import torch

# 打印PyTorch版本
print(f"PyTorch version: {torch.__version__}")

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 如果CUDA可用，打印详细信息
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

