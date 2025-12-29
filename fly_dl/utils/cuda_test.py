import torch
import torchaudio
import torchvision

print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"Torchaudio 版本: {torchaudio.__version__}")

# 检查 CUDA 可用性
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

# 创建简单张量测试
x = torch.randn(3, 4)
print(f"测试张量: {x}")
# GPU 测试（如果可用）
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(f"GPU 张量: {x_gpu.device}")

