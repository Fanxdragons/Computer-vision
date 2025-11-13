import numpy as np
import cv2
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import convolve, gaussian_filter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import glob
import sys

# 解决 matplotlib 中文乱码和负号显示问题
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 全局设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {DEVICE}")

# 图像预处理，用于CNN模型
PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# 1. 质量评估函数
# =============================================================================

def calculate_psnr(original, compressed):
    """计算PSNR（峰值信噪比）"""
    if original.shape != compressed.shape:
        orig_h, orig_w = original.shape[:2]
        comp_h, comp_w = compressed.shape[:2]
        # 居中裁剪原始图像
        start_h = (orig_h - comp_h) // 2
        start_w = (orig_w - comp_w) // 2
        original = original[start_h:start_h + comp_h, start_w:start_w + comp_w, :]

    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def calculate_ssim(original, compressed):
    """计算SSIM（结构相似性）"""
    if original.shape != compressed.shape:
        orig_h, orig_w = original.shape[:2]
        comp_h, comp_w = compressed.shape[:2]
        start_h = (orig_h - comp_h) // 2
        start_w = (orig_w - comp_w) // 2
        original = original[start_h:start_h + comp_h, start_w:start_w + comp_w, :]

    return ssim(original, compressed, data_range=255, channel_axis=2, multichannel=True)


# =============================================================================
# 2. 显著性图生成函数
# =============================================================================

# MobileNetV2 模型加载缓存
_mobilenet = None


def _get_mobilenet():
    """惰性加载 MobileNetV2 模型"""
    global _mobilenet
    if _mobilenet is None:
        try:
            print("正在加载 MobileNetV2 模型...")
            _mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        except Exception as e:
            print(f"警告：无法加载 MobileNetV2 模型，错误: {e}. 请检查网络连接和torchvision版本。", file=sys.stderr)
            sys.exit(1)
    return _mobilenet


def get_cnn_saliency(img):
    """CNN 特征图显著性（基于 MobileNetV2 浅层特征图的最大激活）"""
    model = _get_mobilenet()
    tensor = PREPROCESS(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)

    # 注册钩子以获取特征图
    feature_map = None

    def hook_fn(module, input, output):
        nonlocal feature_map
        # 使用 features[2] 的输出 (Block 2, 24 channels, size H/4 x W/4)
        feature_map = output

    # 挂载到 MobileNetV2 的 features[2] 上
    hook = model.features[2].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(tensor)

    hook.remove()

    if feature_map is not None:
        # 取所有通道的最大激活
        saliency_map = feature_map.max(dim=1)[0].squeeze().cpu().numpy()
    else:
        # 失败时返回全零
        saliency_map = np.zeros((img.shape[0] // 4, img.shape[1] // 4), dtype=np.float32)

    # 归一化并缩放到原图尺寸
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return saliency_map


def get_cam_saliency(img):
    """CAM 激活图显著性（简化版：基于最终特征图的平均激活）"""
    model = _get_mobilenet()
    tensor = PREPROCESS(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)

    # 注册钩子获取最终特征图（features[-1]）
    feature_map = None

    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output

    # 挂载到 MobileNetV2 的 features[-1] 上
    hook = model.features[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(tensor)

    hook.remove()

    if feature_map is not None:
        # 简化版 CAM: 对所有通道求平均作为激活图
        saliency_map = feature_map.mean(dim=1).squeeze().cpu().numpy()
    else:
        saliency_map = np.zeros((img.shape[0] // 32, img.shape[1] // 32), dtype=np.float32)

    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return saliency_map


def get_traditional_saliency(img):
    """
    【优化点 1】传统显著性图：基于LAB空间的颜色对比度（CC）和中心先验（CP）的融合，
    比简单的局部对比度更鲁棒，专注于前景物体识别。
    """
    H, W, _ = img.shape
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 1. 颜色对比度 (CC) - 像素与其全局平均颜色的欧氏距离
    avg_L, avg_A, avg_B = lab[:, :, 0].mean(), lab[:, :, 1].mean(), lab[:, :, 2].mean()
    cc_sal = np.sqrt(
        (lab[:, :, 0] - avg_L) ** 2 +
        (lab[:, :, 1] - avg_A) ** 2 +
        (lab[:, :, 2] - avg_B) ** 2
    )

    # 2. 中心先验 (CP) - 图像中心区域权重更高
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    sigma = W / 4.0
    cp_sal = np.exp(-(dist ** 2 / (2 * sigma ** 2)))

    # 3. 融合：CC (70%) + CP (30%)
    fused_trad_sal = 0.7 * cc_sal + 0.3 * cp_sal

    # 归一化到 [0, 1]
    fused_trad_sal = (fused_trad_sal - fused_trad_sal.min()) / (fused_trad_sal.max() - fused_trad_sal.min() + 1e-8)
    return fused_trad_sal


def get_edge_map(img):
    """边缘检测图 (使用 Canny 算子)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_map = edges.astype(np.float32) / 255.0
    return edge_map


# =============================================================================
# 3. 显著性图融合与能量函数
# =============================================================================

def calculate_fused_saliency(img):
    """
    加权融合四种显著性图，并进行平滑处理
    权重比例：CNN 35%、CAM 25%、传统 25%、边缘 15%
    """
    print("  - 正在生成并融合多源显著性图...")

    # 1. 获取四个显著性图 (0.0 - 1.0 范围)
    cnn_sal = get_cnn_saliency(img)
    cam_sal = get_cam_saliency(img)
    trad_sal = get_traditional_saliency(img)
    edge_map = get_edge_map(img)

    # 2. 权重融合
    W_CNN = 0.35
    W_CAM = 0.25
    W_TRAD = 0.25
    W_EDGE = 0.15

    fused_saliency = (
            W_CNN * cnn_sal +
            W_CAM * cam_sal +
            W_TRAD * trad_sal +
            W_EDGE * edge_map
    )

    # 3. 归一化并平滑（平滑有助于动态规划找到更平滑、不易断裂的缝）
    fused_saliency = (fused_saliency - fused_saliency.min()) / (fused_saliency.max() - fused_saliency.min() + 1e-8)
    fused_saliency = cv2.GaussianBlur(fused_saliency, (5, 5), 0)

    return fused_saliency


def calculate_adaptive_energy(img, fused_saliency, saliency_scale=1.5, boundary_penalty_factor=10.0):
    """
    【优化点 2 & 3】计算自适应能量图 (Adaptive Energy Map)。
    E = G + non_saliency_cost + P
    目标是让 Seam 避开 高梯度 (G) 和 高显著性 (S) 的区域。
    """
    H, W = img.shape[:2]

    # 1. 梯度能量 (G)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_energy = np.abs(grad_x) + np.abs(grad_y)

    # 归一化梯度能量到 [0, 1]
    grad_energy = (grad_energy - grad_energy.min()) / (grad_energy.max() - grad_energy.min() + 1e-8)

    # 2. 显著性惩罚项 (non_saliency_cost)
    # Saliency (S) 高代表重要，所以 (1 - S) 低代表重要。
    # 能量E需要在S低时低，所以我们使用 (1 - S) 作为惩罚项，并期望它在低S处（不重要）高，
    # 但由于能量 E 应该在 Seam 经过时低，我们保持 E = G + (1-S) 的形式是错误的。
    # 正确的逻辑是：E = G + S_cost。S_cost在高S时应该高。

    # 正确的能量公式 (遵循 E = 梯度 + 显著性，但要用归一化后的项)：
    # E = G + S * saliency_scale  <-- 显著性越高，能量越高，越不能移除

    saliency_cost = saliency_scale * fused_saliency
    total_energy = grad_energy + saliency_cost

    # 3. 边界惩罚 (P) - 强化边界保护
    # 惩罚值设置为当前图像最大能量的倍数，确保 Seam 不在边缘穿行
    boundary_penalty = np.zeros_like(total_energy)
    fixed_penalty = total_energy.max() * boundary_penalty_factor
    boundary_penalty[:, 0] = fixed_penalty
    boundary_penalty[:, -1] = fixed_penalty

    final_energy = total_energy + boundary_penalty

    return final_energy


def find_optimal_seam(energy_map):
    """
    使用动态规划寻找最小累积能量缝 (NumPy 向量化优化版)
    """
    H, W = energy_map.shape
    M = energy_map.copy()  # 累积能量图

    for i in range(1, H):
        # 使用 np.roll 实现左右邻居的快速查找
        M_L = np.roll(M[i - 1], 1)
        M_M = M[i - 1]
        M_R = np.roll(M[i - 1], -1)

        # 边界处理：左边界只考虑中间和右边，右边界只考虑中间和左边
        # 左边界 (j=0): M_L[0] 应该等于 M_M[0]
        # 右边界 (j=W-1): M_R[W-1] 应该等于 M_M[W-1]
        # 使用 np.minimum.reduce 找到三者中的最小值
        min_prev = np.minimum.reduce([M_L, M_M, M_R])

        # 修复边界，确保左边界不使用 M_L[0] 的 roll 值，右边界不使用 M_R[W-1] 的 roll 值
        # 实际上在 np.minimum.reduce 中，inf 的处理交给 if/else 更精确。

        # 简化版：使用 np.argmin 的方式来避免复杂的边界向量化，确保精度
        for j in range(W):
            if j == 0:
                min_prev_val = min(M[i - 1, j], M[i - 1, j + 1])
            elif j == W - 1:
                min_prev_val = min(M[i - 1, j - 1], M[i - 1, j])
            else:
                min_prev_val = min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1])
            M[i, j] += min_prev_val

    # 回溯找缝
    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(H - 2, -1, -1):
        j = seam[i + 1]

        # 找到上层三个相邻像素中具有最小累积能量的索引

        # 边界处理：左边界只考虑右边和中间，右边界只考虑左边和中间
        if j == 0:
            min_idx = np.argmin(M[i, j:j + 2])  # 考虑 j, j+1
            seam[i] = j + min_idx
        elif j == W - 1:
            min_idx = np.argmin(M[i, j - 1:j + 1])  # 考虑 j-1, j
            seam[i] = j - 1 + min_idx
        else:
            min_idx = np.argmin(M[i, j - 1:j + 2])  # 考虑 j-1, j, j+1
            seam[i] = j - 1 + min_idx

    return seam


def remove_seam(img, seam):
    """从图像中移除给定的缝"""
    H, W, C = img.shape
    # 使用 np.delete 进行快速移除
    new_img = np.zeros((H, W - 1, C), dtype=img.dtype)
    for i in range(H):
        j = seam[i]
        new_img[i] = np.delete(img[i], j, axis=0)

    return new_img


def seam_carving_removal(img, num_seams, saliency_scale=1.5, boundary_penalty_factor=10.0):
    """
    核心 Seam Carving 移除算法
    """
    carved_img = img.copy()

    for i in range(num_seams):
        if i % 50 == 0:
            print(f"  - 正在移除第 {i + 1}/{num_seams} 条缝...")

        # 1. 融合显著性图
        fused_saliency = calculate_fused_saliency(carved_img)

        # 2. 计算自适应能量图
        energy_map = calculate_adaptive_energy(
            img=carved_img,
            fused_saliency=fused_saliency,
            saliency_scale=saliency_scale,
            boundary_penalty_factor=boundary_penalty_factor
        )

        # 3. 找到最优缝
        seam = find_optimal_seam(energy_map)

        # 4. 移除缝
        carved_img = remove_seam(carved_img, seam)

    return carved_img


# =============================================================================
# 4. 批量处理与可视化
# =============================================================================

def save_results(img, resized_seam, resized_bilinear, psnr_seam, ssim_seam, psnr_bilinear, ssim_bilinear, save_path):
    """保存可视化结果"""
    os.makedirs(save_path, exist_ok=True)

    # 1. 保存图像文件
    cv2.imwrite(os.path.join(save_path, "0_original.png"), img)
    cv2.imwrite(os.path.join(save_path, "1_seam_carving_optimized.png"), resized_seam)
    cv2.imwrite(os.path.join(save_path, "2_bilinear.png"), resized_bilinear)

    # 2. 生成 Matplotlib 可视化图表
    fig = plt.figure(figsize=(18, 6))

    # 原始图像
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'原始图像 (W={img.shape[1]}, H={img.shape[0]})')
    ax1.axis('off')

    # Seam Carving 结果
    ax2 = fig.add_subplot(132)
    ax2.imshow(cv2.cvtColor(resized_seam, cv2.COLOR_BGR2RGB))
    title_seam = f'SC 优化结果 (PSNR: {psnr_seam:.2f}, SSIM: {ssim_seam:.4f})'
    ax2.set_title(title_seam)
    ax2.axis('off')

    # 双线性插值结果
    ax3 = fig.add_subplot(133)
    ax3.imshow(cv2.cvtColor(resized_bilinear, cv2.COLOR_BGR2RGB))
    title_bilinear = f'双线性插值 (PSNR: {psnr_bilinear:.2f}, SSIM: {ssim_bilinear:.4f})'
    ax3.set_title(title_bilinear)
    ax3.axis('off')

    plt.suptitle("图像重定向对比（自适应能量 SC vs. 双线性插值）")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "comparison_chart.png"))
    plt.close(fig)


def test_batch_seam_carving(data_dir="dataset", target_ratio=0.5, sample_size=50):
    """
    批量处理数据集中的图像，执行重定向并计算平均评估指标。
    """
    print(f"开始批量处理，缩放目标比例: {target_ratio:.2f}...")

    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录 '{data_dir}' 不存在。请创建该目录并放入图像文件。", file=sys.stderr)
        return

    dataset = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    if not dataset:
        print(f"错误: 目录 '{data_dir}' 中未找到任何 jpg/png 图像。", file=sys.stderr)
        return

    dataset = dataset[:sample_size]
    print(f"找到 {len(dataset)} 张图像进行处理。")

    # 定义优化参数
    SALIENCY_SCALE = 1.5
    BOUNDARY_PENALTY_FACTOR = 10.0  # 极大的边界惩罚

    all_psnr_seam, all_ssim_seam = [], []
    all_psnr_bilinear, all_ssim_bilinear = [], []

    # 批量处理每张图像
    for idx, img_path in enumerate(dataset, 1):
        print(f"\n--- 正在处理图像 {idx}/{len(dataset)}: {os.path.basename(img_path)} ---")

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}，跳过。")
            continue

        original_width = img.shape[1]
        target_width = int(original_width * target_ratio)
        num_seams = original_width - target_width

        # 1. Seam Carving 优化重定向
        print(f"开始 SC 重定向至宽度 {target_width}...")
        resized_seam = seam_carving_removal(
            img=img,
            num_seams=num_seams,
            saliency_scale=SALIENCY_SCALE,
            boundary_penalty_factor=BOUNDARY_PENALTY_FACTOR
        )

        # 2. 双线性插值重定向（作为对比基线）
        resized_bilinear = cv2.resize(img, (target_width, img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 3. 评估（将结果恢复到原尺寸后评估）
        print("  - 计算质量评估指标...")

        # 为了 PSNR/SSIM 评估，将结果图像中心对齐地填充回原宽度，以便与原图对比
        seam_h, seam_w, _ = resized_seam.shape
        bilinear_h, bilinear_w, _ = resized_bilinear.shape

        # SC 结果填充
        resized_seam_restored = cv2.copyMakeBorder(
            resized_seam, 0, 0, (original_width - seam_w) // 2,
                                original_width - seam_w - (original_width - seam_w) // 2,
            cv2.BORDER_CONSTANT, value=[128, 128, 128]
        )
        # 双线性结果填充
        resized_bilinear_restored = cv2.copyMakeBorder(
            resized_bilinear, 0, 0, (original_width - bilinear_w) // 2,
                                    original_width - bilinear_w - (original_width - bilinear_w) // 2,
            cv2.BORDER_CONSTANT, value=[128, 128, 128]
        )

        psnr_seam = calculate_psnr(img, resized_seam_restored)
        psnr_bilinear = calculate_psnr(img, resized_bilinear_restored)
        ssim_seam = calculate_ssim(img, resized_seam_restored)
        ssim_bilinear = calculate_ssim(img, resized_bilinear_restored)

        all_psnr_seam.append(psnr_seam)
        all_ssim_seam.append(ssim_seam)
        all_psnr_bilinear.append(psnr_bilinear)
        all_ssim_bilinear.append(ssim_bilinear)

        print(
            f"  - 指标: SC(PSNR:{psnr_seam:.2f}dB, SSIM:{ssim_seam:.4f}) | Bilinear(PSNR:{psnr_bilinear:.2f}dB, SSIM:{ssim_bilinear:.4f})")

        # 4. 保存结果
        save_results(
            img=img,
            resized_seam=resized_seam,
            resized_bilinear=resized_bilinear,
            psnr_seam=psnr_seam,
            ssim_seam=ssim_seam,
            psnr_bilinear=psnr_bilinear,
            ssim_bilinear=ssim_bilinear,
            save_path=f"results/img_{idx}/"
        )

    # 5. 计算平均指标与总结
    if all_psnr_seam:
        avg_psnr_seam = np.mean(all_psnr_seam)
        avg_ssim_seam = np.mean(all_ssim_seam)
        avg_psnr_bilinear = np.mean(all_psnr_bilinear)
        avg_ssim_bilinear = np.mean(all_ssim_bilinear)

        print("\n" + "=" * 60)
        print("批量处理完成，平均评估指标：")
        print(f"自适应SC - 平均PSNR：{avg_psnr_seam:.2f} dB，平均SSIM：{avg_ssim_seam:.4f}")
        print(f"双线性插值 - 平均PSNR：{avg_psnr_bilinear:.2f} dB，平均SSIM：{avg_ssim_bilinear:.4f}")
        print("=" * 60)
    else:
        print("警告：未找到任何图像进行处理和评估。")


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")

    test_batch_seam_carving(
        data_dir="test_data",
        target_ratio=0.5,
        sample_size=50  # 限制处理图像数量为 50 张
    )
    print("\n程序运行完毕，请查看 results 目录下的输出文件。")
