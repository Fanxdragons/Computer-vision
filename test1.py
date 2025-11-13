import numpy as np
import cv2
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import convolve, gaussian_filter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
# 解决 matplotlib 中文乱码和负号显示问题
try:
    # 尝试设置常用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    # 解决保存图像时负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    # 如果系统没有找到这些字体，使用默认设置
    pass

# ... (保持原有代码不变)

import os  # 补充os模块，用于结果保存与数据集加载

# 全局设备配置（GPU优先，加速CNN显著性计算，参考文档1-31"有限计算资源"优化）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {DEVICE}")


# 1. 质量评估函数（严格遵循文档1-18至1-22实现）
def calculate_psnr(original, compressed):
    """计算PSNR（峰值信噪比），参考文档1-19实现逻辑"""
    if original.shape != compressed.shape:
        orig_h, orig_w = original.shape[:2]
        comp_h, comp_w = compressed.shape[:2]
        start_h = (orig_h - comp_h) // 2
        start_w = (orig_w - comp_w) // 2
        original = original[start_h:start_h + comp_h, start_w:start_w + comp_w, :]

    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0  # 8位图像像素最大值，与文档1-19一致
    psnr = 10 * math.log10(max_pixel ** 2 / mse)
    return psnr


def calculate_ssim(original, compressed):
    """计算SSIM（结构相似性指数），参考文档1-21实现逻辑"""
    if original.shape != compressed.shape:
        orig_h, orig_w = original.shape[:2]
        comp_h, comp_w = compressed.shape[:2]
        start_h = (orig_h - comp_h) // 2
        start_w = (orig_w - comp_w) // 2
        original = original[start_h:start_h + comp_h, start_w:start_w + comp_w, :]

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    # 匹配文档1-21中"数据范围255"与"多通道处理"逻辑
    return ssim(original_gray, compressed_gray, data_range=255)


# 2. 传统显著性图生成（保留文档1-4至1-10三种传统算法，不删除）
def generate_traditional_saliency(image):
    """生成传统显著性图（融合颜色对比、中心先验、谱残差），参考文档1-4至1-10实现"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2.1 颜色对比显著性（文档1-4至1-5实现）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_lab = np.mean(lab, axis=(0, 1))
    color_sal = np.sqrt(np.sum((lab - mean_lab) ** 2, axis=2))
    color_sal = (color_sal - color_sal.min()) / (color_sal.max() - color_sal.min())

    # 2.2 中心先验显著性（文档1-6至1-7实现）
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    center_sal = np.exp(-(dist_from_center ** 2) / (2 * (max_dist / 2) ** 2))

    # 2.3 谱残差显著性（文档1-8至1-10实现）
    gray_float = gray.astype(np.float32) / 255.0
    fft = np.fft.fft2(gray_float)
    log_amplitude = np.log(np.abs(fft) + 1e-7)
    phase = np.angle(fft)
    spectral_residual = log_amplitude - gaussian_filter(log_amplitude, sigma=3)
    combined = np.exp(spectral_residual + 1j * phase)
    spectral_sal = np.abs(np.fft.ifft2(combined))
    spectral_sal = gaussian_filter(spectral_sal ** 2, sigma=8)
    spectral_sal = (spectral_sal - spectral_sal.min()) / (spectral_sal.max() - spectral_sal.min())

    # 融合三种传统显著性图
    traditional_sal = (0.3 * color_sal + 0.3 * center_sal + 0.4 * spectral_sal) * 255
    return traditional_sal.astype(np.uint8)


# 3. CNN+CAM显著性图生成（参考文档1-30、1-40深度学习增强思路）
def generate_cnn_cam_saliency(image):
    """生成CNN显著性图与CAM图，参考文档1-40"CNN特征图+CAM激活图"逻辑"""
    # 加载轻量级预训练模型（符合文档1-31"有限计算资源"要求）
    model = models.mobilenet_v2(pretrained=True).to(DEVICE)
    model.eval()

    # 提取模型关键层（文档1-40特征提取思路）
    feature_layer = model.features[-1]  # 中层特征层，平衡语义与计算量
    fc_weight = model.classifier[1].weight.data.cpu().numpy()  # 分类层权重用于CAM

    # 图像预处理（适配模型输入）
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # 输入迁移到GPU

    # 钩子函数捕获中间层特征（文档1-40特征获取逻辑）
    feature_output = None

    def hook_fn(module, input, output):
        nonlocal feature_output
        feature_output = output

    hook = feature_layer.register_forward_hook(hook_fn)
    with torch.no_grad():  # 禁用梯度，减少GPU内存占用
        model(input_tensor)
    hook.remove()

    if feature_output is None:
        raise ValueError("未获取到CNN中间层特征，参考文档1-40检查钩子函数")

    # 特征处理（GPU→CPU，仅最终转换减少数据交互）
    feature_map = feature_output.detach().cpu().numpy()

    # 生成CNN显著性图（文档1-40"CNN特征图"逻辑）
    cnn_sal = np.var(feature_map[0], axis=0)  # 通道方差突出显著区域
    cnn_sal = cv2.resize(cnn_sal, (image.shape[1], image.shape[0]))
    cnn_sal = gaussian_filter(cnn_sal, sigma=0.8)
    cnn_sal = (cnn_sal - cnn_sal.min()) / (cnn_sal.max() - cnn_sal.min()) * 255

    # 生成CAM图（文档1-40"CAM激活图"逻辑）
    feature_avg = np.mean(feature_map[0], axis=(1, 2))  # 通道平均池化
    cam = np.dot(feature_map[0].transpose(1, 2, 0), feature_avg)  # 特征×权重
    cam = np.maximum(cam, 0)  # ReLU过滤负响应
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = gaussian_filter(cam, sigma=1.2)
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255

    return cnn_sal.astype(np.uint8), cam.astype(np.uint8)


# 补充1：边缘检测图生成（文档1-40"边缘检测图"组件）
def generate_edge_saliency(image):
    """生成边缘检测显著性图，参考文档1-40与1-13梯度逻辑"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_sal = np.sqrt(dx ** 2 + dy ** 2)
    edge_sal = (edge_sal - edge_sal.min()) / (edge_sal.max() - edge_sal.min()) * 255
    return edge_sal.astype(np.uint8)


# 补充3：动态权重计算函数（创新性优化：根据图像特性调整融合权重）
def calculate_dynamic_weights(image, cnn_sal, cam_sal, traditional_sal, edge_sal):
    """
    根据图像的边缘复杂性和CNN显著性图的稀疏性，动态计算融合权重。

    参数:
        image: 原始图像 (np.array)
        cnn_sal, cam_sal, traditional_sal, edge_sal: 四种显著性图 (np.uint8)

    返回:
        (w_cnn, w_cam, w_trad, w_edge)
    """

    # --- 1. 边缘复杂性指标 (Alpha): 衡量图像中边缘信息的丰富程度 ---
    # 使用Sobel梯度的平均值，反映图像细节程度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.abs(dx) + np.abs(dy)

    # 将梯度均值归一化，范围限制在 [0.1, 0.9]
    alpha = np.mean(gradient_magnitude)
    alpha = np.clip(alpha / 100, 0.1, 0.9)

    # --- 2. 显著性稀疏性指标 (Beta): 衡量CNN显著性区域的集中程度 ---
    # 熵越低，显著性越集中（高 beta）；熵越高，显著性越分散（低 beta）
    cnn_norm = cnn_sal / 255.0
    hist, _ = np.histogram(cnn_norm.ravel(), bins=256, range=(0, 1), density=True)
    hist[hist == 0] = 1e-8
    entropy = -np.sum(hist * np.log2(hist))

    # 熵值反转并归一化到 [0.1, 0.9]
    max_entropy = np.log2(256)
    beta = 1.0 - (entropy / max_entropy)
    beta = np.clip(beta, 0.1, 0.9)

    # --- 3. 动态权重计算与分配（基于作业参考权重 [0.35, 0.25, 0.25, 0.15] 进行调整） ---

    # Edge权重：与 alpha 正相关 (边缘越复杂，权重越高)
    w_edge = 0.15 * (0.5 + alpha / 2)

    # CNN/CAM权重：与 beta 正相关 (显著性越集中，权重越高)
    w_cnn_cam_factor = (0.35 + 0.25) * (0.5 + beta / 2)

    # Traditional权重：与 beta 负相关 (显著性越集中，传统方法权重越低)
    w_trad = 0.25 * (1.5 - beta)

    # 根据原比例分配 CNN/CAM 权重
    w_cnn = w_cnn_cam_factor * (0.35 / (0.35 + 0.25))
    w_cam = w_cnn_cam_factor * (0.25 / (0.35 + 0.25))

    # 归一化所有权重，确保总和为 1
    weights = np.array([w_cnn, w_cam, w_trad, w_edge])
    total_weight = np.sum(weights)

    w_cnn, w_cam, w_trad, w_edge = weights / total_weight

    # 打印动态权重（用于报告分析）
    print(f"动态权重: CNN={w_cnn:.3f}, CAM={w_cam:.3f}, 传统={w_trad:.3f}, 边缘={w_edge:.3f}")

    return w_cnn, w_cam, w_trad, w_edge
# 4. 多源显著性融合（保留传统成分，参考文档1-40加权融合思路）
# 4. 多源显著性融合（融合CNN、CAM、传统显著性、边缘检测，**使用动态权重**）
def generate_fused_saliency(image):
    """融合CNN、CAM、传统显著性、边缘检测，**使用动态权重**"""

    # 4.1 传统显著性图
    traditional_sal = generate_traditional_saliency(image)
    # 4.2 CNN+CAM显著性图
    cnn_sal, cam_sal = generate_cnn_cam_saliency(image)
    # 4.3 边缘检测图
    edge_sal = generate_edge_saliency(image)

    # **核心修改：调用动态权重计算**
    w_cnn, w_cam, w_trad, w_edge = calculate_dynamic_weights(
        image, cnn_sal, cam_sal, traditional_sal, edge_sal
    )

    # 应用动态加权融合
    fused_sal = (
            w_cnn * cnn_sal +
            w_cam * cam_sal +
            w_trad * traditional_sal +
            w_edge * edge_sal
    )

    fused_sal = np.clip(fused_sal, 0, 255)
    fused_sal = cv2.GaussianBlur(fused_sal, (5, 5), 0)
    return fused_sal.astype(np.uint8)


# 5. Seam Carving类（严格遵循文档1-11至1-15核心逻辑）
class SeamCarving:
    """Seam Carving图像重定向算法，参考文档1-12完整实现"""

    def __init__(self, image, saliency_map=None):
        self.original_image = image.copy()  # 原始图像备份（文档1-12要求）
        self.image = image.copy()  # 当前处理图像
        self.saliency_map = saliency_map  # 多源融合显著性图

    def calculate_energy(self, use_saliency=True):
        """计算能量图（梯度+显著性融合），参考文档1-13实现"""
        # 基础能量：Sobel梯度（文档1-13核心逻辑）
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(dx) + np.abs(dy)

        # 融合多源显著性图（文档1-13"显著区域能量更高"逻辑）
        if use_saliency and self.saliency_map is not None:
            resized_saliency = cv2.resize(self.saliency_map, (self.image.shape[1], self.image.shape[0]))
            saliency_weight = 700  # 权重系数与文档1-13一致
            energy = energy + saliency_weight * (resized_saliency / 255.0)

        return energy

    def find_vertical_seam(self, energy):
        """动态规划找最小能量垂直缝，使用NumPy向量化优化速度"""
        h, w = energy.shape
        dp = energy.copy()

        # 计算累积最小能量：使用向量化操作代替内层循环
        for i in range(1, h):
            # C_L: 累积到左侧的能量，C_M: 累积到中间的能量, C_R: 累积到右侧的能量
            # C_L 对应上一行 [:, 0] + [:, 0:w-1]
            C_L = np.roll(dp[i - 1], 1)
            C_L[0] = dp[i - 1, 0]  # 左边界特殊处理

            C_M = dp[i - 1]

            # C_R 对应上一行 [:, w-1] + [:, 1:w]
            C_R = np.roll(dp[i - 1], -1)
            C_R[-1] = dp[i - 1, -1]  # 右边界特殊处理

            # 找到最小累积能量
            min_prev = np.minimum.reduce([C_L, C_M, C_R])

            dp[i, :] += min_prev

        # 回溯找缝（回溯部分难以向量化，保持原样，但整体速度已提升）
        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1])
        for i in range(h - 2, -1, -1):
            j = seam[i + 1]

            # 找到上一行最小能量的索引
            if j == 0:
                min_idx = np.argmin(dp[i, j:j + 2])
                seam[i] = j + min_idx
            elif j == w - 1:
                min_idx = np.argmin(dp[i, j - 1:j + 1])
                seam[i] = j - 1 + min_idx
            else:
                min_idx = np.argmin(dp[i, j - 1:j + 2])
                seam[i] = j - 1 + min_idx

        return seam

    def find_horizontal_seam(self, energy):
        """找最小能量水平缝，参考文档1-14"转置复用垂直缝算法"逻辑"""
        energy_T = energy.T
        seam = self.find_vertical_seam(energy_T)
        return seam

    def remove_vertical_seam(self, seam):
        """移除垂直缝（含显著性图同步更新），参考文档1-12实现"""
        h, w, c = self.image.shape
        output = np.zeros((h, w - 1, c), dtype=self.image.dtype)

        # 移除图像垂直缝
        for i in range(h):
            j = seam[i]
            output[i, :, :] = np.delete(self.image[i, :, :], j, axis=0)
        self.image = output

        # 同步更新融合显著性图（文档1-12显著性图适配逻辑）
        if self.saliency_map is not None:
            h_s, w_s = self.saliency_map.shape
            output_saliency = np.zeros((h_s, w_s - 1), dtype=self.saliency_map.dtype)
            seam_scaled = (seam * w_s / w).astype(int)  # 缝尺寸适配

            for i in range(h_s):
                j = seam_scaled[min(i, len(seam_scaled) - 1)]
                output_saliency[i, :] = np.delete(self.saliency_map[i, :], j, axis=0)
            self.saliency_map = output_saliency

    def remove_horizontal_seam(self, seam):
        """移除水平缝，参考文档1-12"转置复用垂直缝移除逻辑"实现"""
        orig_shape = self.image.shape
        self.image = np.transpose(self.image, (1, 0, 2))
        self.remove_vertical_seam(seam)
        self.image = np.transpose(self.image, (1, 0, 2))

        # 同步更新融合显著性图
        if self.saliency_map is not None:
            h_s, w_s = self.saliency_map.shape
            self.saliency_map = self.saliency_map.T
            output_saliency = np.zeros((w_s - 1, h_s), dtype=self.saliency_map.dtype)
            seam_scaled = (seam * h_s / orig_shape[0]).astype(int)

            for i in range(w_s):
                j = seam_scaled[min(i, len(seam_scaled) - 1)]
                output_saliency[i, :] = np.delete(self.saliency_map[i, :], j, axis=0)
            self.saliency_map = output_saliency.T

    def resize(self, new_width=None, new_height=None, use_saliency=True):
        """调整图像大小，参考文档1-15实现逻辑"""
        # 缩小宽度（移除垂直缝）
        if new_width is not None:
            delta_w = self.image.shape[1] - new_width
            if delta_w > 0:
                for _ in range(delta_w):
                    energy = self.calculate_energy(use_saliency)
                    seam = self.find_vertical_seam(energy)
                    self.remove_vertical_seam(seam)
                    if _ % 10 == 0:
                        print(f"已移除 {_ + 1}/{delta_w} 条垂直缝")

        # 缩小高度（移除水平缝）
        if new_height is not None:
            delta_h = self.image.shape[0] - new_height
            if delta_h > 0:
                for _ in range(delta_h):
                    energy = self.calculate_energy(use_saliency)
                    seam = self.find_horizontal_seam(energy)
                    self.remove_horizontal_seam(seam)
                    if _ % 10 == 0:
                        print(f"已移除 {_ + 1}/{delta_h} 条水平缝")

        return self.image


# 补充2：结果保存函数（文档2.1-2.2"质量评估数据记录"要求）
def save_results(img, resized_seam, resized_bilinear, psnr_seam, ssim_seam, psnr_bilinear, ssim_bilinear, save_path):
    os.makedirs(save_path, exist_ok=True)
    # 保存图像
    cv2.imwrite(os.path.join(save_path, "original_image.jpg"), img)
    cv2.imwrite(os.path.join(save_path, "seam_carving_result.jpg"), resized_seam)
    cv2.imwrite(os.path.join(save_path, "bilinear_result.jpg"), resized_bilinear)
    # 保存指标
    with open(os.path.join(save_path, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"实验配置（基于文档cv-exp-4）\n")
        f.write(f"原始图像尺寸：{img.shape[1]}×{img.shape[0]}（宽×高）\n")
        f.write(f"缩放比例：{resized_seam.shape[1]/img.shape[1]:.2%}（文档要求40%-60%）\n")
        f.write(f"重定向后尺寸（Seam Carving）：{resized_seam.shape[1]}×{resized_seam.shape[0]}\n")
        f.write(f"重定向后尺寸（双线性插值）：{resized_bilinear.shape[1]}×{resized_bilinear.shape[0]}\n\n")
        f.write(f"质量评估指标（文档2.1-2.2）\n")
        f.write(f"1. PSNR（峰值信噪比）\n")
        f.write(f"   - CNN增强Seam Carving：{psnr_seam:.2f} dB\n")
        f.write(f"   - 双线性插值：{psnr_bilinear:.2f} dB\n\n")
        f.write(f"2. SSIM（结构相似性指数）\n")
        f.write(f"   - CNN增强Seam Carving：{ssim_seam:.4f}\n")
        f.write(f"   - 双线性插值：{ssim_bilinear:.4f}\n")


# 5. 评估指标可视化（新增：对比PSNR和SSIM的柱状图）
def plot_metrics_comparison(avg_psnr_seam, avg_ssim_seam, avg_psnr_bilinear, avg_ssim_bilinear):
    """
    绘制PSNR和SSIM的平均指标对比柱状图。
    """
    labels = ['CNN增强Seam Carving', '双线性插值']

    # --- PSNR 对比图 ---
    psnr_values = [avg_psnr_seam, avg_psnr_bilinear]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, psnr_values, color=['darkred', 'darkblue'])
    plt.ylabel('平均 PSNR (dB)')
    plt.title('平均 PSNR 对比')

    # 在柱子上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f} dB', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--')
    plt.savefig('results/avg_psnr_comparison.png')
    plt.close()

    print("已生成 PSNR 对比图: results/avg_psnr_comparison.png")

    # --- SSIM 对比图 ---
    ssim_values = [avg_ssim_seam, avg_ssim_bilinear]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, ssim_values, color=['darkred', 'darkblue'])
    plt.ylabel('平均 SSIM')
    plt.title('平均 SSIM 对比')

    # 在柱子上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--')
    plt.ylim(0, 1.0)  # SSIM 范围通常在 0 到 1
    plt.savefig('results/avg_ssim_comparison.png')
    plt.close()

    print("已生成 SSIM 对比图: results/avg_ssim_comparison.png")

# 6. 批量测试函数（符合文档1-35、1-40要求）
def test_batch_seam_carving(sample_size=50):
    """
    批量测试CNN增强显著性引导的Seam Carving
    符合文档要求：50张多样化图像、40%-60%缩放、PSNR/SSIM评估
    """

    # 1. 加载数据集（文档要求：不少于50张，场景多样化）
    def load_images(data_dir):
        """加载数据集，返回图像路径列表（符合文档数据集要求）"""
        img_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tif']
        img_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    img_paths.append(os.path.join(root, file))
        # 小批量测试不强制要求50张，但需保证有足够样本
        assert len(img_paths) >= sample_size, f"数据集图像数量不足{sample_size}张（当前{len(img_paths)}张）"
        return img_paths[:sample_size]  # 只返回前N张图像路径

        # 加载测试图像（通过sample_size控制数量）

    dataset = load_images("test_data/")
    print(f"成功加载 {len(dataset)} 张测试图像（小批量测试），开始处理...")

    # 存储所有图像的评估指标（用于计算平均值，符合文档2.1-2.2要求）
    all_psnr_seam = []
    all_ssim_seam = []
    all_psnr_bilinear = []
    all_ssim_bilinear = []

    # 2. 批量处理每张图像（核心逻辑，严格遵循文档算法流程）
    for idx, img_path in enumerate(dataset, 1):
        print(f"\n处理第 {idx}/{len(dataset)} 张图像：{os.path.basename(img_path)}")

        # 2.1 读取图像（文档1-179读取逻辑）
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的图像：{img_path}")
            continue
        orig_h, orig_w = img.shape[:2]
        print(f"原始图像尺寸：{orig_w}×{orig_h}")

        # 2.2 生成多源显著性图并**动态**融合
        print("生成并动态融合多源显著性图...")
        # 此时 generate_fused_saliency 内部已经完成了所有组件的生成和动态加权融合
        fused_sal = generate_fused_saliency(img)

        # 2.3 Seam Carving重定向（文档1-11至1-15核心流程，缩放至60%）
        print("执行Seam Carving重定向...")
        sc = SeamCarving(img, fused_sal)
        target_w = int(orig_w * 0.6)
        assert 0.4 <= target_w / orig_w <= 0.6, f"缩放比例{target_w / orig_w:.2f}超出文档要求的40%-60%"
        resized_seam = sc.resize(new_width=target_w)
        print(f"Seam Carving重定向后尺寸：{resized_seam.shape[1]}×{resized_seam.shape[0]}")

        # 2.4 双线性插值对比（文档要求的传统算法对比组，1-35）
        print("执行双线性插值缩放...")
        resized_bilinear = cv2.resize(img, (target_w, orig_h), interpolation=cv2.INTER_LINEAR)
        print(f"双线性插值后尺寸：{resized_bilinear.shape[1]}×{resized_bilinear.shape[0]}")

        # 2.5 评估指标计算（文档2.1-2.2要求：恢复至原始尺寸后计算）
        print("计算PSNR与SSIM指标...")
        resized_seam_restored = cv2.resize(resized_seam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        resized_bilinear_restored = cv2.resize(resized_bilinear, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        psnr_seam = calculate_psnr(img, resized_seam_restored)
        psnr_bilinear = calculate_psnr(img, resized_bilinear_restored)
        ssim_seam = calculate_ssim(img, resized_seam_restored)
        ssim_bilinear = calculate_ssim(img, resized_bilinear_restored)

        # 记录指标
        all_psnr_seam.append(psnr_seam)
        all_ssim_seam.append(ssim_seam)
        all_psnr_bilinear.append(psnr_bilinear)
        all_ssim_bilinear.append(ssim_bilinear)

        # 2.6 保存结果
        print("保存处理结果...")
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

    # 3. 计算平均指标（文档2.1-2.2要求）
    if all_psnr_seam:
        avg_psnr_seam = np.mean(all_psnr_seam)
        avg_ssim_seam = np.mean(all_ssim_seam)
        avg_psnr_bilinear = np.mean(all_psnr_bilinear)
        avg_ssim_bilinear = np.mean(all_ssim_bilinear)

        print("\n" + "=" * 60)
        print("批量处理完成，平均评估指标（基于文档2.1-2.2）：")
        print(f"CNN增强Seam Carving - 平均PSNR：{avg_psnr_seam:.2f} dB，平均SSIM：{avg_ssim_seam:.4f}")
        print(f"双线性插值 - 平均PSNR：{avg_psnr_bilinear:.2f} dB，平均SSIM：{avg_ssim_bilinear:.4f}")
        print("=" * 60)

        # **新增：调用绘图函数生成对比图**
        plot_metrics_comparison(
            avg_psnr_seam, avg_ssim_seam,
            avg_psnr_bilinear, avg_ssim_bilinear
        )
    else:
        print("未成功处理任何图像，无法计算平均指标")


if __name__ == "__main__":
    test_batch_seam_carving()