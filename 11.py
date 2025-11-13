import numpy as np
import cv2
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import convolve, gaussian_filter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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
    """优化版CNN+CAM显著性图生成，提升特征表达能力"""
    model = models.mobilenet_v2(pretrained=True).to(DEVICE)
    model.eval()

    # 优化点1：选择更深层特征（原代码使用features[6]，改为features[-1]获取高层语义）
    feature_layer = model.features[-1]  # 最后一层特征层，语义信息更丰富
    fc_weight = model.classifier[1].weight.data.cpu().numpy()

    # 优化点2：改进预处理，保留更多细节
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image.shape[0], image.shape[1]),  # 保持原尺寸
                          transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # 钩子函数捕获特征
    feature_output = None

    def hook_fn(module, input, output):
        nonlocal feature_output
        feature_output = output

    hook = feature_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_tensor)
    hook.remove()

    # 优化点3：多尺度特征融合（增加浅层特征）
    shallow_feature_layer = model.features[2]  # 浅层特征，保留细节
    shallow_output = None

    def shallow_hook_fn(module, input, output):
        nonlocal shallow_output
        shallow_output = output

    shallow_hook = shallow_feature_layer.register_forward_hook(shallow_hook_fn)
    with torch.no_grad():
        model(input_tensor)
    shallow_hook.remove()

    # 融合深浅层特征
    feature_map = feature_output.detach().cpu().numpy()
    shallow_map = shallow_output.detach().cpu().numpy()
    shallow_map = cv2.resize(shallow_map[0], (feature_map.shape[3], feature_map.shape[2]))
    combined_features = 0.7 * feature_map[0] + 0.3 * shallow_map  # 加权融合

    # 生成CNN显著性图（改进：使用通道最大值而非方差）
    cnn_sal = np.max(combined_features, axis=0)  # 最大值更能突出显著区域
    cnn_sal = cv2.resize(cnn_sal, (image.shape[1], image.shape[0]))
    cnn_sal = gaussian_filter(cnn_sal, sigma=1.0)  # 适度平滑
    cnn_sal = (cnn_sal - cnn_sal.min()) / (cnn_sal.max() - cnn_sal.min() + 1e-8) * 255

    # 优化点4：改进CAM计算（使用全局平均池化）
    gap = np.mean(feature_map[0], axis=(1, 2))  # 全局平均池化
    cam = np.dot(feature_map[0].transpose(1, 2, 0), gap)
    cam = np.maximum(cam, 0)  # ReLU激活
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = gaussian_filter(cam, sigma=1.5)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) * 255

    return cnn_sal.astype(np.uint8), cam.astype(np.uint8)


# 4. 多源显著性融合（保留传统成分，参考文档1-40加权融合思路）
def generate_fused_saliency(image):
    """融合CNN、CAM、传统显著性、边缘检测，参考文档1-40四组件融合逻辑"""
    # 4.1 传统显著性图（保留，不删除）
    traditional_sal = generate_traditional_saliency(image)
    # 4.2 CNN+CAM显著性图
    cnn_sal, cam_sal = generate_cnn_cam_saliency(image)
    # 4.3 边缘检测图（文档1-40"边缘检测图"组件）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_sal = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_sal = np.abs(edge_sal)
    edge_sal = cv2.resize(edge_sal, (image.shape[1], image.shape[0]))
    edge_sal = (edge_sal - edge_sal.min()) / (edge_sal.max() - edge_sal.min()) * 255
    edge_sal = edge_sal.astype(np.uint8)

    # 加权融合（权重参考文档1-40建议比例，保留传统成分）
    fused_sal = (0.35 * cnn_sal + 0.25 * cam_sal + 0.25 * traditional_sal + 0.15 * edge_sal)
    return fused_sal.astype(np.uint8)  # 仅返回融合结果，内部保留传统计算


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
            saliency_weight = 1000  # 权重系数与文档1-13一致
            energy = energy + saliency_weight * (resized_saliency / 255.0)

        return energy

    def find_vertical_seam(self, energy):
        """动态规划找最小能量垂直缝，参考文档1-12实现"""
        h, w = energy.shape
        dp = energy.copy()

        # 计算累积最小能量（文档1-12动态规划逻辑）
        for i in range(1, h):
            for j in range(w):
                if j == 0:
                    dp[i, j] += min(dp[i - 1, j], dp[i - 1, j + 1])
                elif j == w - 1:
                    dp[i, j] += min(dp[i - 1, j - 1], dp[i - 1, j])
                else:
                    dp[i, j] += min(dp[i - 1, j - 1], dp[i - 1, j], dp[i - 1, j + 1])

        # 回溯找缝（文档1-12回溯逻辑）
        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1])
        for i in range(h - 2, -1, -1):
            j = seam[i + 1]
            if j == 0:
                seam[i] = j + np.argmin(dp[i, j:j + 2])
            elif j == w - 1:
                seam[i] = j - 1 + np.argmin(dp[i, j - 1:j + 1])
            else:
                seam[i] = j - 1 + np.argmin(dp[i, j - 1:j + 2])

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


# 6. 测试代码（仅输出融合结果与双线性插值结果，不输出传统单独结果）
def test_fused_seam_carving():
    """测试多源融合显著性引导的Seam Carving，参考文档1-35"定量对比插值算法"要求"""
    # 读取测试图像（参考文档1-179测试图像路径风格）
    image_path = 'Xian.jpg'
    test_image = cv2.imread(image_path)
    if test_image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}，参考文档1-179检查路径")

    # 生成多源融合显著性图（保留传统成分，不删除）
    print("正在生成多源融合显著性图（含传统显著性成分）...")
    fused_sal = generate_fused_saliency(test_image)

    # 1. 融合显著性引导的Seam Carving（文档1-34核心要求）
    print("使用多源融合显著性图进行图像重定向...")
    sc_fused = SeamCarving(test_image, fused_sal)
    resized_fused = sc_fused.resize(new_width=400, use_saliency=True)  # 目标宽度参考文档1-197

    # 2. 双线性插值对比（文档1-35要求与插值算法对比）
    print("使用双线性插值进行图像缩放...")
    resized_bilinear = cv2.resize(test_image, (400, test_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 计算质量评估指标（文档1-35要求的PSNR、SSIM）
    print("计算PSNR与SSIM评估指标...")
    psnr_fused = calculate_psnr(test_image, resized_fused)
    psnr_bilinear = calculate_psnr(test_image, resized_bilinear)
    ssim_fused = calculate_ssim(test_image, resized_fused)
    ssim_bilinear = calculate_ssim(test_image, resized_bilinear)

    # 输出结果（仅输出融合与双线性插值，不输出传统单独结果）
    print("\n===== 图像重定向结果对比（仅融合vs双线性） =====")
    print(f"多源融合显著性Seam Carving: PSNR={psnr_fused:.2f} dB, SSIM={ssim_fused:.4f}")
    print(f"双线性插值: PSNR={psnr_bilinear:.2f} dB, SSIM={ssim_bilinear:.4f}")

    # 可视化结果（仅展示融合与双线性插值，不展示传统单独结果）
    fig = plt.figure(figsize=(15, 7))

    # 原始图像
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 融合显著性Seam Carving结果
    ax2 = fig.add_subplot(132)
    ax2.imshow(cv2.cvtColor(resized_fused, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Fused Saliency Seam Carving\nPSNR={psnr_fused:.2f}, SSIM={ssim_fused:.4f}')
    ax2.axis('off')

    # 双线性插值结果
    ax3 = fig.add_subplot(133)
    ax3.imshow(cv2.cvtColor(resized_bilinear, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'Bilinear Interpolation\nPSNR={psnr_bilinear:.2f}, SSIM={ssim_bilinear:.4f}')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_fused_seam_carving()