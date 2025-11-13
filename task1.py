import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class BasicSaliencyDetector:
    """基础显著性检测器：实现颜色对比度和中心先验"""

    def __init__(self):
        pass

    def color_contrast_saliency(self, image):
        """基于颜色对比度的显著性检测"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # 计算图像的平均颜色
        mean_lab = np.mean(lab, axis=(0, 1))
        # 计算每个像素与平均颜色的欧氏距离
        saliency = np.sqrt(np.sum((lab - mean_lab) ** 2, axis=2))
        # 归一化到[0, 255]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        saliency = (saliency * 255).astype(np.uint8)

        return saliency

    def center_prior_saliency(self, image):
        """基于中心先验的显著性检测"""
        h, w = image.shape[:2]
        # 创建中心先验图
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        # 计算到中心的距离
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        # 使用高斯函数建模中心先验
        center_prior = np.exp(-(dist_from_center ** 2) / (2 * (max_dist / 2) ** 2))
        center_prior = (center_prior * 255).astype(np.uint8)

        return center_prior

    def spectral_residual_saliency(self, image):
        """谱残差显著性检测"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        # 傅里叶变换
        fft = np.fft.fft2(gray)
        log_amplitude = np.log(np.abs(fft) + 1e-7)
        phase = np.angle(fft)
        # 计算谱残差
        spectral_residual = log_amplitude - gaussian_filter(log_amplitude, sigma=3)
        # 重建图像
        combined = np.exp(spectral_residual + 1j * phase)
        saliency = np.abs(np.fft.ifft2(combined))
        # 后处理
        saliency = gaussian_filter(saliency ** 2, sigma=8)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        saliency = (saliency * 255).astype(np.uint8)
        return saliency

    def combine_saliency_maps(self, image, weights=[0.3, 0.2, 0.5]):
        """组合多种显著性检测方法"""
        color_sal = self.color_contrast_saliency(image)
        center_sal = self.center_prior_saliency(image)
        spectral_sal = self.spectral_residual_saliency(image)

        # 加权组合
        combined = (weights[0] * color_sal +
                    weights[1] * center_sal +
                    weights[2] * spectral_sal)

        combined = np.clip(combined, 0, 255).astype(np.uint8)
        return combined, color_sal, center_sal, spectral_sal


# 测试代码
def test_basic_saliency():
    """测试基础显著性检测"""
    # 读取图像
    image_path = 'sysu-sz.jpg'
    test_image = cv2.imread(image_path)

    if test_image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 初始化检测器
    detector = BasicSaliencyDetector()

    # 获取显著性图
    combined, color_sal, center_sal, spectral_sal = detector.combine_saliency_maps(test_image)

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(color_sal, cmap='hot')
    axes[0, 1].set_title('Color Contrast Saliency Map')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(center_sal, cmap='hot')
    axes[0, 2].set_title('Center Prior Saliency Map')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(spectral_sal, cmap='hot')
    axes[1, 0].set_title('Spectral Residual Saliency Map')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(combined, cmap='hot')
    axes[1, 1].set_title('Combined Saliency Map')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    return combined


if __name__ == "__main__":
    saliency_map = test_basic_saliency()
    print("显著性检测完成！")
    save_path = "saliency_map.png"
    cv2.imwrite(save_path, saliency_map)