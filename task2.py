import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


class SeamCarving:
    """Seam Carving图像重定位算法"""

    def __init__(self, image, saliency_map=None):
        """
        初始化Seam Carving

        Args:
            image: 输入图像
            saliency_map: 显著性图（可选）
        """
        self.original_image = image.copy()
        self.image = image.copy()
        self.saliency_map = saliency_map

    def calculate_energy(self, use_saliency=True):
        """
        计算能量图

        Args:
            use_saliency: 是否使用显著性图增强能量
        """
        # 转换为灰度图
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Sobel梯度
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # 梯度幅值作为基础能量
        energy = np.abs(dx) + np.abs(dy)
        # 如果有显著性图，将其融合到能量图中
        if use_saliency and self.saliency_map is not None:
            # 调整显著性图大小以匹配当前图像
            resized_saliency = cv2.resize(self.saliency_map,
                                          (self.image.shape[1], self.image.shape[0]))
            # 融合显著性信息（显著区域能量更高）
            saliency_weight = 1000  # 权重系数
            energy = energy + saliency_weight * (resized_saliency / 255.0)

        return energy

    def find_vertical_seam(self, energy):
        """
        使用动态规划找到最小能量垂直缝

        Args:
            energy: 能量图

        Returns:
            seam: 垂直缝的列索引数组
        """
        h, w = energy.shape
        # 动态规划表
        dp = energy.copy()
        # 从第二行开始，计算累积最小能量
        for i in range(1, h):
            for j in range(w):
                # 检查上一行的三个可能位置
                if j == 0:
                    dp[i, j] += min(dp[i - 1, j], dp[i - 1, j + 1])
                elif j == w - 1:
                    dp[i, j] += min(dp[i - 1, j - 1], dp[i - 1, j])
                else:
                    dp[i, j] += min(dp[i - 1, j - 1], dp[i - 1, j], dp[i - 1, j + 1])
        # 回溯找到最小能量缝
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
        """找到最小能量水平缝"""
        # 转置能量图，使用垂直缝算法，然后转换回来
        energy_T = energy.T
        seam = self.find_vertical_seam(energy_T)
        return seam

    def remove_vertical_seam(self, seam):
        """移除垂直缝"""
        h, w, c = self.image.shape
        output = np.zeros((h, w - 1, c), dtype=self.image.dtype)

        for i in range(h):
            j = seam[i]
            output[i, :, :] = np.delete(self.image[i, :, :], j, axis=0)

        self.image = output

        # 同时更新显著性图
        if self.saliency_map is not None:
            h_s, w_s = self.saliency_map.shape
            output_saliency = np.zeros((h_s, w_s - 1), dtype=self.saliency_map.dtype)

            # 调整seam以匹配显著性图的尺寸
            seam_scaled = (seam * w_s / w).astype(int)

            for i in range(h_s):
                j = seam_scaled[min(i, len(seam_scaled) - 1)]
                output_saliency[i, :] = np.delete(self.saliency_map[i, :], j, axis=0)

            self.saliency_map = output_saliency

    def remove_horizontal_seam(self, seam):
        """移除水平缝"""
        self.image = np.transpose(self.image, (1, 0, 2))
        self.remove_vertical_seam(seam)
        self.image = np.transpose(self.image, (1, 0, 2))

        if self.saliency_map is not None:
            self.saliency_map = self.saliency_map.T
            # 这里简化处理，实际应该也要调整seam
            self.saliency_map = self.saliency_map.T

    def resize(self, new_width=None, new_height=None, use_saliency=True):
        """
        调整图像大小

        Args:
            new_width: 目标宽度
            new_height: 目标高度
            use_saliency: 是否使用显著性引导
        """
        if new_width is not None:
            delta_w = self.image.shape[1] - new_width

            if delta_w > 0:  # 缩小宽度
                for _ in range(delta_w):
                    energy = self.calculate_energy(use_saliency)
                    seam = self.find_vertical_seam(energy)
                    self.remove_vertical_seam(seam)

                    if _ % 10 == 0:
                        print(f"已移除 {_ + 1}/{delta_w} 条垂直缝")

        if new_height is not None:
            delta_h = self.image.shape[0] - new_height

            if delta_h > 0:  # 缩小高度
                for _ in range(delta_h):
                    energy = self.calculate_energy(use_saliency)
                    seam = self.find_horizontal_seam(energy)
                    self.remove_horizontal_seam(seam)

                    if _ % 10 == 0:
                        print(f"已移除 {_ + 1}/{delta_h} 条水平缝")

        return self.image

    def visualize_seam(self, seam, direction='vertical'):
        """可视化缝的位置"""
        vis_image = self.image.copy()

        if direction == 'vertical':
            for i in range(len(seam)):
                vis_image[i, seam[i]] = [0, 0, 255]  # 红色标记
        else:  # horizontal
            for i in range(len(seam)):
                vis_image[seam[i], i] = [0, 0, 255]

        return vis_image


# 测试代码
def test_seam_carving():
    """测试Seam Carving算法"""
    # 创建测试图像
    image_path = 'sysu-sz.jpg'
    test_image = cv2.imread(image_path)

    if test_image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 创建简单的显著性图（标记重要区域）
    saliency = cv2.imread('saliency_map.png', cv2.IMREAD_GRAYSCALE)

    # 初始化Seam Carving
    sc = SeamCarving(test_image, saliency)

    # 计算并可视化第一条缝
    energy = sc.calculate_energy(use_saliency=True)
    seam = sc.find_vertical_seam(energy)
    seam_vis = sc.visualize_seam(seam)

    # 调整图像大小
    print("原始大小:", test_image.shape[:2])
    resized = sc.resize(new_width=400, use_saliency=True)
    print("调整后大小:", resized.shape[:2])

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(saliency, cmap='hot')
    axes[0, 1].set_title('Saliency Map')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(energy, cmap='hot')
    axes[0, 2].set_title('Energy Map')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(seam_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('First Seam')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Resized Image ({resized.shape[1]}x{resized.shape[0]})')
    axes[1, 1].axis('off')

    # 对比不使用显著性的结果
    sc_no_sal = SeamCarving(test_image, None)
    resized_no_sal = sc_no_sal.resize(new_width=400, use_saliency=False)

    axes[1, 2].imshow(cv2.cvtColor(resized_no_sal, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('No Saliency Map Resized Image')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_seam_carving()