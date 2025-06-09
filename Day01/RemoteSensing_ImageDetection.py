import numpy as np
import rasterio
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def output_fire_detection(tif_file, output_rgb="output_bright.png", output_fire="output_fire.png"):
    with rasterio.open(tif_file) as src:
        bands = src.read()

    # 读取波段，其中swir是用于火灾检测的
    blue = bands[0].astype(float)
    green = bands[1].astype(float)
    red = bands[2].astype(float)
    nir = bands[3].astype(float)
    swir = bands[4].astype(float)

    rgb_origin = np.dstack((red, green, blue))
    rgb_normalized = ((rgb_origin - 0) / (10000 - 0)) * 255
    rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

    rgb_original_display = np.clip(rgb_origin, 0, 10000) / 10000  # Scale to 0-1 for display

    img_rgb = Image.fromarray(rgb_normalized)
    enhancer = ImageEnhance.Brightness(img_rgb)
    img_bright = enhancer.enhance(1.5)

    img_bright.save(output_rgb)
    print(f"增强亮度的 RGB 图像已保存在 {output_rgb}")

    # 使用swir进行火灾检测
    nbr = (nir - swir) / (nir + swir + 1e-5)
    nbr = np.clip(nbr, -1, 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_original_display)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)

    plt.imshow(img_bright)
    plt.title("Brightened RGB Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)

    fire_img = plt.imshow(nbr, cmap='hot', vmin=-1, vmax=1)
    plt.title("Fire Detection (NBR)")
    plt.colorbar(fire_img, shrink=0.7)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_fire)
    plt.show()
    print(f"火灾检测图像已保存在 {output_fire}")


output_fire_detection("2019_1101_nofire_B2348_B12_10m_roi.tif")