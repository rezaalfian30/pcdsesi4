import imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# 1. Membaca gambar
image_path = 'images.noise.jpg' # Ganti dengan nama file gambar Anda
image = imageio.imread(image_path)

if len(image.shape) == 3:
    # Rumus luminance standar
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
else:
    gray = image

# 3. Pengolahan Citra dengan SciPy ndimage
# A. Penghalusan (Denoising) menggunakan Gaussian Filter
# Sigma mengatur seberapa banyak noise yang dihilangkan
img_smoothed = ndimage.gaussian_filter(gray, sigma=2)

# B. Deteksi Tepi (Edge Detection) menggunakan Sobel Filter
# Mendeteksi perubahan intensitas secara drastis (fitur/tepi objek)
sobel_x = ndimage.sobel(img_smoothed, axis=0)
sobel_y = ndimage.sobel(img_smoothed, axis=1)
# Menghitung magnitudo (kekuatan tepi)
img_edges = np.hypot(sobel_x, sobel_y)

# Normalisasi hasil deteksi tepi ke rentang 0-255 agar bisa ditampilkan
img_edges = (img_edges / np.max(img_edges)) * 255
img_edges = img_edges.astype(np.uint8)

# 4. Menampilkan Hasil dengan Matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('1. Citra Asli (Grayscale)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_smoothed, cmap='gray')
plt.title('2. Gaussian Filter (Smoothing)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_edges, cmap='gray')
plt.title('3. Sobel Filter (Deteksi Tepi)')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Program berhasil dijalankan dan gambar disimpan sebagai 'hasil_scipy.png'")