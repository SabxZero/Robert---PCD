import imageio as img
import numpy as np
import matplotlib.pyplot as plt

image = img.imread("Pewdiepie1.jpg", mode='F')

robertX = np.array([
    [1, 0],
    [0, -1]
])

robertY = np.array([
    [0, 1],
    [-1, 0]
])

imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

for y in range(1, imgPad.shape[0]-1):
    for x in range(1, imgPad.shape[1]-1):
        region = imgPad[y-1:y+1, x-1:x+1]  
        Gx[y-1, x-1] = (region * robertX).sum()
        Gy[y-1, x-1] = (region * robertY).sum()


G = np.sqrt(Gx**2 + Gy**2)
G = (G / G.max()) * 255  
G = np.clip(G, 0, 255)  
G = G.astype(np.uint8)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Gradient X (Robert)")
plt.imshow(Gx, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Gradient Y (Robert)")
plt.imshow(Gy, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Edge Detection (Robert)")
plt.imshow(G, cmap='gray')

plt.tight_layout()
plt.show()
