from Deblur_gan_ops import GAN
from PIL import Image as im
import numpy as np
from scipy.ndimage import median_filter
import pywt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
path = r"C:\Users\Dennis Pkemoi\Pictures\Camera Roll\PythonTestImage.jpg"
img = im.open(path)
#img.show(title="original")
img = np.array(img)
print(img.shape)
gray_img = np.mean(img, -1)
print(gray_img.shape)
test = im.fromarray(gray_img)
resized= test.resize([256, 256])
#resized.show()
resized = np.array(resized)
row, col = resized.shape
count = 8

def ressample(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(A)

splitres = ressample(resized, 8)
print(splitres.shape)

"""
while i <= row:
    print("Grid made, count: ", i)
    chunk = resized[i:count, i:count]
    med = np.median(chunk)
    medSub = np.full(chunk.shape, med)
    if count  == row:
        break
    i = count
    count += 8
n, w = 2, "db1"
coeffs = pywt.wavedec2(gray_img, wavelet=w, level=n)

coeffs[0] /= np.abs(coeffs[0]).max()
for level in range(n):
    coeffs[level+1] = [d/np.abs(d).max() for d in coeffs[level+1]]

arr, slices = pywt.coeffs_to_array(coeffs)
arr2 = pywt.array_to_coeffs(arr, slices, "wavedec2")
print(len(arr2))
#arr = arr[slices[2]['dd']]
arr = arr
print(arr.shape)
out = im.fromarray((arr* 255).astype(np.uint8))
out.show()
plt.imshow(arr, cmap="gray_r", vmin=-0.25, vmax=0.75)
plt.rcParams["figure.figsize"] = [16, 16]
plt.show()
------------------------------------------------------------------------------------------------------------------
background = median_filter(img, size=3)
print(background.shape)
fore = img - background
resampled = np.zeros(img.shape)
for i in range(fore.shape[-1]):
    resampled[:, :, i] = fore[:, :, i] 
resampled = resampled - background
print(resampled.shape)
background = im.fromarray(background)
background.show(title="Background")
fore = im.fromarray(fore)
fore.show(title="foreground")
resampled = im.fromarray((resampled * 255).astype(np.uint8))
resampled.show()
-------------------------------------------------------------------------------------------------------------------
"""
