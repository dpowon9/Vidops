import os
from PIL import Image
import numpy as np
  
def compressMe(file):
    filepath = os.path.join(os.getcwd(), file)
    picture = Image.open(filepath)
    picture.save("Examples\\Compressed.jpg", "JPEG", optimize = True, quality = 5)
    return
  

if __name__ == "__main__":
    path = r"C:\Users\Dennis Pkemoi\Downloads\balloons_noisy.png"
    #compressMe(path)
    img = Image.open(path)
    img = np.array(img)
    print(img.shape)
    gray_img = np.mean(img, -1)
    print(gray_img.shape)
    test = Image.fromarray(gray_img)
    resized= test.resize([256, 256])
    resized = np.array(resized)
    out = np.array(np.vsplit(resized, resized.shape[0]/8))
    print(out.shape)
    comb1 = np.array(np.concatenate(out, axis=1))
    print(comb1.shape)