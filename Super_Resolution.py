import cv2
import matplotlib.pyplot as plt

def super_res(img, scale):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    if scale == 2:
        path = "SuperResModels\EDSR_x2.pb"
        sr.readModel(path)
        sr.setModel("edsr",2)
    elif scale == 3:
        path = "SuperResModels\EDSR_x3.pb"
        sr.readModel(path)
        sr.setModel("edsr",3)
    elif scale == 4:
        path = "SuperResModels\EDSR_x4.pb"
        sr.readModel(path)
        sr.setModel("edsr",4)
    result = sr.upsample(img)
    return result[:,:,::-1]

if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\Dennis Pkemoi\Desktop\Vidops\Examples\deblurred72.jpg")
    # Original image
    plt.imshow(img[:,:,::-1])
    plt.title("Original Image")
   # OpenCV upscaled
    resized = cv2.resize(img, dsize=None, fx=4, fy=4)
    fig3 = plt.figure()
    plt.imshow(resized[:,:,::-1])
    plt.title("OpenCV upscaled")
    # SR upscaled
    fig2 = plt.figure()
    plt.imshow(super_res(img, 4))
    plt.axis("off")
    plt.savefig("Examples\super_res_SR_4.jpg")
    plt.show()
    
