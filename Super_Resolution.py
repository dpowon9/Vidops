import cv2

def super_res(img, scale, model="edsr"):
    """
    img: Input image array
    models: edsr(scale: 2, 3, 4), espcn(scale: 2, 3, 4), fsrcnn(scale: 2, 3, 4), lapsrn(scale: 2, 4, 8)
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    if model == "edsr":
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
        else:
            print("EDSR models only scale at 2x, 3x, and 4x")
    elif model == "espcn":
        if scale == 2:
            path = "SuperResModels\ESPCN_x2.pb"
            sr.readModel(path)
            sr.setModel("espcn",2)
        elif scale == 3:
            path = "SuperResModels\ESPCN_x3.pb"
            sr.readModel(path)
            sr.setModel("espcn",3)
        elif scale == 4:
            path = "SuperResModels\ESPCN_x4.pb"
            sr.readModel(path)
            sr.setModel("espcn",4)
        else:
            print("ESPCN models only scale at 2x, 3x, and 4x")
    elif model == "fsrcnn":
        if scale == 2:
            path = "SuperResModels\FSRCNN_x2.pb"
            sr.readModel(path)
            sr.setModel("fsrcnn",2)
        elif scale == 3:
            path = "SuperResModels\FSRCNN_x3.pb"
            sr.readModel(path)
            sr.setModel("fsrcnn",3)
        elif scale == 4:
            path = "SuperResModels\FSRCNN_x4.pb"
            sr.readModel(path)
            sr.setModel("fsrcnn",4)
        else:
            print("FSRCNN models only scale at 2x, 3x, and 4x")
    elif model == "lapsrn":
        if scale == 2:
            path = "SuperResModels\LapSRN_x2.pb"
            sr.readModel(path)
            sr.setModel("lapsrn",2)
        elif scale == 4:
            path = "SuperResModels\LapSRN_x4.pb"
            sr.readModel(path)
            sr.setModel("lapsrn",4)
        elif scale == 8:
            path = "SuperResModels\LapSRN_x8.pb"
            sr.readModel(path)
            sr.setModel("lapsrn",8)
        else:
            print("LapSRN models only scale at 2x, 4x, and 8x")
    result = sr.upsample(img)
    return result[:,:,::-1]

if __name__ == "__main__":
    # Original image
    img = cv2.imread(r"C:\Users\Dennis Pkemoi\Desktop\Vidops\Examples\deblurred72.jpg")
    cv2.imshow("original Image", img)
    # Upscale
    model = "lapsrn"
    scale = 8
    res = super_res(img, scale=scale, model=model)
    cv2.imshow("Super Resolution", res)
    cv2.imwrite("Examples\{}_{}.jpg".format(model, str(scale)), res)
    cv2.waitKey(0)


    
