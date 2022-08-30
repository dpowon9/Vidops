import os
from PIL import Image
  
def compressMe(file):
    filepath = os.path.join(os.getcwd(), file)
    picture = Image.open(filepath)
    picture.save("Examples\\Compressed.jpg", "JPEG", optimize = True, quality = 5)
    return
  

if __name__ == "__main__":
    path = r"C:\Users\Dennis Pkemoi\Pictures\Camera Roll\Free Raw Files - Tag @signatureeditsco APG_0323.CR2"
    compressMe(path)