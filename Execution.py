from Video_functions import smooth, play_multiple
from Deblur_gan_ops import GAN
video_path = r"C:\Users\Dennis Pkemoi\Videos\Pexels Videos 2603.mp4"
image_path = r"C:\Users\Dennis Pkemoi\Downloads\scott-umstattd-lmClF825VYI-unsplash.jpg"
direc = r"C:\Users\Dennis Pkemoi\Videos\Python_practice"
save_path = direc + '/sharpened.mp4'

# smooth(video_path, save_path)
master = GAN()
master.video_deblur(vid_out=save_path)
play_multiple(video_path, save_path)
