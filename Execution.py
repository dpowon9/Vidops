from Video_functions import smooth, play_multiple
from Deblur_gan_ops import GAN
import os
video_path = r"C:\Users\Dennis Pkemoi\Videos\Blurred_videos\London_NightTrafficBokeh_1080p.mp4"
direc = r"C:\Users\Dennis Pkemoi\Videos\Python_practice"
save_path = direc + '/sharpened.mp4'

# smooth(video_path, save_path)
master = GAN()
master.video_deblur()
