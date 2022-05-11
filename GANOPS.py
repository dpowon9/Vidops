from Deblur_gan_ops import GAN

worker = GAN()
worker.deblur_image(save=True)
# worker.video_deblur(model="lapsrn", scale=8)
# worker.deblur_directory()
