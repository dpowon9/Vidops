from Video_functions import smooth, play_multiple
video_path = r"C:\Users\Dennis Pkemoi\Videos\Pexels Videos 2603.mp4"
image_path = r"C:\Users\Dennis Pkemoi\Pictures\Saved Pictures\BMW.jpg"
direc = r"C:\Users\Dennis Pkemoi\Pictures\Python_practice"
save_path = direc + '/smoothed.mp4'

smooth(video_path, save_path, Methods='bilateral')
play_multiple(video_path, save_path)
