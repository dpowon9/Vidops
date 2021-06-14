import cv2
import numpy as np

path = r"Butterfly Flying Away From A Flower.mp4"


def play(mode, capture=False, save=False, path_to_save=None, f=(400, 400)):
    """
    :param f: Default size of 400 by 400
    :param path_to_save: path to save video frames
    :param save: To save all frames of the video
    :param capture: If selected it will capture and save using the first camera
    :param mode: file for playing
    :return: Plays, captures, saves a video, photo, photo in a 400 by 400 frame, press Q to exit
    """
    # Create a VideoCapture object and read from input file
    # if capture is selected mode will be changed to engage camera 1
    if capture:
        print('Mode automatically changed to use camera 1')
        mode = 0
    cap = cv2.VideoCapture(mode)
    # Check if camera opened successfully
    print('VidCapture object status:', cap.isOpened())
    if not cap.isOpened():
        print("Error opening video file")
        exit()
    # Get the total number of frames
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('The input video has %d frames.' % length)
    # Get video speed in frames per second
    speed = int(cap.get(cv2.CAP_PROP_FPS))
    print('The input video plays at %d fps.' % speed)
    # Read until video is completed
    count = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        res = cv2.resize(frame, f)
        if ret:
            if save:
                while count < length:
                    cv2.imwrite(path_to_save + '/' + 'frame%d.jpg' % count, res)
                break
            else:
                # Display the resulting frame
                cv2.imshow('Frame', res)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    if capture:
                        cv2.imwrite(path_to_save + '/' + 'capture.jpg', res)
                        print('Photo captured!! \n', path_to_save + '/' + 'capture.jpg')
                    else:
                        print('(Q) pressed exiting...')
                    break
            # Break the loop
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        count += 1
        if length != -1:
            if count >= length:
                break
    # When everything done, release
    # the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def blur(mode, percent, vid_out='blurred.avi'):
    """
    :param percent: Blur percentage desired
    :param mode: Video path to blur or 0 to turn on camera 1 and blur video captured
    :param vid_out: Output blurred video
    :return: Video with median blur applied and saved
    """
    cappy = cv2.VideoCapture(mode)
    print('VidCapture object status:', cappy.isOpened())
    if not cappy.isOpened():
        print("Error opening video file")
        exit()
    # Get the total number of frames
    length = int(cappy.get(cv2.CAP_PROP_FRAME_COUNT))
    print('The input video has %d frames.' % length)
    # Get video speed in frames per second
    speed = int(cappy.get(cv2.CAP_PROP_FPS))
    print('The input video plays at %d fps.' % speed)
    # Read until video is completed
    frame_width = int(cappy.get(3))
    frame_height = int(cappy.get(4))
    # define codec and create VideoWriter object
    out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                          (frame_width, frame_height))
    count = 0
    while cappy.isOpened():
        ret, frame = cappy.read()
        # frame = cv2.resize(frame, f)
        if ret:
            frame2 = cv2.medianBlur(frame, percent)
            # Display the resulting frame
            out.write(frame2)
            cv2.imshow('Frame', frame2)
            # press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('(Q) pressed exiting...')
                break
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        count += 1
        if length != -1:
            if count >= length:
                break
    cappy.release()
    # Closes all the frames
    cv2.destroyAllWindows()


blur(path, 35)
