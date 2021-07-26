import cv2


def blur_meth(array, Method, percent=35, sigma=3, box=(5, 5), fill=(15, 75, 75)):
    """
    :param Method: Blurring Method
    :param fill: Bilateral filter parameters, default is (15, 75, 75)
    :param box: Averaging desired kernel size, default is (5, 5)
    :param array: Input image array
    :param percent: Percent of median blur to apply, default is 35
    :param sigma: Standard deviation of the gaussian kernel, the kernel size is 3*sigma in all directions, i.e 2*3*sigma
    :return: Smoothed out image
    """
    if not Method.isupper():
        Method = Method.upper()
    if Method == 'MEDIAN':
        out = cv2.medianBlur(array, percent)
    elif Method == 'GAUSSIAN':
        size = (6 * sigma, 6 * sigma)
        out = cv2.GaussianBlur(array, size, sigma)
    elif Method == 'AVERAGING':
        out = cv2.blur(array, box)
    elif Method == 'BILATERAL':
        d, s1, s2 = fill
        out = cv2.bilateralFilter(array, d, s1, s2)
    else:
        out = None
    return out


def play_process(mode, capture=False, save=False, path_to_save=None, f=(400, 400)):
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
                if path_to_save is None:
                    path_to_save = input('Path to save is needed for saving, please input:')
                while count < length:
                    cv2.imwrite(path_to_save + '/' + 'frame%d.jpg' % count, res)
                break
            else:
                # Display the resulting frame
                cv2.imshow('Frame', res)
                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if capture:
                        if path_to_save is None:
                            path_to_save = input('Path to save is needed for saving, please input:')
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
    cap.release()
    cv2.destroyAllWindows()


def smooth(mode, vid_out, percent=45):
    """
    :param percent: Percent of median blur to apply, default is 45
    :param mode: Video path to blur or 0 to turn on camera 1 and blur video captured
    :param vid_out: Output blurred video
    :return: Video with median blur applied and saved or a blur removed
    """
    cappy = cv2.VideoCapture(mode)
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
    out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame_width, frame_height))
    count = 0
    while cappy.isOpened():
        ret, frame = cappy.read()
        if ret:
            frame2 = cv2.medianBlur(frame, percent)
            out.write(frame2)
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
    cv2.destroyAllWindows()


def play_multiple(*args, size=(300, 300)):
    """
    :param args: Videos  to play
    :param size: Size of video windows
    :return: Plays videos side by side
    """
    # Create a video capture object
    try:
        caps = [cv2.VideoCapture(i) for i in args]
        window_titles = [str(i) for i in range(len(args))]
    except Exception:
        # If a list is passed in it will unpack the list
        caps = [cv2.VideoCapture(i) for i in args[0]]
        window_titles = [str(i) for i in range(len(args[0]))]
    length = []
    print('Number of videos: %d' % len(caps))
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print("Error opening video %d stream or file" % i)
        # Get the number of frames in all videos
        length.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('The longest input video has %d frames.' % max(length))
    # preallocating
    frames = [None] * len(caps)
    ret = [None] * len(caps)
    count = 1
    # Adjusting locations on the monitor where the videos will play
    # This is to ensure they will not overlap
    # They will be displayed side by side horizontally
    x, y = 0, 100
    for i in range(len(window_titles)):
        cv2.namedWindow(window_titles[i])
        cv2.moveWindow(window_titles[i], x, y)
        x += size[0] + 30
    while True:
        for i, c in enumerate(caps):
            if c is not None:
                ret[i], frames[i] = c.read()
                frames[i] = cv2.resize(frames[i], size)
        for i, f in enumerate(frames):
            if ret[i] is True:
                cv2.imshow(window_titles[i], f)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        count += 1
        if count >= max(length):
            print('Max frames played!')
            break
    for c in caps:
        if c is not None:
            c.release()
    cv2.destroyAllWindows()
