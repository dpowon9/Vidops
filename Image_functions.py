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
