import numpy as np
import cv2


def extract_roi(*, img_path: str, roi: np.array):
    """
        Function to extract a single ROI from an image that contains other features in it. Takes two parameters. one for
        the path of the image from which we want to extract the ROI, and the other is an NumPy array containing points
        that define a polygon which encloses the desired ROI to be extracted.
        :param img_path: String defining the path on the image from which the ROI will be extracted
        :param roi: NumPy array with the points that form the polygon which encloses the ROI to be extracted
    """
    # load chosen image
    img = cv2.imread(img_path, 0)
    # Generates the black mask that will be used to isolate ROI
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    # Draw contours of the ROI in the image based on the points array
    cv2.drawContours(mask, [roi], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # Determination of the ROI based on bitwise and of the original image and the mask
    res = cv2.bitwise_and(img, img, mask=mask)
    # returns (x,y,w,h) of bounding rectangle around the extracted roi
    rect = cv2.boundingRect(roi)
    # Cropping the image to have only the ROI withing a black bounding rectangle
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # Dimension of the new cropped image that I want dim(width, height)
    dim = (100, 100)

    # Resizing of the cropped image. That ensures that all the ROIs extracted have the same size
    cropped = cv2.resize(cropped, dim)

    return cropped
