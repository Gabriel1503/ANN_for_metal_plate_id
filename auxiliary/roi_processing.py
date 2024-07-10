import numpy as np


def roi_processing(*, roi: np.array):
    """
        Function to preprocess the ROI extracted from the original image. The function also plots the utilized image and the
        magnitude spectrum obtained from the 2D FFT.
        :param roi: roi definition as a numpy array resulting from the image cropping.
    """

    # Performs 2D FFT on the image
    f = np.fft.fft2(roi)
    # Shifts zero-frequency component of the FFT to the center of the spectrum
    fshift = np.fft.fftshift(f)
    # Calculates the magnitude spectrum in decibels
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum
