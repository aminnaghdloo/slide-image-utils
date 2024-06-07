import numpy as np
import scipy
import sys
from skimage import io
from scipy.fftpack import fft2
from scipy.ndimage.measurements import sum as nd_sum


def rps(img):
	assert img.ndim == 2
	radii2 = (np.arange(img.shape[0]).reshape((img.shape[0], 1)) ** 2) + (
		np.arange(img.shape[1]) ** 2
	)
	radii2 = np.minimum(radii2, np.flipud(radii2))
	radii2 = np.minimum(radii2, np.fliplr(radii2))
	maxwidth = (
		min(img.shape[0], img.shape[1]) / 8.0
	)  # truncate early to avoid edge effects
	if img.ptp() > 0:
		# Normalizing pixel intensities by median absolute deviation (MAD)
		img = img / np.median(abs(img - img.mean()))  # intensity invariant
	mag = abs(fft2(img - np.mean(img)))
	power = mag ** 2
	radii = np.floor(np.sqrt(radii2)).astype(np.int64) + 1
	labels = (
		np.arange(2, np.floor(maxwidth)).astype(np.int64).tolist()
	)  # skip DC component
	if len(labels) > 0:
		magsum = nd_sum(mag, radii, labels)
		powersum = nd_sum(power, radii, labels)
		return np.array(labels), np.array(magsum), np.array(powersum)
	return [2], [0], [0]


def PLLS(image): # Power log-log slope as sharpness metric
	radii, magnitude, power = rps(image)
	if sum(magnitude) > 0 and len(np.unique(image)) > 1:
		valid = magnitude > 0
		radii = radii[valid].reshape((-1, 1))
		power = power[valid].reshape((-1, 1))
		if radii.shape[0] > 1:
			idx = np.isfinite(np.log(power))
			powerslope = scipy.linalg.basic.lstsq(
				np.hstack(
					(
						np.log(radii)[idx][:, np.newaxis],
						np.ones(radii.shape)[idx][:, np.newaxis],
					)
				),
				np.log(power)[idx][:, np.newaxis],
				)[0][0]
		else:
			powerslope = 0
	else:
		powerslope = 0
	return powerslope


def main():
	image = io.imread(sys.argv[1])
	print(PLLS(image)[0])


if __name__ == '__main__':
	main()
