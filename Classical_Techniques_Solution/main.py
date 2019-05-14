import numpy as np
import cv2

from techniques import *

import argparse
import os

if __name__ == '__main__':
	# ARGUMENT PARSING
	# -m image / directory -p PATH
	parser = argparse.ArgumentParser(description='Polyp Segmentation from colonoscopy images using Image Processing.')

	parser.add_argument('-m', '--mode', default='image', required=True,
		help='Processing mode to load the system in. There are two modes available: image and directory')
	parser.add_argument('-p', '--path', required=True,
					help='Path of image or directory to process. In case of image, add extension as well.')

	# parse arguments
	args = parser.parse_args()

	if args.mode == 'directory':
		src = args.path

		for file in os.listdir(src):
			img = cv2.imread(os.path.join(src, file),0)

			img = cropBlack(img)
			mask = adaptiveThresholding(img)


			cv2.imshow(file+' Output', mask)

			cv2.waitKey()
			cv2.destroyWindow(file+' Output')

	elif args.mode == 'image':
		file = args.path
		img = cv2.imread(file,0)

		img = cropBlack(img)
		mask = adaptiveThresholding(img)


		cv2.imshow(file+' Output', mask)

		cv2.waitKey()
		cv2.destroyWindow(file+' Output')

	else:
		print('Invalid Argument')
		exit(-1)