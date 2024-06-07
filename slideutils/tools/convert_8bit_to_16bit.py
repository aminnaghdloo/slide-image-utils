import cv2
import sys
import os

images = sys.argv[1:]
os.makedirs('16bit', exist_ok=True)

print("Converting 8-bit input images to 16-bit images...")

for image in images:
	x = cv2.imread(image, -1)
	x = x.astype('uint16')
	x = x * 256
	cv2.imwrite(f'16bit/{image}', x)
	print(f"converted {image}")

print("Conversion finished successfully!")
