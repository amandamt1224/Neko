from gpuBlocks import gpuBlocks
import cv2
#from makeBlocks import makeBlocks

predict = gpuBlocks("uppy_grayscale.JPG")
#arr = makeBlocks(64, 64, 8, "uppy_grayscale.JPG")
image = cv2.imread("uppy_grayscale.JPG", cv2.CV_LOAD_IMAGE_GRAYSCALE)
clone = image.copy()
for i in range(0, predict.shape[0]):
	if predict[i][1] > .4:
		cv2.circle(clone, ((i % 64)+32, (i/64)+32), 4, (255, 255, 0), -1)
cv2.imshow("output", clone)


