from makeBlocks import makeBlocks
import cv2

#image = cv2.imread("puppy.jpg")
picArr = makeBlocks(64, 64, 8, "puppy.jpg")
i = 0
for i in picArr:
	cv2.imwrite("output/pic%d.jpg" %(i), picArr[i])
