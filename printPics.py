from makeBlocks import makeBlocks
import cv2

image = cv2.imread("12TVK440400.tif")
picArr = makeBlocks(64, 64, 8, "12TVK440400.tif")

i = 0
for i in picArr:
	cv2.imwrite("output/frame%d.jpg" %(i), image[picArr[i].y1:picArr[i].y2, picArr[i].x1:picArr[i].x2])
