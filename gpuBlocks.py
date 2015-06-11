from makeBlocks import makeBlocks
import cv2
import numpy as np
#import classifier
import argparse
#referenced nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/classification.ipynb
import caffe
import caffe.io
import time
import sys
import pdb


start_time = time.time()
mean = np.load('/home/amt29588/vision/mean_test.npy')
net = caffe.Classifier('/home/amt29588/vision/cifar10_quick.prototxt','/home/amt29588/vision/cifar10_quick_iter_10000.caffemodel',mean=mean, image_dims=(64,64), raw_scale=255)
caffe.set_mode_gpu()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("--chunkSize", required=True, help="how big are frame chunks?")
args = vars(ap.parse_args())



picArr = makeBlocks(64, 64, 8, args["image"])
print("---%s seconds ---" %(time.time() - start_time))
cSize = int(args["chunkSize"])
print len(picArr)

#image = cv2.imread(imageName)
#frame = picArr[50].copy()
#cv2.imwrite("frame.tif", frame)

if len(picArr) <= cSize:
	predictions = net.predict(picArr) 
	#change the prediction code to print the coords and make the circle
else:
	predictions = np.empty([len(picArr), 2])
	numChunks = len(picArr)/cSize #this will give the number of 200000 sized blocks
	lastIndex = len(picArr) - (cSize * numChunks)
	indices = []
	i = 0
	x = 0
	for i in range (0, cSize*(numChunks), cSize):
		indices.append((i, i+cSize))
		print indices[x]
		x += 1
	indices.append((i + cSize,len(picArr)))
	print indices[x]

	for j in range(0,numChunks+1):
		start = indices[j][0]
		end = indices[j][1]
		predictions[start:end] = net.predict(picArr[start:end])
		
print("---%s seconds ---" %(time.time() - start_time))


#part two
#process the prediction data and make an output image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
clone = image.copy()
xcoords = (image.shape[1]/8) - (8 - 1) #width of picture/stepSize - (stepSize - 1) 
ycoords = (image.shape[0]/8) - (8 - 1) #height of picture/stepSize - (stepSize - 1) 
for i in range(0, predictions.shape[0]):
        if predictions[i][1] > .4:
		
		circX = ((i % xcoords)*8)+32
		circY = ((i / xcoords)*8)+32 #caution! This may be a hacky fix!!!!
                cv2.circle(clone, (circX, circY), 4, 255, -1)
		#print circX, circY, predictions[i][1]
cv2.imwrite("output.jpg", clone)
