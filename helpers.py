import cv2
import numpy as np
import caffe.io
import skimage
import time
import imutils
import caffe
import csv
#from Queue import Queue
import multiprocessing

#from pyimagesearch.com
def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			yield (x, y, (image[y:y + windowSize[1], x:x + windowSize[0]]))

def makeCoords(blockW, blockH, step, imageArr):
	coordsArr = []
	i = 0
	for (x, y, window) in sliding_window(imageArr, stepSize = step, windowSize=(blockW, blockH)):
		if window.shape[0] != blockH or window.shape[1] != blockW:
			continue 
					
		coordsArr.append((x,y))
	
		i += 1
	                                                       
	return coordsArr	


def makeIndices(cSize, numFrames):
	numChunks = numFrames/cSize
	lastIndex = numFrames - (cSize * numChunks)
	indices = []
	i = 0
	x = 0
	for i in range(0, cSize *(numChunks), cSize):
		indices.append((i, i+cSize))
		x += 1
	indices.append((i + cSize, numFrames))
	return indices


def writeToOutput(predictions, start, end, writer, coordsArr):
	
	for i in range(0, predictions.shape[0]):
		if predictions[i][1] > .4:
			circX = coordsArr[i + start][0]+32
			circY = coordsArr[i + start][1]+32
			p = predictions[i][1]
			writer.writerow([circX, circY, p])





def coordsToFrames(image, coordsArr, start, end):
	frameArr = []
	for i in range(start, end):
		x = coordsArr[i][0]
		y = coordsArr[i][1]
		window = image[y:y + 64, x:x + 64]
		window = window[:, :, np.newaxis]
		window = skimage.img_as_float(window).astype(np.float32) 			
		frameArr.append(window.copy())
	return frameArr

def workerFunc(q, coordsArr, image, fileName, idNumber):
	f = open(fileName, 'wb')
	writer = csv.writer(f, delimiter=",")
	#load caffe files and initialize Classifier
	mean = np.load('/home/amt29588/vision/mean_test.npy')
	net = caffe.Classifier('/home/amt29588/vision/cifar10_quick.prototxt', '/home/amt29588/vision/cifar10_quick_iter_10000.caffemodel', mean=mean, image_dims=(64,64),raw_scale=255)
	#this beginning verison will just use GPU mode
	if idNumber < 4:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()	
	while True:
		task = q.get() #this will be in the format (start coords #, end coords #)
		#modify make blocks to take indices of image
		start = task[0]
		end = task[1]
		picArr = coordsToFrames(image, coordsArr, start, end) 
		predictions = net.predict(picArr)
		writeToOutput(predictions, start, end, writer, coordsArr)
		q.task_done()

def combineOutput(numProcesses):
	fout = open("output/coords.csv", "a")
	for line in open("output/coords0.csv"):
		fout.write(line)
	for i in range(1,numProcesses):
		f = open("output/coords"+str(i)+".csv")
		for line in f:
			fout.write(line)
		f.close()
	fout.close

def writeToPic(filename):	
	image = cv2.imread(filename)
	
	output = image.copy()
	with open('output/coords.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			cv2.circle(output, (int(row[0]), int(row[1])), 4, 255, -1)
	cv2.imwrite("output.jpg", output)
