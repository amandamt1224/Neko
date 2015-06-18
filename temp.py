import cv2
import numpy as np
import argparse
from multiprocessing import JoinableQueue, Process
#from Queue import Queue
from helpers import *

#load arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
ap.add_argument("-c", "--chunkSize", required=True, help="Set chunk size")
ap.add_argument("-p", "--processes", required=True, help="Set number of processes")
args = vars(ap.parse_args())


#calculate how many batches we will make
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
height = image.shape[0]
width = image.shape[1]
xFrames = (width/8) - (8-1) #(width/step size) - (step size - 1)
yFrames = (height/8) - (8-1) #(height/step size) - (step size - 1) 
numFrames = xFrames * yFrames
cSize = int(args["chunkSize"])
#predictions = np.empty([numFrames, 2])
coordsArr = makeCoords(64, 64, 8, image)

if len(coordsArr) <= cSize:
	#load caffe files and initialize Classifier
	mean = np.load('/home/amt29588/vision/mean_test.npy')
	net = caffe.Classifier('/home/amt29588/vision/cifar10_quick.prototxt', '/home/amt29588/vision/cifar10_quick_iter_10000.caffemodel', mean=mean, image_dims=(64,64),raw_scale=255)
	#this beginning verison will just use GPU mode
	caffe.set_mode_gpu()
	predictions = np.empty([numFrames, 2])
	picArr = coordsToFrames(image, coordsArr, 0, len(coordsArr))
	predictions = net.predict(picArr)
	output = image.copy()
	writeToOutput(predictions, 0, len(coordsArr), output, coordsArr)
	cv2.imwrite("output_%s.jpg" %args["image"], output)
	quit()

#if you have more than one block make some new processes

q = JoinableQueue(maxsize=0)
num_processes = int(args["processes"])


for i in range(num_processes):
	fileName = "output/coords%d.csv" %i
	worker = Process(target=workerFunc, args=(q, coordsArr, image, fileName, i))
	worker.daemon = True
	worker.start()

indices = makeIndices(cSize, numFrames)
#put tasks on queue
for j in range(0, len(indices)):
	q.put(indices[j])

print "parent is waiting"
q.join()
print "process have joined"
#cv2.imwrite("output_%s.jpg" %args["image"], output)





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
			# yield the current window
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


def writeToOutput(predictions, start, end, f, writer, coordsArr):
	
	for i in range(0, predictions.shape[0]):
		if predictions[i][1] > .4:
			circX = coordsArr[i + start][0]+32
			circY = coordsArr[i + start][1]+32
			p = predictions[i][1]
			#print type(p)
			#print type(circX)
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
	f = open(fileName, 'w')
	writer = csv.writer(f, delimiter=",")
	print "Process enters workerFunc"
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
		print start
		print end
		print "!!!!process going into coords to frames"
		picArr = coordsToFrames(image, coordsArr, start, end)
		print "!!!!process out of coords to frames" 
		predictions = net.predict(picArr)
		print "!!!process out of predictions"
		#should I write to input in this function or after? so far i think hereeee
		print "!!!process is about to write to picture"
		writeToOutput(predictions, start, end, f, writer, coordsArr)
		print "!!!!process has written to picture"
		q.task_done()
