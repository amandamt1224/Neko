#version that I am using for Stu's pools
#modifications have been made to handle color pictures

import cv2
import numpy as np
import argparse
from multiprocessing import JoinableQueue, Process
#from Queue import Queue
from helpers import *

#load arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,type=str, help="Path to image")
ap.add_argument("-c", "--chunkSize", type=int, help="Set chunk size", default=4000)
ap.add_argument("-p", "--processes", required=True, type=int, help="Set number of processes")
ap.add_argument("-s", "--stepSize", type=int, help="Set frame step size", default=8)
args = ap.parse_args()


#calculate how many batches we will make
image = cv2.imread(args.image)
#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
height = image.shape[0]
width = image.shape[1]
xFrames = int(width/args.stepSize) - (int(64/args.stepSize)-1) #(width/step size) - (frameSize/step size - 1)
yFrames = int(height/args.stepSize) - (int(64/args.stepSize)-1) #(height/step size) - (frameSize/step size - 1) 
numFrames = xFrames * yFrames
cSize = args.chunkSize
coordsArr = makeCoords(64, 64, args.stepSize, image)

if len(coordsArr) <= cSize:
	#load caffe files and initialize Classifier
	mean = np.load('/home/amt29588/vision/pool_stuff/out.npy')
	net = caffe.Classifier('/home/amt29588/vision/pool_stuff/pools_quick.prototxt', '/home/amt29588/vision/pool_stuff/pools_quick_iter_2000.caffemodel', mean=mean, image_dims=(64,64),raw_scale=255)
	#this beginning verison will just use GPU mode
	caffe.set_mode_gpu()
	predictions = np.empty([numFrames, 2])
	picArr = coordsToFrames(image, coordsArr, 0, len(coordsArr))
	predictions = net.predict(picArr)
	f = open("coords0.csv", "wb")
	writer = csv.writer(f, delimiter=",")
	writeToOutput(predictions, 0, len(coordsArr), writer, coordsArr)
	quit()

#if you have more than one block make some new processes

q = JoinableQueue(maxsize=0)
num_processes = args.processes

#add code to pick best number of processes
for i in range(num_processes):
	fileName = "output/coords%d.csv" %i
	worker = Process(target=workerFunc, args=(q, coordsArr, image, fileName, i,))
	worker.daemon = True
	worker.start()

indices = makeIndices(cSize, numFrames)
#put tasks on queue
for j in range(0, len(indices)):
	q.put(indices[j])


q.join()

combineOutput(num_processes)
writeToPic(args.image)




