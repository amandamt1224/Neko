import cv2
import numpy as np
import argparse
from multiprocessing import JoinableQueue, Process
from helpers import *
import sys

#load arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="Path to image")
ap.add_argument("-b", "--batchSize", type=int, help="Set batch size", default=4000)
ap.add_argument("-p", "--processes", required=True, type=int, help="Set number of processes")
ap.add_argument("-s", "--stepSize", type=int, help="Set frame step size", default=8)
ap.add_argument("-t", "--threshold", type=float, help="Set prediction threshold (example: .80)", default=.80)
args = ap.parse_args()

PROTO = '/home/amt29588/vision/new_model_kerr_7-21/cars_quick.prototxt'
MODEL = '/home/amt29588/vision/new_model_kerr_7-21/cars_quick_iter_10000.caffemodel'
MEAN = np.load('/home/amt29588/vision/new_model_kerr_7-21/out.npy')  


#calculate how many batches we will make
image = cv2.imread(args.image)
#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
height = image.shape[0]
width = image.shape[1]
xFrames = int(width/args.stepSize) - (int(64/args.stepSize)-1) #(width/step size) - (frameSize/step size - 1)
yFrames = int(height/args.stepSize) - (int(64/args.stepSize)-1) #(height/step size) - (frameSize/step size - 1) 
numFrames = xFrames * yFrames
bSize = int(args.batchSize)
#predictions = np.empty([numFrames, 2])
coordsArr = makeCoords(64, 64, args.stepSize, image)


#this only runs if your batch size is greater than or equal to your total number of frames
if len(coordsArr) <= bSize:
	#load caffe files and initialize Classifier
	net = caffe.Classifier(PROTO, MODEL, mean=MEAN, image_dims=(64,64),raw_scale=255)
	caffe.set_mode_gpu() #USES GPU BY DEFAULT
	#predictions = np.empty([numFrames, 2])
	picArr = coordsToFrames(image, coordsArr, 0, len(coordsArr))
	predictions = net.predict(picArr)
	filename = "output/coords0.csv"
	sys.stdout = open(filename, 'wb')
	writeToOutput(predictions, 0, len(coordsArr), coordsArr, args.threshold)
	sys.stdout.flush()
	combineOutput(args.processes)
	writeToPic(args.image, args.threshold)	
	quit()

#if you have more than one batch make some new processes

q = JoinableQueue(maxsize=0)
num_processes = args.processes


for i in range(num_processes):
	fileName = "output/coords%d.csv" %i
	worker = Process(target=workerFunc, args=(q, coordsArr, image, fileName, i, args.threshold, PROTO, MODEL, MEAN,))
	worker.daemon = True
	worker.start()

indices = makeIndices(bSize, numFrames)
#put tasks on queue
for j in range(0, len(indices)):
	q.put(indices[j])

q.join()

combineOutput(num_processes)
writeToPic(args.image, args.threshold)



