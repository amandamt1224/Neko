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





