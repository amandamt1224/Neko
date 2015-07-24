# import the necessary packages
import sliding_window
import argparse
import time
import cv2
import pdb 
import numpy as np 
import caffe
import skimage

###
#pdb.set_trace()
PROTO = '/home/amt29588/vision/new_model_kerr_7-21/cars_quick.prototxt'
MODEL = '/home/amt29588/vision/new_model_kerr_7-21/cars_quick_iter_10000.caffemodel'
mean = np.load('/home/amt29588/vision/new_model_kerr_7-21/out.npy')
net = caffe.Classifier(PROTO, MODEL, mean=mean, image_dims=(64,64), raw_scale=256) 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (64,64)
output = image.copy()
for (x,y,window) in sliding_window.sliding_window(image,stepSize = 8, windowSize=(winW, winH)):
	if window.shape[0] != winH or window.shape[1] != winW:
		continue
	#window = window[:, :, np.newaxis]
	w = skimage.img_as_float(window).astype(np.float32)
	prediction = net.predict([w])
	print x,y,prediction[0][1]
	cv2.imshow("Window",window)	
	clone = output.copy()
	if prediction[0][1] > .8:
		cv2.circle(output,(x+32,y+32),4,(255,255,0),-1)	
	cv2.rectangle(clone,(x,y),(x+winW,y+winH),(255,255,0),2)
	cv2.imshow("Window", clone)
	cv2.waitKey(1)
	time.sleep(.005)
 		
