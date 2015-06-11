import sys
import numpy as np
def classifier():
#	caffe_root = '/user/local/include/caffe'
	import caffe
	
	
	MODEL_FILE = '/home/amt29588/vision/cifar10_quick.prototxt'
	PRETRAINED = '/home/amt29588/vision/cifar10_quick_iter_10000.caffemodel'
	mean_file = '/home/amt29588/vision/mean_test.npy'
	mean = np.load(mean_file)
	net= caffe.Classifier(MODEL_FILE, PRETRAINED,
                	mean=mean,
                	raw_scale=255,
                	image_dims=(64, 64))

	
	return net

