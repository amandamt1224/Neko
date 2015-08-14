/**
	* @file scheduler.h
	* Creates image frames, pushes tasks to queue, creates worker processes
	* @author Amanda Thomas 
	* @date (created) August 2015
	*/

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>
#include "classifier.h"
#ifndef SCHEDULER_H
#define SCHEDULER_H

using namespace std;
using namespace cv;

	typedef pair<int, int> task;

	struct Process{
			
		bool GPU_mode; //whether or not this process uses the GPU
		Classifier::Classifier * classifier;
		


	};
	
	/** Runs a "sliding window" across the image 
	* and creates a vector of all of the centroid coordinates */
	vector<pair<int, int>> * makeCoords();


	/** Split the coordinates up into batches */
	vector<task> * makeTasks(int numFrames);

	/** Takes start and end indices and makes a vector of images
	* It also takes that vector and iteratively runs the classifier*/
	vector<float> * coordsToFrames(int start, int end, Process * process);

	/** Takes vector of predictions and writes circles on the output image */
	void writeToOutput(vector<float> * predictions, int start);

	
	

#endif
