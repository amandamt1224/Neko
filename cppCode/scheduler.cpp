/**
	* @file scheduler.cpp
	* Creates image frames, pushes tasks to queue, creates worker processes
	* @author Amanda Thomas 
	* @date (created) August 2015
	*/


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <cstdlib>
#include "classifier.h"
#include "scheduler.h"
#include <pthread.h>

int STEP_SIZE = 8;
int PROC_NUM = 30;  
int BATCH_SIZE = 5000;
int FRAME_SIZE = 64; 
float THRESHOLD = .40; 
string IMAGE_NAME = "../cars.jpg"; 
string OUTPUT_NAME = "test_output.jpg";
string MODEL = "../gray_model/cars_quick.prototxt"; 
string PROTO = "../gray_model/cars_quick_iter_10000.caffemodel"; 
string MEAN = "../gray_model/mean_train_cars_all.binaryproto"; 
Mat image = imread(IMAGE_NAME, CV_LOAD_IMAGE_GRAYSCALE); //image will be loaded without alpha channel
Mat output;
Scalar color = Scalar(255, 255, 255);
vector<int> compression_params;
vector<pair<int, int>> * coordArray;

vector<pair<int, int>> * makeCoords(){
	Size s = image.size();
	int rows = s.height;	
	int cols = s.width;
	vector<pair<int, int>> * retval = new vector<pair<int, int>>();
	for(int i = 0; i < rows; i+= STEP_SIZE){
		for(int j = 0; j < cols; j += STEP_SIZE){
			int centerX = j + (FRAME_SIZE/2);
			int centerY = i + (FRAME_SIZE/2);
			if((centerX + (FRAME_SIZE/2)) > cols || (centerY + (FRAME_SIZE/2)) > rows)
				break;
			(*retval).push_back(pair<int, int>(centerX, centerY));
		}

	}
	return retval;

}

vector<task> * makeTasks(int numFrames){
	int numBatches = int(numFrames/BATCH_SIZE);
	//int lastIndex = numFrames - (BATCH_SIZE * numBatches);
	vector<task> * tasks = new vector<task>();
	int i;
	for(i = 0; i < BATCH_SIZE * numBatches; i += BATCH_SIZE){
		(*tasks).push_back(task(i, i+BATCH_SIZE));
		cout << i << ' ' << i + BATCH_SIZE << '\n';
	}
	if(i + BATCH_SIZE != numFrames){
		(*tasks).push_back(task(i, numFrames));
	}	
	return tasks;		
	
}


vector<float> * coordsToFrames(int start, int end, Process * process){
	//part 1 -> turn coordinates into frames
	vector<Mat> frames;
	for(int i = start; i < end; i++){
		int x = (*coordArray)[i].first;
		int y = (*coordArray)[i].second;
		Mat newFrame;
		image(Rect(x - (FRAME_SIZE/2), y - (FRAME_SIZE/2), FRAME_SIZE, FRAME_SIZE)).copyTo(newFrame);
		frames.push_back(newFrame);
	}

	
	//part 2 -> get prediction values!
	vector<float> * predictions = new vector<float>();
	for(int i = 0; i < int(frames.size()); i++){
		float pred_value = process->classifier->Classify(frames[i], 5);
		(*predictions).push_back(pred_value);
				
	}
	
	return predictions;


}

void writeToOutput(vector<float> * predictions, int start){
	for(int i = 0; i < int((*predictions).size()); i++){
		if((*predictions)[i] >= THRESHOLD){ //if this prediction is over the threshold
			int x = (*coordArray)[i + start].first;
			int y = (*coordArray)[i + start].second;
			circle(output, Point(x, y), 4, color,-1, 8, 0);				
			// cout << x << ' ' << y << ' ' << (*predictions)[i] << '\n';
	
		} 

	}
	
	delete predictions;
	predictions = NULL;

}

int main(int argc, char ** argv){
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	
	
	if(!image.data){
		cout << "Could not open or find the image \n" ;
		return -1;
	}
	image.copyTo(output); //Don't want to overwrite original image
	Size s = image.size();	
	int height = s.height;
	int width = s.width;
	int xFrames = int(width/STEP_SIZE) - (int(FRAME_SIZE/STEP_SIZE)-1); //(width/STEP_SIZE) - ((FRAME_SIZE/STEP_SIZE)-1)
	int yFrames = int(height/STEP_SIZE) - (int(FRAME_SIZE/STEP_SIZE)-1); //(height/STEP_SIZE) - ((FRAME_SIZE/STEP_SIZE)-1)
	int numFrames = yFrames * xFrames;
	coordArray = new vector<pair<int, int>>();
	coordArray = makeCoords();

	//single process handles a single batch
	if(numFrames <= BATCH_SIZE){
		Classifier classifier = Classifier::Classifier(MODEL, PROTO, MEAN, true);
		Process process;
		process.classifier = &classifier;
		vector<float> * predictArray = coordsToFrames(0, numFrames, &process);
		writeToOutput(predictArray, 0);
		imwrite(OUTPUT_NAME, output, compression_params);
		return 0;

	}


	//You have more than one batch so you need more than one process
	makeTasks(numFrames);
	//cout << numFrames;
	return 0;	

}


