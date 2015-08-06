/**
	* @file scheduler.cpp
	* Creates image frames, pushes tasks to queue, creates worker processes
	* @author Amanda Thomas 
	* @date (created) August 2015
	*/

#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>

//using namespace std;
using namespace cv;


/// Global Variables

int STEP_SIZE    = 8;
int PROC_NUM     = 30;	
int CHUNK_SIZE   = 2000;
int FRAME_SIZE   = 64;
float THRESHOLD  = .80;
string IMAGE_NAME = "FILLER.JPEG"
string PROTO = "PROTO"
string MODEL = "MODEL"
string MEAN = "MEAN"

int main(int argc, char ** argv){

	Mat input = imread("IMAGE_NAME", 1) //image will be loaded without alpha channel
		if(!input.data){
			cout << "Could not open or find the image";
			return -1;
		}	
	int height = image.rows;
	int width = image.cols;
	int xFrames = int(width/STEP_SIZE) - (int(FRAME_SIZE/STEP_SIZE)-1) //(width/STEP_SIZE) - ((FRAME_SIZE/STEP_SIZE)-1)
	int yFrames = int(height/STEP_SIZE) - (int(FRAME_SIZE/STEP_SIZE)-1) //(height/STEP_SIZE) - ((FRAME_SIZE/STEP_SIZE)-1)
	int numFrames = yFrames * xFrames;
	std::vector<std::pair<int, int>> = makeCoords(FRAME_SIZE, FRAME_SIZE, STEP_SIZE, input);
	




}
