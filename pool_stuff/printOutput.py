import csv
import cv2
import os, os.path

def combineOutput(numProcesses):
	fout = open("coords.csv", "a")
	for line in open("output/coords0.csv"):
		fout.write(line)
	for i in range(1,numProcesses):
		f = open("output/coords"+str(i)+".csv")
		for line in f:
			fout.write(line)
		f.close()
	fout.close

def writeToPic():	
	image = cv2.imread("pool_pic_july.jpg")
	
	output = image.copy()
	with open('output/coords.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			cv2.circle(output, (int(row[0]), int(row[1])), 4, 255, -1)
	cv2.imwrite("output.jpg", output)

if __name__ == "__main__": combineOutput(4)
