import csv
import os, os.path

def combineOutput(numProcesses):
	fout = open("coords.csv", "a")
	for line in open("coords0.csv"):
		fout.write(line)
	for i in range(1,numProcesses):
		f = open("coords"+str(i)+".csv")
		for line in f:
			fout.write(line)
		f.close()
	fout.close

if __name__ == "__main__": combineOutput(30)