CXX=g++
CFLAGS=-c -Wall -std=c++0x -g3 -msse2  -I. -I/usr/local/cuda/include -I/home/blcv/LIB/caffe_bn/include/ -I/home/blcv/LIB/caffe_bn/src/
LDFLAGS= -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/usr/lib64/atlas -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcudnn  -lcblas -latlas -L/home/blcv/LIB/caffe_bn/build/lib/ -lcaffe -lproto
SOURCES=${wildcard *.cpp }
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=scheduler

all:  $(SOURCES) $(EXECUTABLE)


clean:
	rm -f *.o
	rm -f $(EXECUTABLE)
	find ./ -name \*.o  -delete

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CXX) $(CFLAGS) $< -o $@

#Source: https://groups.google.com/forum/#!topic/caffe-users/tMwpaf3s8GQ
