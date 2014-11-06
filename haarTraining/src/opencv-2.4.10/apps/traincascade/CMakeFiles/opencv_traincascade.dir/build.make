# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10

# Include any dependencies generated for this target.
include apps/traincascade/CMakeFiles/opencv_traincascade.dir/depend.make

# Include the progress variables for this target.
include apps/traincascade/CMakeFiles/opencv_traincascade.dir/progress.make

# Include the compile flags for this target's objects.
include apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make

apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o: apps/traincascade/traincascade.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/traincascade.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/traincascade.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/traincascade.cpp > CMakeFiles/opencv_traincascade.dir/traincascade.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/traincascade.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/traincascade.cpp -o CMakeFiles/opencv_traincascade.dir/traincascade.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o: apps/traincascade/cascadeclassifier.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/cascadeclassifier.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/cascadeclassifier.cpp > CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/cascadeclassifier.cpp -o CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o: apps/traincascade/boost.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/boost.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/boost.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/boost.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/boost.cpp > CMakeFiles/opencv_traincascade.dir/boost.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/boost.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/boost.cpp -o CMakeFiles/opencv_traincascade.dir/boost.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o: apps/traincascade/features.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/features.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/features.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/features.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/features.cpp > CMakeFiles/opencv_traincascade.dir/features.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/features.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/features.cpp -o CMakeFiles/opencv_traincascade.dir/features.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o: apps/traincascade/haarfeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/haarfeatures.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/haarfeatures.cpp > CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/haarfeatures.cpp -o CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o: apps/traincascade/lbpfeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/lbpfeatures.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/lbpfeatures.cpp > CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/lbpfeatures.cpp -o CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o: apps/traincascade/HOGfeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/HOGfeatures.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/HOGfeatures.cpp > CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/HOGfeatures.cpp -o CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o

apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o: apps/traincascade/CMakeFiles/opencv_traincascade.dir/flags.make
apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o: apps/traincascade/imagestorage.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/imagestorage.cpp

apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/imagestorage.cpp > CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.i

apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/imagestorage.cpp -o CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.s

apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.requires:
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.provides: apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.requires
	$(MAKE) -f apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.provides.build
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.provides

apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.provides.build: apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o

# Object files for target opencv_traincascade
opencv_traincascade_OBJECTS = \
"CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/boost.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/features.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o" \
"CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o"

# External object files for target opencv_traincascade
opencv_traincascade_EXTERNAL_OBJECTS =

bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/build.make
bin/opencv_traincascade: lib/libopencv_core.so.2.4.10
bin/opencv_traincascade: lib/libopencv_ml.so.2.4.10
bin/opencv_traincascade: lib/libopencv_imgproc.so.2.4.10
bin/opencv_traincascade: lib/libopencv_objdetect.so.2.4.10
bin/opencv_traincascade: lib/libopencv_highgui.so.2.4.10
bin/opencv_traincascade: lib/libopencv_calib3d.so.2.4.10
bin/opencv_traincascade: lib/libopencv_video.so.2.4.10
bin/opencv_traincascade: lib/libopencv_features2d.so.2.4.10
bin/opencv_traincascade: lib/libopencv_flann.so.2.4.10
bin/opencv_traincascade: lib/libopencv_legacy.so.2.4.10
bin/opencv_traincascade: lib/libopencv_haartraining_engine.a
bin/opencv_traincascade: lib/libopencv_objdetect.so.2.4.10
bin/opencv_traincascade: lib/libopencv_legacy.so.2.4.10
bin/opencv_traincascade: lib/libopencv_ml.so.2.4.10
bin/opencv_traincascade: lib/libopencv_calib3d.so.2.4.10
bin/opencv_traincascade: lib/libopencv_video.so.2.4.10
bin/opencv_traincascade: lib/libopencv_features2d.so.2.4.10
bin/opencv_traincascade: lib/libopencv_highgui.so.2.4.10
bin/opencv_traincascade: lib/libopencv_imgproc.so.2.4.10
bin/opencv_traincascade: lib/libopencv_flann.so.2.4.10
bin/opencv_traincascade: lib/libopencv_core.so.2.4.10
bin/opencv_traincascade: apps/traincascade/CMakeFiles/opencv_traincascade.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../bin/opencv_traincascade"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_traincascade.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/traincascade/CMakeFiles/opencv_traincascade.dir/build: bin/opencv_traincascade
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/build

apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/traincascade.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/cascadeclassifier.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/boost.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/features.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/haarfeatures.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/lbpfeatures.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/HOGfeatures.cpp.o.requires
apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires: apps/traincascade/CMakeFiles/opencv_traincascade.dir/imagestorage.cpp.o.requires
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/requires

apps/traincascade/CMakeFiles/opencv_traincascade.dir/clean:
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade && $(CMAKE_COMMAND) -P CMakeFiles/opencv_traincascade.dir/cmake_clean.cmake
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/clean

apps/traincascade/CMakeFiles/opencv_traincascade.dir/depend:
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/apps/traincascade/CMakeFiles/opencv_traincascade.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/traincascade/CMakeFiles/opencv_traincascade.dir/depend

