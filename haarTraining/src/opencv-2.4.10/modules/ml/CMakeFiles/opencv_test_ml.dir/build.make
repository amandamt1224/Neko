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
include modules/ml/CMakeFiles/opencv_test_ml.dir/depend.make

# Include the progress variables for this target.
include modules/ml/CMakeFiles/opencv_test_ml.dir/progress.make

# Include the compile flags for this target's objects.
include modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o: modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make
modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o: modules/ml/test/test_emknearestkmeans.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -o CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_emknearestkmeans.cpp

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_emknearestkmeans.cpp > CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.i

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_emknearestkmeans.cpp -o CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.s

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.requires:
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.requires

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.provides: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.requires
	$(MAKE) -f modules/ml/CMakeFiles/opencv_test_ml.dir/build.make modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.provides.build
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.provides

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.provides.build: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o: modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make
modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o: modules/ml/test/test_gbttest.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -o CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_gbttest.cpp

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_gbttest.cpp > CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.i

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_gbttest.cpp -o CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.s

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.requires:
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.requires

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.provides: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.requires
	$(MAKE) -f modules/ml/CMakeFiles/opencv_test_ml.dir/build.make modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.provides.build
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.provides

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.provides.build: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o: modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make
modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o: modules/ml/test/test_main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -o CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_main.cpp

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_main.cpp > CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.i

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_main.cpp -o CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.s

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.requires:
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.requires

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.provides: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.requires
	$(MAKE) -f modules/ml/CMakeFiles/opencv_test_ml.dir/build.make modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.provides.build
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.provides

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.provides.build: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o: modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make
modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o: modules/ml/test/test_mltests.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -o CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_mltests.cpp

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_mltests.cpp > CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.i

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_mltests.cpp -o CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.s

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.requires:
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.requires

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.provides: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.requires
	$(MAKE) -f modules/ml/CMakeFiles/opencv_test_ml.dir/build.make modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.provides.build
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.provides

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.provides.build: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o: modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make
modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o: modules/ml/test/test_mltests2.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -o CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_mltests2.cpp

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_mltests2.cpp > CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.i

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_mltests2.cpp -o CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.s

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.requires:
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.requires

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.provides: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.requires
	$(MAKE) -f modules/ml/CMakeFiles/opencv_test_ml.dir/build.make modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.provides.build
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.provides

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.provides.build: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o: modules/ml/CMakeFiles/opencv_test_ml.dir/flags.make
modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o: modules/ml/test/test_save_load.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -o CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o -c /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_save_load.cpp

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.i"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -E /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_save_load.cpp > CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.i

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.s"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && /usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS)  -include "/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test_precomp.hpp" -Winvalid-pch  -S /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/test/test_save_load.cpp -o CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.s

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.requires:
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.requires

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.provides: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.requires
	$(MAKE) -f modules/ml/CMakeFiles/opencv_test_ml.dir/build.make modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.provides.build
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.provides

modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.provides.build: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o

# Object files for target opencv_test_ml
opencv_test_ml_OBJECTS = \
"CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o" \
"CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o" \
"CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o" \
"CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o" \
"CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o" \
"CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o"

# External object files for target opencv_test_ml
opencv_test_ml_EXTERNAL_OBJECTS =

bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/build.make
bin/opencv_test_ml: lib/libopencv_core.so.2.4.10
bin/opencv_test_ml: lib/libopencv_ml.so.2.4.10
bin/opencv_test_ml: lib/libopencv_ts.a
bin/opencv_test_ml: lib/libopencv_highgui.so.2.4.10
bin/opencv_test_ml: lib/libopencv_core.so.2.4.10
bin/opencv_test_ml: lib/libopencv_flann.so.2.4.10
bin/opencv_test_ml: lib/libopencv_imgproc.so.2.4.10
bin/opencv_test_ml: lib/libopencv_highgui.so.2.4.10
bin/opencv_test_ml: lib/libopencv_features2d.so.2.4.10
bin/opencv_test_ml: lib/libopencv_video.so.2.4.10
bin/opencv_test_ml: lib/libopencv_highgui.so.2.4.10
bin/opencv_test_ml: lib/libopencv_flann.so.2.4.10
bin/opencv_test_ml: lib/libopencv_imgproc.so.2.4.10
bin/opencv_test_ml: lib/libopencv_core.so.2.4.10
bin/opencv_test_ml: modules/ml/CMakeFiles/opencv_test_ml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../bin/opencv_test_ml"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_test_ml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/ml/CMakeFiles/opencv_test_ml.dir/build: bin/opencv_test_ml
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/build

modules/ml/CMakeFiles/opencv_test_ml.dir/requires: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_emknearestkmeans.cpp.o.requires
modules/ml/CMakeFiles/opencv_test_ml.dir/requires: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_gbttest.cpp.o.requires
modules/ml/CMakeFiles/opencv_test_ml.dir/requires: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_main.cpp.o.requires
modules/ml/CMakeFiles/opencv_test_ml.dir/requires: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests.cpp.o.requires
modules/ml/CMakeFiles/opencv_test_ml.dir/requires: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_mltests2.cpp.o.requires
modules/ml/CMakeFiles/opencv_test_ml.dir/requires: modules/ml/CMakeFiles/opencv_test_ml.dir/test/test_save_load.cpp.o.requires
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/requires

modules/ml/CMakeFiles/opencv_test_ml.dir/clean:
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml && $(CMAKE_COMMAND) -P CMakeFiles/opencv_test_ml.dir/cmake_clean.cmake
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/clean

modules/ml/CMakeFiles/opencv_test_ml.dir/depend:
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/ml/CMakeFiles/opencv_test_ml.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/ml/CMakeFiles/opencv_test_ml.dir/depend

