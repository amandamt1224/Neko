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

# Utility rule file for pch_Generate_opencv_calib3d.

# Include the progress variables for this target.
include modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/progress.make

modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d: modules/calib3d/precomp.hpp.gch/opencv_calib3d_RELEASE.gch

modules/calib3d/precomp.hpp.gch/opencv_calib3d_RELEASE.gch: modules/calib3d/src/precomp.hpp
modules/calib3d/precomp.hpp.gch/opencv_calib3d_RELEASE.gch: modules/calib3d/precomp.hpp
modules/calib3d/precomp.hpp.gch/opencv_calib3d_RELEASE.gch: lib/libopencv_calib3d_pch_dephelp.a
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating precomp.hpp.gch/opencv_calib3d_RELEASE.gch"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d && /usr/bin/cmake -E make_directory /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/precomp.hpp.gch
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d && /usr/lib64/ccache/c++ -O3 -DNDEBUG -DNDEBUG -fPIC -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/features2d/include" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/highgui/include" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/imgproc/include" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/flann/include" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/core/include" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/src" -I"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/include" -isystem"/home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10" -isystem"/usr/include/eigen3" -fsigned-char -W -Wall -Werror=return-type -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wno-narrowing -Wno-delete-non-virtual-dtor -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -msse -msse2 -msse3 -ffunction-sections -DCVAPI_EXPORTS -x c++-header -o /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/precomp.hpp.gch/opencv_calib3d_RELEASE.gch /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/precomp.hpp

modules/calib3d/precomp.hpp: modules/calib3d/src/precomp.hpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating precomp.hpp"
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d && /usr/bin/cmake -E copy /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/src/precomp.hpp /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/precomp.hpp

pch_Generate_opencv_calib3d: modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d
pch_Generate_opencv_calib3d: modules/calib3d/precomp.hpp.gch/opencv_calib3d_RELEASE.gch
pch_Generate_opencv_calib3d: modules/calib3d/precomp.hpp
pch_Generate_opencv_calib3d: modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/build.make
.PHONY : pch_Generate_opencv_calib3d

# Rule to build all files generated by this target.
modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/build: pch_Generate_opencv_calib3d
.PHONY : modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/build

modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/clean:
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d && $(CMAKE_COMMAND) -P CMakeFiles/pch_Generate_opencv_calib3d.dir/cmake_clean.cmake
.PHONY : modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/clean

modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/depend:
	cd /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10 /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d /home/andrew/Projects/computer_vision/car_detection/Neko/haarTraining/src/opencv-2.4.10/modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/calib3d/CMakeFiles/pch_Generate_opencv_calib3d.dir/depend

