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
CMAKE_SOURCE_DIR = /home/azbret/Projects/hog_svm/dlib-18.12/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/azbret/Projects/hog_svm/dlib-18.12/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/pipe_ex_2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pipe_ex_2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pipe_ex_2.dir/flags.make

CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o: CMakeFiles/pipe_ex_2.dir/flags.make
CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o: ../pipe_ex_2.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/azbret/Projects/hog_svm/dlib-18.12/examples/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o -c /home/azbret/Projects/hog_svm/dlib-18.12/examples/pipe_ex_2.cpp

CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/azbret/Projects/hog_svm/dlib-18.12/examples/pipe_ex_2.cpp > CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.i

CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/azbret/Projects/hog_svm/dlib-18.12/examples/pipe_ex_2.cpp -o CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.s

CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.requires:
.PHONY : CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.requires

CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.provides: CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.requires
	$(MAKE) -f CMakeFiles/pipe_ex_2.dir/build.make CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.provides.build
.PHONY : CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.provides

CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.provides.build: CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o

# Object files for target pipe_ex_2
pipe_ex_2_OBJECTS = \
"CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o"

# External object files for target pipe_ex_2
pipe_ex_2_EXTERNAL_OBJECTS =

pipe_ex_2: CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o
pipe_ex_2: CMakeFiles/pipe_ex_2.dir/build.make
pipe_ex_2: dlib_build/libdlib.a
pipe_ex_2: /usr/lib64/libpthread.so
pipe_ex_2: /usr/lib64/libnsl.so
pipe_ex_2: /usr/lib64/libSM.so
pipe_ex_2: /usr/lib64/libICE.so
pipe_ex_2: /usr/lib64/libX11.so
pipe_ex_2: /usr/lib64/libXext.so
pipe_ex_2: /usr/lib64/libpng.so
pipe_ex_2: /opt/intel/mkl/lib/intel64/libmkl_rt.so
pipe_ex_2: /usr/lib64/libfftw3.so
pipe_ex_2: CMakeFiles/pipe_ex_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable pipe_ex_2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pipe_ex_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pipe_ex_2.dir/build: pipe_ex_2
.PHONY : CMakeFiles/pipe_ex_2.dir/build

CMakeFiles/pipe_ex_2.dir/requires: CMakeFiles/pipe_ex_2.dir/pipe_ex_2.cpp.o.requires
.PHONY : CMakeFiles/pipe_ex_2.dir/requires

CMakeFiles/pipe_ex_2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pipe_ex_2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pipe_ex_2.dir/clean

CMakeFiles/pipe_ex_2.dir/depend:
	cd /home/azbret/Projects/hog_svm/dlib-18.12/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/azbret/Projects/hog_svm/dlib-18.12/examples /home/azbret/Projects/hog_svm/dlib-18.12/examples /home/azbret/Projects/hog_svm/dlib-18.12/examples/build /home/azbret/Projects/hog_svm/dlib-18.12/examples/build /home/azbret/Projects/hog_svm/dlib-18.12/examples/build/CMakeFiles/pipe_ex_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pipe_ex_2.dir/depend
