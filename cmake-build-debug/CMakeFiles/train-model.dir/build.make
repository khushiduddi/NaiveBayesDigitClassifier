# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/train-model.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/train-model.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/train-model.dir/flags.make

CMakeFiles/train-model.dir/apps/train_model_main.cc.o: CMakeFiles/train-model.dir/flags.make
CMakeFiles/train-model.dir/apps/train_model_main.cc.o: ../apps/train_model_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/train-model.dir/apps/train_model_main.cc.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train-model.dir/apps/train_model_main.cc.o -c /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/apps/train_model_main.cc

CMakeFiles/train-model.dir/apps/train_model_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train-model.dir/apps/train_model_main.cc.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/apps/train_model_main.cc > CMakeFiles/train-model.dir/apps/train_model_main.cc.i

CMakeFiles/train-model.dir/apps/train_model_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train-model.dir/apps/train_model_main.cc.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/apps/train_model_main.cc -o CMakeFiles/train-model.dir/apps/train_model_main.cc.s

CMakeFiles/train-model.dir/src/core/digit_classifier.cc.o: CMakeFiles/train-model.dir/flags.make
CMakeFiles/train-model.dir/src/core/digit_classifier.cc.o: ../src/core/digit_classifier.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/train-model.dir/src/core/digit_classifier.cc.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train-model.dir/src/core/digit_classifier.cc.o -c /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/digit_classifier.cc

CMakeFiles/train-model.dir/src/core/digit_classifier.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train-model.dir/src/core/digit_classifier.cc.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/digit_classifier.cc > CMakeFiles/train-model.dir/src/core/digit_classifier.cc.i

CMakeFiles/train-model.dir/src/core/digit_classifier.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train-model.dir/src/core/digit_classifier.cc.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/digit_classifier.cc -o CMakeFiles/train-model.dir/src/core/digit_classifier.cc.s

CMakeFiles/train-model.dir/src/core/model.cpp.o: CMakeFiles/train-model.dir/flags.make
CMakeFiles/train-model.dir/src/core/model.cpp.o: ../src/core/model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/train-model.dir/src/core/model.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train-model.dir/src/core/model.cpp.o -c /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/model.cpp

CMakeFiles/train-model.dir/src/core/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train-model.dir/src/core/model.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/model.cpp > CMakeFiles/train-model.dir/src/core/model.cpp.i

CMakeFiles/train-model.dir/src/core/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train-model.dir/src/core/model.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/model.cpp -o CMakeFiles/train-model.dir/src/core/model.cpp.s

CMakeFiles/train-model.dir/src/core/sample.cpp.o: CMakeFiles/train-model.dir/flags.make
CMakeFiles/train-model.dir/src/core/sample.cpp.o: ../src/core/sample.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/train-model.dir/src/core/sample.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train-model.dir/src/core/sample.cpp.o -c /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/sample.cpp

CMakeFiles/train-model.dir/src/core/sample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train-model.dir/src/core/sample.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/sample.cpp > CMakeFiles/train-model.dir/src/core/sample.cpp.i

CMakeFiles/train-model.dir/src/core/sample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train-model.dir/src/core/sample.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/src/core/sample.cpp -o CMakeFiles/train-model.dir/src/core/sample.cpp.s

# Object files for target train-model
train__model_OBJECTS = \
"CMakeFiles/train-model.dir/apps/train_model_main.cc.o" \
"CMakeFiles/train-model.dir/src/core/digit_classifier.cc.o" \
"CMakeFiles/train-model.dir/src/core/model.cpp.o" \
"CMakeFiles/train-model.dir/src/core/sample.cpp.o"

# External object files for target train-model
train__model_EXTERNAL_OBJECTS =

train-model: CMakeFiles/train-model.dir/apps/train_model_main.cc.o
train-model: CMakeFiles/train-model.dir/src/core/digit_classifier.cc.o
train-model: CMakeFiles/train-model.dir/src/core/model.cpp.o
train-model: CMakeFiles/train-model.dir/src/core/sample.cpp.o
train-model: CMakeFiles/train-model.dir/build.make
train-model: /usr/local/lib/libboost_program_options-mt.dylib
train-model: CMakeFiles/train-model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable train-model"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train-model.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/train-model.dir/build: train-model

.PHONY : CMakeFiles/train-model.dir/build

CMakeFiles/train-model.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/train-model.dir/cmake_clean.cmake
.PHONY : CMakeFiles/train-model.dir/clean

CMakeFiles/train-model.dir/depend:
	cd /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2 /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2 /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug /Users/khushiduddi/Downloads/Cinder/my-projects/naive-bayes-kduddi2/cmake-build-debug/CMakeFiles/train-model.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/train-model.dir/depend

