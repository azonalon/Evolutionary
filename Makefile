# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/eduard/programming/Evolutionary

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eduard/programming/Evolutionary

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target test
test:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running tests..."
	/usr/bin/ctest --force-new-ctest-process $(ARGS)
.PHONY : test

# Special rule for the target test
test/fast: test

.PHONY : test/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/eduard/programming/Evolutionary/CMakeFiles /home/eduard/programming/Evolutionary/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/eduard/programming/Evolutionary/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named Evolutionary

# Build rule for target.
Evolutionary: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Evolutionary
.PHONY : Evolutionary

# fast build rule for target.
Evolutionary/fast:
	$(MAKE) -f src/CMakeFiles/Evolutionary.dir/build.make src/CMakeFiles/Evolutionary.dir/build
.PHONY : Evolutionary/fast

#=============================================================================
# Target rules for targets named FluidTest

# Build rule for target.
FluidTest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 FluidTest
.PHONY : FluidTest

# fast build rule for target.
FluidTest/fast:
	$(MAKE) -f test/CMakeFiles/FluidTest.dir/build.make test/CMakeFiles/FluidTest.dir/build
.PHONY : FluidTest/fast

#=============================================================================
# Target rules for targets named Test1

# Build rule for target.
Test1: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Test1
.PHONY : Test1

# fast build rule for target.
Test1/fast:
	$(MAKE) -f test/CMakeFiles/Test1.dir/build.make test/CMakeFiles/Test1.dir/build
.PHONY : Test1/fast

#=============================================================================
# Target rules for targets named Test3

# Build rule for target.
Test3: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Test3
.PHONY : Test3

# fast build rule for target.
Test3/fast:
	$(MAKE) -f test/CMakeFiles/Test3.dir/build.make test/CMakeFiles/Test3.dir/build
.PHONY : Test3/fast

#=============================================================================
# Target rules for targets named Test2

# Build rule for target.
Test2: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Test2
.PHONY : Test2

# fast build rule for target.
Test2/fast:
	$(MAKE) -f test/CMakeFiles/Test2.dir/build.make test/CMakeFiles/Test2.dir/build
.PHONY : Test2/fast

#=============================================================================
# Target rules for targets named SurfaceTest

# Build rule for target.
SurfaceTest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 SurfaceTest
.PHONY : SurfaceTest

# fast build rule for target.
SurfaceTest/fast:
	$(MAKE) -f test/CMakeFiles/SurfaceTest.dir/build.make test/CMakeFiles/SurfaceTest.dir/build
.PHONY : SurfaceTest/fast

#=============================================================================
# Target rules for targets named CollisionTest

# Build rule for target.
CollisionTest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 CollisionTest
.PHONY : CollisionTest

# fast build rule for target.
CollisionTest/fast:
	$(MAKE) -f test/CMakeFiles/CollisionTest.dir/build.make test/CMakeFiles/CollisionTest.dir/build
.PHONY : CollisionTest/fast

#=============================================================================
# Target rules for targets named OptimizerTest

# Build rule for target.
OptimizerTest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 OptimizerTest
.PHONY : OptimizerTest

# fast build rule for target.
OptimizerTest/fast:
	$(MAKE) -f test/CMakeFiles/OptimizerTest.dir/build.make test/CMakeFiles/OptimizerTest.dir/build
.PHONY : OptimizerTest/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... test"
	@echo "... edit_cache"
	@echo "... Evolutionary"
	@echo "... FluidTest"
	@echo "... Test1"
	@echo "... Test3"
	@echo "... Test2"
	@echo "... SurfaceTest"
	@echo "... CollisionTest"
	@echo "... OptimizerTest"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

