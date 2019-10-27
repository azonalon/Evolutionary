cmake_minimum_required (VERSION 3.1)
project(Evolutionary)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG -Wall -Wno-unused-function -Wno-int-in-bool-context")
# set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# option(EVOLUTIONARY_BUILD_TESTS ON)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ./lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)

# message("inside evo source dir: ${PROJECT_SOURCE_DIR}/include")
include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/dependencies"
    "${PROJECT_SOURCE_DIR}"
)

add_subdirectory(src)

if(EVOLUTIONARY_BUILD_TESTS)
    add_subdirectory("${PROJECT_SOURCE_DIR}/test")
    enable_testing()
#    add_test(NAME Test1 COMMAND Test1)
#    add_test(NAME Test2 COMMAND Test2)
#    add_test(NAME Test3 COMMAND Test3)
#    add_test(NAME CollisionTest COMMAND CollisionTest)
endif()
