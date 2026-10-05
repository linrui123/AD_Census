cmake_minimum_required(VERSION 3.10)
project(ad_census)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(opencl)

