cmake_minimum_required(VERSION 3.10)
project(cl_ad_census)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLGAS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

message("build opencl source")

file(GLOB SRCS source/*)

include_directories(/home/librui/Documents/OpenCL-SDK/install/include)

add_executable(cl_ad_census ${SRCS})

target_link_libraries(cl_ad_census 
  pthread
  /home/librui/Documents/OpenCL-SDK/install/lib/libOpenCL.so
  /home/librui/Documents/OpenCL-SDK/install/lib/libOpenCLUtils.so
  /home/librui/Documents/OpenCL-SDK/install/lib/libOpenCLSDK.so)