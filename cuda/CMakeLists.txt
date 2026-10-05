cmake_minimum_required(VERSION 3.10)
project(ad_census)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1")
set(ARCH ${CMAKE_SYSTEM_PROCESSOR})

if (${ARCH} STREQUAL "x86_64")
include_directories(/home/librui/Documents/opencv-4.x/install/include/opencv4)
file(GLOB OpenCV_LIBRARIES /home/librui/Documents/opencv-4.x/install/lib/*.so)
find_package(CUDA REQUIRED)
elseif (${ARCH} STREQUAL "aarch64")
include_directories(
  /workspace/opencv-4.x/build_aarch64/install/include/opencv4/)
file(GLOB OpenCV_LIBRARIES /workspace/opencv-4.x/build_aarch64/install/lib/*.so)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../config/)

message("build cuda source")
include_directories(include 
                    ${CUDA_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include/ 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../config/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/
                    )
file(GLOB source_file 
     "source/*" 
     "${CMAKE_CURRENT_SOURCE_DIR}/../config/*.cpp" 
     "${CMAKE_CURRENT_SOURCE_DIR}/../cpu/source/*.cpp" 
     "${CMAKE_CURRENT_SOURCE_DIR}/../utils/*"
    )
list(REMOVE_ITEM 
     source_file 
     "${CMAKE_CURRENT_SOURCE_DIR}/../cpu/source/main.cpp")

# add_compile_definitions("-fmad=false")
cuda_add_executable(cu_ad_census ${source_file})
# target_compile_options(cu_ad_census -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1)
target_link_libraries(cu_ad_census ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})
