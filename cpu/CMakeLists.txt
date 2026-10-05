cmake_minimum_required(VERSION 3.10)
project(ad_census)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(ARCH ${CMAKE_SYSTEM_PROCESSOR})

if (${ARCH} STREQUAL "x86_64")
include_directories(/home/librui/Documents/opencv-4.x/install/include/opencv4)
file(GLOB OpenCV_LIBRARIES /home/librui/Documents/opencv-4.x/install/lib/*.so)
elseif (${ARCH} STREQUAL "aarch64")
include_directories(
  /workspace/opencv-4.x/build_aarch64/install/include/opencv4/)
file(GLOB OpenCV_LIBRARIES /workspace/opencv-4.x/build_aarch64/install/lib/*.so)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../config/)
  
message("build cpu source")
include_directories(include)
file(GLOB source_file "source/*" "${CMAKE_CURRENT_SOURCE_DIR}/../config/*")
add_executable(ad_census ${source_file})
target_link_libraries(ad_census ${OpenCV_LIBRARIES})

