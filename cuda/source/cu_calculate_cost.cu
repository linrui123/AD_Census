#include <iostream>
#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_types.h"
#include "cuda_texture_types.h"

#include "../include/cu_ad_census.hpp"

#include "sys/time.h"

#include "../include/execute.cuh"

extern cudaStream_t streams[2];
extern float * dev_left_costs;
extern float * dev_right_costs;
extern uint8_t * dev_image1;
extern uint8_t * dev_image2;

__device__ static float ad_cost(
  const int row1, const int col1,
  const int row2, const int col2,
  const int height, const int width
)
{
  float ad_cost = 0;
  uint8_t i;
  uint8_t left_pixel_rgb;
  uint8_t right_pixel_rgb;
  
  for (i = 0; i < 3; i++)
  {
    left_pixel_rgb = tex1Dfetch(tex1, row1 * width * 3 + col1 * 3 + i);

    right_pixel_rgb = tex1Dfetch(tex2, row2 * width * 3 + col2 * 3 + i);

    ad_cost += abs(left_pixel_rgb - right_pixel_rgb);
  }
  
  return ad_cost / 3;
}

__device__ static float census_cost(
  const int row1, const int col1,
  const int row2, const int col2,
  const int height, const int width
)
{
  float census_cost = 0;
  int i = 0;
  int j = 0;
  uint8_t compare = 0;
  float pixel_gray;

  uint8_t r1 = tex1Dfetch(tex1, row1 * width * 3 + col1 * 3);
  uint8_t g1 = tex1Dfetch(tex1, row1 * width * 3 + col1 * 3 + 1);
  uint8_t b1 = tex1Dfetch(tex1, row1 * width * 3 + col1 * 3 + 2);
  float mid_pixel_gray1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;

  uint8_t r2 = tex1Dfetch(tex2, row2 * width * 3 + col2 * 3);
  uint8_t g2 = tex1Dfetch(tex2, row2 * width * 3 + col2 * 3 + 1);
  uint8_t b2 = tex1Dfetch(tex2, row2 * width * 3 + col2 * 3 + 2);
  float mid_pixel_gray2 = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2;

  bool hanming1, hanming2;
  for (i = -DEV_CENSUS_HEIGHT / 2; i <= DEV_CENSUS_HEIGHT / 2; i++)
  {
    for (j = -DEV_CENSUS_WIDTH / 2; j <= DEV_CENSUS_WIDTH / 2; j++)
    {
      hanming1 = false;
      if ((row1 + i >= 0) && (row1 + i < height) && (col1 + j >= 0) && (col1 + j < width))
      {
        uint8_t r = tex1Dfetch(tex1, (row1 + i) * width * 3 + (col1 + j) * 3);
        uint8_t g = tex1Dfetch(tex1, (row1 + i) * width * 3 + (col1 + j) * 3 + 1);
        uint8_t b = tex1Dfetch(tex1, (row1 + i) * width * 3 + (col1 + j) * 3 + 2);
        pixel_gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        if (pixel_gray < mid_pixel_gray1)
        {
          hanming1 = true;
        }
      }

      hanming2 = false;
      if ((row2 + i >= 0) && (row2 + i < height) && (col2 + j >= 0) && (col2 + j < width))
      {
        uint8_t r = tex1Dfetch(tex2, (row2 + i) * width * 3 + (col2 + j) * 3);
        uint8_t g = tex1Dfetch(tex2, (row2 + i) * width * 3 + (col2 + j) * 3 + 1);
        uint8_t b = tex1Dfetch(tex2, (row2 + i) * width * 3 + (col2 + j) * 3 + 2);
        pixel_gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        
        if (pixel_gray < mid_pixel_gray2)
        {
          hanming2 = true;
        }
      }
      
      if (hanming1 != hanming2)
      {
        census_cost++;
      }
    }
  }

  return census_cost;
}

__global__ void calculate_left_cost(float * costs)
{
  int tid = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;

  float cost1;
  float cost2;

  if (tid < (blockDim.x) * (gridDim.x) * (gridDim.y))
  {
    if ((blockIdx.y - threadIdx.x < 0) 
        || (blockIdx.y - threadIdx.x >= gridDim.y)
        || (blockIdx.y < 0)
        || (blockIdx.y >= gridDim.y))
    {
      costs[tid] = 1.0;
    }
    else
    {
      cost1 = ad_cost(blockIdx.x, blockIdx.y,
                      blockIdx.x, blockIdx.y - threadIdx.x,
                      gridDim.x, gridDim.y);

      cost2 = census_cost(blockIdx.x, blockIdx.y,
                          blockIdx.x, blockIdx.y - threadIdx.x,
                          gridDim.x, gridDim.y);

      costs[tid] = 2 - exp(-cost1 / DEV_LAMDA_AD) - exp(-cost2 / DEV_LAMDA_CENSUS);
    }
  }
}

__global__ static void calculate_right_cost(float * costs)
{
  int tid = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;

  float cost1;
  float cost2;

  if (tid < (blockDim.x) * (gridDim.x) * (gridDim.y))
  {
    if ((blockIdx.y + threadIdx.x < 0) 
        || (blockIdx.y + threadIdx.x >= gridDim.y)
        || (blockIdx.y < 0)
        || (blockIdx.y >= gridDim.y))
    {
      costs[tid] = 1.0;
    }
    else
    {
      cost1 = ad_cost(blockIdx.x, blockIdx.y + threadIdx.x,
                      blockIdx.x, blockIdx.y,
                      gridDim.x, gridDim.y);

      cost2 = census_cost(blockIdx.x, blockIdx.y + threadIdx.x,
                          blockIdx.x, blockIdx.y,
                          gridDim.x, gridDim.y);

      costs[tid] = 2 - exp(-cost1 / DEV_LAMDA_AD) - exp(-cost2 / DEV_LAMDA_CENSUS);
    }
  }
}

void CuADCensus::calculate_cost()
{
  HANDLE_ERROR(cudaMemcpyAsync((void *)dev_image1, 
                               (const void *)m_left_image.data, 
                               m_rows * m_cols * m_channels, 
                               cudaMemcpyHostToDevice,
                               streams[0]));
  HANDLE_ERROR(cudaMemcpyAsync((void *)dev_image2, 
                               (const void *)m_right_image.data, 
                               m_rows * m_cols * m_channels, 
                               cudaMemcpyHostToDevice,
                               streams[1]));

  HANDLE_ERROR(cudaBindTexture(0, tex1, (const void *)dev_image1));
  HANDLE_ERROR(cudaBindTexture(0, tex2, (const void *)dev_image2));

  HANDLE_ERROR(cudaStreamSynchronize(streams[0]));
  HANDLE_ERROR(cudaStreamSynchronize(streams[1]));
 

  dim3 grid(m_left_image.rows, m_left_image.cols);
  dim3 block(m_opt.m_max_disparity);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  uint64_t t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  calculate_left_cost<<<grid, block, 0, streams[0]>>>(dev_left_costs);
  calculate_right_cost<<<grid, block, 0, streams[1]>>>(dev_right_costs);

  // HANDLE_ERROR(cudaMemcpyAsync((void *)m_left_init_costs,
  //                              (const void *)dev_left_costs,
  //                              sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity,
  //                              cudaMemcpyKind::cudaMemcpyDeviceToHost,
  //                              streams[0]));

  // HANDLE_ERROR(cudaMemcpyAsync((void *)m_right_init_costs,
  //                              (const void *)dev_right_costs,
  //                              sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity,
  //                              cudaMemcpyKind::cudaMemcpyDeviceToHost,
  //                              streams[1]));

  cudaEventRecord(end);

  // cudaStreamSynchronize(streams[0]);
  // cudaStreamSynchronize(streams[1]);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cout << "calculate cost elapsed_time: " << elapsed_time << "ms" << std::endl;

  uint64_t t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "calculate cost cpu time: " << (float)(t2 - t1) / 1000000 << " ms" << std::endl;
}
