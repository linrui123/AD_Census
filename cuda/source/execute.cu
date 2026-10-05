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

cudaStream_t streams[2];
float * dev_left_costs;
float * dev_right_costs;
float * dev_left_aggregate_costs;
float * dev_right_aggregate_costs;
uint8_t * dev_image1;
uint8_t * dev_image2;
texture<float, 1, cudaReadModeElementType> costs_tex1;
texture<float, 1, cudaReadModeElementType> costs_tex2;

// __constant__ uint8_t /*dev_t1,*/ dev_t2;
// __constant__ int dev_l1, dev_l2;
// __constant__ int dev_cross_arm_max_len;
// __constant__ uint8_t dev_aggregate_iter;
__constant__ uint16_t dev_max_disparity;
CuCrossArm * dev_left_cross_arms;
CuCrossArm * dev_right_cross_arms;
CuArmPixelCnt * dev_left_arms_pixel_cnts;
CuArmPixelCnt * dev_right_arms_pixel_cnts;

void CuADCensus::init(void)
{
  for (int i = 0; i < 2; i++)
  {
    HANDLE_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }
  
  HANDLE_ERROR(cudaMalloc((void **)&dev_left_costs, 
                          sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity));
  HANDLE_ERROR(cudaMalloc((void **)&dev_right_costs, 
                          sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity));
  HANDLE_ERROR(cudaMalloc((void **)&dev_left_aggregate_costs,
                          sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity));
  HANDLE_ERROR(cudaMalloc((void **)&dev_right_aggregate_costs,
                          sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity));
  HANDLE_ERROR(cudaMalloc((void **)&dev_left_cross_arms, 
                          sizeof(CuCrossArm) * m_rows * m_cols));
  HANDLE_ERROR(cudaMalloc((void **)&dev_right_cross_arms,
                          sizeof(CuCrossArm) * m_rows * m_cols));
  HANDLE_ERROR(cudaMalloc((void **)&dev_image1, 
                          sizeof(uint8_t) * m_rows * m_cols * m_channels));
  HANDLE_ERROR(cudaMalloc((void **)&dev_image2, 
                          sizeof(uint8_t) * m_rows * m_cols * m_channels));
  HANDLE_ERROR(cudaMalloc((void **)&dev_left_arms_pixel_cnts,
                          sizeof(CuArmPixelCnt) * m_rows * m_cols));
  HANDLE_ERROR(cudaMalloc((void **)&dev_right_arms_pixel_cnts, 
                          sizeof(CuArmPixelCnt) * m_rows * m_cols));
  HANDLE_ERROR(cudaMalloc((void **)&m_dev_left_disparity_image,
                          sizeof(uint8_t) * m_rows * m_cols));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_window_height, 
                                       (const void *)&m_opt.m_census_size_height, 
                                       sizeof(uint8_t), 
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_window_width, 
                                       (const void *)&m_opt.m_census_size_width,
                                       sizeof(uint8_t),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_lamda_ad,
                                       (const void *)&m_opt.lamda_ad,
                                       sizeof(float),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_lamda_census,
                                       (const void *)&m_opt.lamda_census,
                                       sizeof(float),
                                       0, 
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_t1,
                                       (const void *)&m_opt.cross_t1,
                                       sizeof(uint8_t),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_t2,
                                       (const void *)&m_opt.cross_t2,
                                       sizeof(uint8_t),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[1]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_l1,
                                       (const void *)&m_opt.cross_l1,
                                       sizeof(int),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_l2,
                                       (const void *)&m_opt.cross_l2,
                                       sizeof(int),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[1]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_cross_arm_max_len,
                                       (const void *)&m_opt.m_cross_arm_max_len,
                                       sizeof(int),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_aggregate_iter,
                                       (const void *)&m_opt.aggregate_iter,
                                       sizeof(uint8_t),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[1]));
  HANDLE_ERROR(cudaMemcpyToSymbolAsync((const void *)&dev_max_disparity,
                                       (const void *)&m_opt.m_max_disparity,
                                       sizeof(uint16_t),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       streams[0]));

  HANDLE_ERROR(cudaStreamSynchronize(streams[0]));
  HANDLE_ERROR(cudaStreamSynchronize(streams[1]));
}

void CuADCensus::load_image()
{
}

void CuADCensus::execute(void)
{
  calculate_cost();
  aggregate_cost();
  scanline_optimization();
  calculate_disparity();
}

void CuADCensus::release(void)
{
  cudaUnbindTexture(tex1);
  cudaUnbindTexture(tex2);
  cudaFree((void *)dev_image1);
  cudaFree((void *)dev_image2);
  cudaFree((void *)dev_left_costs);
  cudaFree((void *)dev_right_costs);
  cudaFree((void *)dev_left_aggregate_costs);
  cudaFree((void *)dev_right_aggregate_costs);
  cudaFree((void *)dev_left_cross_arms);
  cudaFree((void *)dev_right_cross_arms);
  cudaFree((void *)dev_left_arms_pixel_cnts);
  cudaFree((void *)dev_right_arms_pixel_cnts);
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
}