#include "cu_ad_census.hpp"
#include "../include/execute.cuh"

extern cudaStream_t streams[2];
extern float * dev_left_costs;
extern float * dev_right_costs;
extern float * dev_left_aggregate_costs;
extern float * dev_right_aggregate_costs;

__global__ void calculate_disparity_inner(float * costs, uint8_t * disparity)
{
  int tid = threadIdx.x;
  int idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid;
  __shared__ float min_costs[8];
  __shared__ int min_dis[8];

  int dis = tid;
  float cost = (tid < DEV_MAX_DIS) ? costs[idx] : FLT_MAX;

  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

#pragma unroll
  for (int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    float cost_temp = __shfl_down_sync(0xffffffff, cost, stride);
    int dis_temp = __shfl_down_sync(0xffffffff, dis, stride);
    if (cost > cost_temp)
    {
      cost = cost_temp;
      dis = dis_temp;
    }
  }

  if (lane_id == 0)
  {
    min_costs[warp_id] = cost;
    min_dis[warp_id] = dis;
  }

  __syncthreads();

  if (tid < 8)
  {
    cost = min_costs[tid];
    dis = min_dis[tid];

#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1)
    {
      float cost_temp = __shfl_down_sync(0xff, cost, stride);
      int dis_temp = __shfl_down_sync(0xff, dis, stride);
      if (cost > cost_temp)
      {
        cost = cost_temp;
        dis = dis_temp;
      }
    }

    if (tid == 0)
    {
      idx = blockIdx.x * gridDim.y + blockIdx.y;
      disparity[idx] = (uint8_t)dis;
    }
  }
}

void CuADCensus::calculate_disparity()
{
  dim3 grid(m_rows, m_cols);
  dim3 block(256);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  uint64_t t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  calculate_disparity_inner<<<grid, block, 0, streams[0]>>>(dev_left_aggregate_costs, m_dev_left_disparity_image);

  HANDLE_ERROR(cudaMemcpyAsync((void *)m_left_disparity_image.data,
                               (const void *)m_dev_left_disparity_image,
                               sizeof(uint8_t) * m_rows * m_cols, 
                               cudaMemcpyKind::cudaMemcpyDeviceToHost,
                               streams[0]));

  cudaEventRecord(end);

  cudaStreamSynchronize(streams[0]);
  cudaStreamSynchronize(streams[1]);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cout << "calculate disparity elapsed_time: " << elapsed_time << "ms" << std::endl;

  uint64_t t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "calculate disparity cpu time: " << (float)(t2 - t1) / 1000000 << " ms" << std::endl;

  cv::imwrite("cu_disparity.jpg", m_left_disparity_image);
}