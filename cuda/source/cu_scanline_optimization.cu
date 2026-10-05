#include "cu_ad_census.hpp"
#include "../include/execute.cuh"

extern cudaStream_t streams[2];
extern float * dev_left_costs;
extern float * dev_right_costs;
extern float * dev_left_aggregate_costs;
extern float * dev_right_aggregate_costs;
extern uint8_t * dev_image1;
extern uint8_t * dev_image2;

static __device__ uint8_t color_distance(
  const int row1, const int col1, 
  const int row2, const int col2,
  bool left_or_right)
{
  int i;
  uint8_t max_color_dis = 0;
  uint8_t color1;
  uint8_t color2;

  for (i = 0; i < 3; i++)
  {
    if (left_or_right == true)
    {
      color1 = tex1Dfetch(tex1, row1 * gridDim.y * 3 + col1 * 3 + i);
      color2 = tex1Dfetch(tex1, row2 * gridDim.y * 3 + col2 * 3 + i);
    }
    else
    {
      color1 = tex1Dfetch(tex2, row1 * gridDim.y * 3 + col1 * 3 + i);
      color2 = tex1Dfetch(tex2, row2 * gridDim.y * 3 + col2 * 3 + i);
    }
    
    if (max_color_dis < abs(color1 - color2))
    {
      max_color_dis = abs(color1 - color2);
    }
  }

  return max_color_dis;
}

__global__ void scanline_optimization_inner(float * init_costs, float * aggr_costs)
{
  int tid = threadIdx.x;

  int idx;
  float cost = FLT_MAX;

  if ((tid % 256) < DEV_MAX_DIS)
  {
    switch (tid / 256)
    {
    case 0: // up
      if (blockIdx.x >= 1)
      {
        idx = (blockIdx.x - 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid / 4;
        cost = init_costs[idx];
      }
      break;
    case 1: // left
      if (blockIdx.y >= 1)
      {
        idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y - 1) * DEV_MAX_DIS + tid / 4;
        cost = init_costs[idx];
      }
      break;
    case 2: // down
      if (blockIdx.x + 1 < gridDim.x)
      {
        idx = (blockIdx.x + 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid / 4;
        cost = init_costs[idx];
      }
      break;
    case 3: // right
      if (blockIdx.y + 1 < gridDim.y)
      {
        idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y + 1) * DEV_MAX_DIS + tid / 4;
        cost = init_costs[idx];
      }
      break;
    }
  }

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  __shared__ float min_costs[32];

#pragma unroll 
  for (int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    float temp = __shfl_down_sync(0xffffffff, cost, stride);
    cost = MIN(cost, temp);
  }

  if (lane_id == 0)
  {
    min_costs[lane_id] = cost;
  }

  __syncthreads();

  if (tid < 32)
  {
    cost = min_costs[tid];

#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1)
    {
      float temp = __shfl_down_sync(0xff, cost, stride, 8);
      cost = MIN(cost, temp);
    }

    if ((tid % 8) == 0)
    {
      min_costs[tid / 8] = cost;
    }
  }

  __syncthreads();

  if (tid < DEV_MAX_DIS)
  {
    uint8_t dl1, dl2, dl3, dl4;
    uint8_t dr1, dr2, dr3, dr4;
    float p11, p21, p12, p22, p13, p23, p14, p24;

    dl1 = (blockIdx.x >= 1) ? 
            color_distance(blockIdx.x, blockIdx.y, blockIdx.x - 1, blockIdx.y, true) : DEV_TSO;
    dl2 = (blockIdx.y >= 1) ? 
            color_distance(blockIdx.x, blockIdx.y, blockIdx.x, blockIdx.y - 1, true) : DEV_TSO;
    dl3 = (blockIdx.x + 1 < gridDim.x) ? 
            color_distance(blockIdx.x, blockIdx.y, blockIdx.x + 1, blockIdx.y, true) : DEV_TSO;
    dl4 = (blockIdx.y + 1 < gridDim.y) ?
            color_distance(blockIdx.x, blockIdx.y, blockIdx.x, blockIdx.y + 1, true) : DEV_TSO;

    dr1 = ((blockIdx.x >= 1) && (blockIdx.y >= tid)) ? 
            color_distance(blockIdx.x, blockIdx.y - tid, blockIdx.x - 1, blockIdx.y - tid, false) : DEV_TSO;
    dl2 = ((blockIdx.y >= 1) && (blockIdx.y >= tid)) ? 
            color_distance(blockIdx.x, blockIdx.y - tid, blockIdx.x, blockIdx.y - 1 - tid, true) : DEV_TSO;
    dl3 = ((blockIdx.x + 1 < gridDim.x) && (blockIdx.y >= tid)) ? 
            color_distance(blockIdx.x, blockIdx.y - tid, blockIdx.x + 1, blockIdx.y - tid, true) : DEV_TSO;
    
    dl4 = ((blockIdx.y + 1 < gridDim.y) && (blockIdx.y >= tid)) ?
            color_distance(blockIdx.x, blockIdx.y - tid, blockIdx.x, blockIdx.y + 1 - tid, true) : DEV_TSO;

    if ((dl1 < DEV_TSO) && (dr1 < DEV_TSO))
    {
      p11 = DEV_P1; 
      p21 = DEV_P2;
    }
    else if (((dl1 < DEV_TSO) && (dr1 >= DEV_TSO)) || ((dl1 >= DEV_TSO) && (dr1 < DEV_TSO)))
    {
      p11 = DEV_P1 / 4;
      p21 = DEV_P2 / 4;
    }
    else if ((dl1 >= DEV_TSO) && (dr1 >= DEV_TSO))
    {
      p11 = DEV_P1 / 10;
      p21 = DEV_P2 / 10;
    }

    if ((dl2 < DEV_TSO) && (dr2 < DEV_TSO))
    {
      p12 = DEV_P1; 
      p22 = DEV_P2;
    }
    else if (((dl2 < DEV_TSO) && (dr2 >= DEV_TSO)) || ((dl2 >= DEV_TSO) && (dr2 < DEV_TSO)))
    {
      p12 = DEV_P1 / 4;
      p22 = DEV_P2 / 4;
    }
    else if ((dl2 >= DEV_TSO) && (dr2 >= DEV_TSO))
    {
      p12 = DEV_P1 / 10;
      p22 = DEV_P2 / 10;
    }

    if ((dl3 < DEV_TSO) && (dr3 < DEV_TSO))
    {
      p13 = DEV_P1; 
      p23 = DEV_P2;
    }
    else if (((dl3 < DEV_TSO) && (dr3 >= DEV_TSO)) || ((dl3 >= DEV_TSO) && (dr3 < DEV_TSO)))
    {
      p13 = DEV_P1 / 4;
      p23 = DEV_P2 / 4;
    }
    else if ((dl3 >= DEV_TSO) && (dr3 >= DEV_TSO))
    {
      p13 = DEV_P1 / 10;
      p23 = DEV_P2 / 10;
    }

    if ((dl4 < DEV_TSO) && (dr4 < DEV_TSO))
    {
      p14 = DEV_P1; 
      p24 = DEV_P2;
    }
    else if (((dl4 < DEV_TSO) && (dr4 >= DEV_TSO)) || ((dl4 >= DEV_TSO) && (dr4 < DEV_TSO)))
    {
      p14 = DEV_P1 / 4;
      p24 = DEV_P2 / 4;
    }
    else if ((dl4 >= DEV_TSO) && (dr4 >= DEV_TSO))
    {
      p14 = DEV_P1 / 10;
      p24 = DEV_P2 / 10;
    }

    float c1 = 0.0; 
    float c2 = 0.0; 
    float c3 = 0.0; 
    float c4 = 0.0;
    
    idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid;
    float cost1 = init_costs[idx];
    float cost2 = 1000.0;
    float cost3 = 1000.0;
    float cost4 = 1000.0;

    if (blockIdx.x >= 1)
    {
      idx = (blockIdx.x - 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid;
      cost2 = init_costs[idx];
      if (tid >= 1)
      {
        idx = (blockIdx.x - 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + (tid - 1);
        cost3 = init_costs[idx];
      }
      if (tid + 1 < DEV_MAX_DIS)
      {
        idx = (blockIdx.x - 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + (tid + 1);
        cost4 = init_costs[idx];
      }
    }

    c1 = cost1 + MIN(MIN(cost2, cost3 + p11), MIN(cost4 + p11, min_costs[0] + p21)) - min_costs[0];

    cost2 = cost3 = cost4 = 1000.0;

    if (blockIdx.y >= 1)
    {
      idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y - 1) * DEV_MAX_DIS + tid;
      cost2 = init_costs[idx];
      if (tid >= 1)
      {
        idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y - 1) * DEV_MAX_DIS + (tid - 1);
        cost3 = init_costs[idx];
      }
      if (tid + 1 < DEV_MAX_DIS)
      {
        idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y - 1) * DEV_MAX_DIS + (tid + 1);
        cost4 = init_costs[idx];
      }
    }

    c2 = cost1 + MIN(MIN(cost2, cost3 + p12), MIN(cost4 + p12, min_costs[1] + p22)) - min_costs[1];

    cost2 = cost3 = cost4 = 1000.0;

    if (blockIdx.x + 1 < gridDim.x)
    {
      idx = (blockIdx.x + 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid;
      cost2 = init_costs[idx];
      if (tid >= 1)
      {
        idx = (blockIdx.x + 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + (tid - 1);
        cost3 = init_costs[idx];
      }
      if (tid + 1 < DEV_MAX_DIS)
      {
        idx = (blockIdx.x + 1) * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + (tid + 1);
        cost4 = init_costs[idx];
      }
    }

    c3 = cost1 + MIN(MIN(cost2, cost3 + p13), MIN(cost4 + p13, min_costs[2] + p23)) - min_costs[2];

    cost2 = cost3 = cost4 = 1000.0;

    if (blockIdx.y + 1 < gridDim.y)
    {
      idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y + 1) * DEV_MAX_DIS + tid;
      cost2 = init_costs[idx];
      if (tid >= 1)
      {
        idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y + 1) * DEV_MAX_DIS + (tid - 1);
        cost3 = init_costs[idx];
      }
      if (tid + 1 < DEV_MAX_DIS)
      {
        idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + (blockIdx.y + 1) * DEV_MAX_DIS + (tid + 1);
        cost4 = init_costs[idx];
      }
    }

    c4 = cost1 + MIN(MIN(cost2, cost3 + p14), MIN(cost4 + p14, min_costs[3] + p24)) - min_costs[3];

    idx = blockIdx.x * gridDim.y * DEV_MAX_DIS + blockIdx.y * DEV_MAX_DIS + tid;
    aggr_costs[idx] = (c1 + c2 + c3 + c4) / 4;
  }
}

void CuADCensus::scanline_optimization()
{
  dim3 block(1024);
  dim3 grid(m_rows, m_cols);
  
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  uint64_t t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  HANDLE_ERROR(cudaBindTexture(0, tex1, (const void *)dev_image1));
  HANDLE_ERROR(cudaBindTexture(0, tex2, (const void *)dev_image2));

  HANDLE_ERROR(cudaMemcpyAsync((void *)dev_left_costs,
                               (const void *)dev_left_aggregate_costs,
                               sizeof(float) * m_rows * m_cols * DEV_MAX_DIS,
                               cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                               streams[0]));

  scanline_optimization_inner<<<grid, block, 0, streams[0]>>>(dev_left_costs, dev_left_aggregate_costs);

  // HANDLE_ERROR(cudaMemcpyAsync((void *)m_left_aggregate_costs,
  //                              (const void *)dev_left_aggregate_costs,
  //                              sizeof(float) * m_rows * m_cols * m_opt.m_max_disparity,
  //                              cudaMemcpyKind::cudaMemcpyDeviceToHost,
  //                              streams[0]));

  // std::cout << m_left_aggregate_costs[0] << std::endl;

  cudaEventRecord(end);

  // cudaStreamSynchronize(streams[0]);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cout << "scanline optimizaion elapsed_time: " << elapsed_time << "ms" << std::endl;

  uint64_t t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "scanline optimizaion cpu time: " << (float)(t2 - t1) / 1000000 << " ms" << std::endl;
}