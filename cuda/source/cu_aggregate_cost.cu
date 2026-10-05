#include "../include/cu_ad_census.hpp"
#include "../include/execute.cuh"

extern cudaStream_t streams[2];
// extern __constant__ uint8_t dev_t2;
// extern __constant__ int dev_l1, dev_l2;
// extern __constant__ int dev_cross_arm_max_len;
// extern __constant__ uint8_t dev_aggregate_iter;
extern __constant__ uint16_t dev_max_disparity;
// extern texture<uint8_t, 1, cudaReadModeElementType> tex1;
// extern texture<uint8_t, 1, cudaReadModeElementType> tex2;
extern uint8_t * dev_image1;
extern uint8_t * dev_image2;
extern texture<float, 1, cudaReadModeElementType> costs_tex1;
extern texture<float, 1, cudaReadModeElementType> costs_tex2;
extern CuCrossArm * dev_left_cross_arms;
extern CuCrossArm * dev_right_cross_arms;
extern CuArmPixelCnt * dev_left_arms_pixel_cnts;
extern CuArmPixelCnt * dev_right_arms_pixel_cnts;
extern float * dev_left_costs;
extern float * dev_right_costs;
extern float * dev_left_aggregate_costs;
extern float * dev_right_aggregate_costs;

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

__device__ static int spatial_distance(
  int row1, int col1,
  int row2, int col2)
{
  return sqrtf((row1 - row2) * (row1 - row2) + (col1 - col2) * (col1 - col2));
}

__device__ static bool judge(
  int offset_h, int offset_w, 
  int offset_h2, int offset_w2,
  bool left_or_right
)
{
  uint8_t color_dist = color_distance(blockIdx.x, blockIdx.y, 
                                      blockIdx.x + offset_h, blockIdx.y + offset_w, 
                                      left_or_right);
  uint8_t color_dist2 = color_distance(blockIdx.x + offset_h, blockIdx.y + offset_w,
                                       blockIdx.x + offset_h2, blockIdx.y + offset_w2,
                                       left_or_right);
  int spacial_dist = sqrtf(offset_h * offset_h + offset_w * offset_w);

  if (color_dist >= DEV_CROSS_T1)
  {
    return false;
  }
  
  if (color_dist2 >= DEV_CROSS_T1)
  {
    return false;
  }
  
  if (spacial_dist > DEV_CROSS_L1)
  {
    return false;
  }
  
  if (spacial_dist > DEV_CROSS_L2)
  {
    if (color_dist >= DEV_CROSS_T2)
    {
      return false;
    }
  }

  return true;
}

__global__ static void build_arms_left(
  CuCrossArm * left_cross_arms
)
{
  __shared__ int continuouses[8];

  int idx = blockIdx.x * gridDim.y + blockIdx.y;
  int tid = threadIdx.x;

  bool choose = false;

  if (tid == 0)
  {
    choose = true;
  }
  else
  {
    switch (blockIdx.z)
    {
    case 0:
      choose = (blockIdx.y < tid) ? false : judge(0, -tid,
                                                  0, -tid + 1, 
                                                  true);
      break;
    case 1:
      choose = (blockIdx.y + tid >= gridDim.y) ? false : judge(0, tid, 
                                                              0, tid - 1, 
                                                              true);
      break;
    case 2:
      choose = (blockIdx.x < tid) ? false : judge(-tid, 0, 
                                                  -tid + 1, 0,
                                                  true);
      break;
    case 3:
      choose = (blockIdx.x + tid >= gridDim.x) ? false : judge(tid, 0,
                                                              tid - 1, 0,
                                                              true);
      break;
    default:
      break;
    }
  }

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int continuous = (choose == true) ? 1 : 0;

#pragma unroll
  for (unsigned int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    int temp = __shfl_down_sync(0xffffffff, continuous, stride);
    if (continuous == stride)
    {
      continuous += temp;
    }
  }

  if (lane_id == 0)
  {
    continuouses[warp_id] = continuous;
  }

  __syncthreads();

  if (tid < 8)
  {
    continuous = continuouses[tid];

#pragma unroll
    for (int stride = 1; stride < 8; stride *= 2)
    {
      int temp = __shfl_down_sync(0xff, continuous, stride);
      if (continuous == (stride * WARP_SIZE))
      {
        continuous += temp;
      }
    }

    if (tid == 0)
    {
      if (continuous > 0)
      {
        continuous--;
      }

      switch (blockIdx.z)
      {
      case 0:
        left_cross_arms[idx].left = blockIdx.y - continuous;
        break;
      case 1:
        left_cross_arms[idx].right = blockIdx.y + continuous;
        break;
      case 2:
        left_cross_arms[idx].top = blockIdx.x - continuous;
        break;
      case 3:
        left_cross_arms[idx].bottom = blockIdx.x + continuous;
        break;
      default:
        break;
      }
    }
  }
}

__global__ static void build_arms_right(
  CuCrossArm * right_cross_arms
)
{
  __shared__ int continuouses[8];

  int idx = blockIdx.x * gridDim.y + blockIdx.y;
  int tid = threadIdx.x;

  bool choose = false;

  if (tid == 0)
  {
    choose = true;
  }
  else
  {
    switch (blockIdx.z)
    {
    case 0:
      choose = (blockIdx.y < tid) ? false : judge(0, -tid,
                                                  0, -tid + 1, 
                                                  false);
      break;
    case 1:
      choose = (blockIdx.y + tid >= gridDim.y) ? false : judge(0, tid, 
                                                              0, tid - 1, 
                                                              false);
      break;
    case 2:
      choose = (blockIdx.x < tid) ? false : judge(-tid, 0, 
                                                  -tid + 1, 0,
                                                  false);
      break;
    case 3:
      choose = (blockIdx.x + tid >= gridDim.x) ? false : judge(tid, 0,
                                                              tid - 1, 0,
                                                              false);
      break;
    default:
      break;
    }
  }

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int continuous = (choose == true) ? 1 : 0;

#pragma unroll
  for (int stride = 1; stride < WARP_SIZE; stride *= 2)
  {
    int temp = __shfl_down_sync(0xffffffff, continuous, stride);
    if (continuous == stride)
    {
      continuous += temp;
    }
  }

  if (lane_id == 0)
  {
    if ((tid == 0) && (continuous > 0))
    {
      continuous--;
    }
    continuouses[warp_id] = continuous;
  }

  __syncthreads();

  if (tid < 8)
  {
    continuous = continuouses[tid];

#pragma unroll
    for (int stride = 1; stride < 8; stride *= 2)
    {
      int temp = __shfl_down_sync(0xff, continuous, stride);
      if (continuous == (stride * WARP_SIZE))
      {
        continuous += temp;
      }
    }

    if (tid == 0)
    {
      switch (blockIdx.z)
      {
      case 0:
        right_cross_arms[idx].left = blockIdx.y - continuous;
        break;
      case 1:
        right_cross_arms[idx].right = blockIdx.y + continuous;
        break;
      case 2:
        right_cross_arms[idx].top = blockIdx.x - continuous;
        break;
      case 3:
        right_cross_arms[idx].bottom = blockIdx.x + continuous;
        break;
      default:
        break;
      }
    }
  }
}

__global__ static void compute_arms_pixel_count(
  CuCrossArm * cross_arms,
  CuArmPixelCnt * arms_pixel_cnts
)
{
  int tid = threadIdx.x;
  int idx = blockIdx.x * gridDim.y + blockIdx.y;

  CuCrossArm arm = cross_arms[idx];
  __shared__ int cnts[8];

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int vertical_idx = arm.top + tid;
  int sum = 0;

  if (vertical_idx <= arm.bottom)
  {
    CuCrossArm arm2 = cross_arms[vertical_idx * gridDim.y + blockIdx.y];
    sum = arm2.right - arm2.left + 1;
  }

#pragma unroll
  for (unsigned int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  }

  if (lane_id == 0)
  {
    cnts[warp_id] = sum;
  }

  __syncthreads();

  if (tid < 8)
  {
    sum = cnts[tid];

#pragma unroll
    for (unsigned int stride = 1; stride < 8; stride <<= 1)
    {
      sum += __shfl_down_sync(0xff, sum, stride);
    }

    if (tid == 0)
    {
      arms_pixel_cnts[idx].cnts[0] = sum;
    }
  }


  int horizon_idx = arm.left + tid;
  sum = 0;

  if (horizon_idx <= arm.right)
  {
    CuCrossArm arm2 = cross_arms[blockIdx.x * gridDim.y + horizon_idx];
    sum = arm2.bottom - arm2.top + 1;
  }

#pragma unroll
  for (unsigned int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  }

  if (lane_id == 0)
  {
    cnts[warp_id] = sum;
  }

  __syncthreads();

  if (tid < 8)
  {
    sum = cnts[tid];

#pragma unroll
    for (unsigned int stride = 1; stride < 8; stride <<= 1)
    {
      sum += __shfl_down_sync(0xff, sum, stride);
    }

    if (tid == 0)
    {
      arms_pixel_cnts[idx].cnts[1] = sum;
    }
  }
}

__device__ static void aggregate_costs_left_to_right(
  CuCrossArm * cross_arm,
  float * src_costs,
  float * dst_costs,
  float * costs_temp,
  int cnt
)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  int tid = threadIdx.x;
  int idx_dis = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;

  float cost = 0;

  int arm_idx = cross_arm->left + tid;
  if (arm_idx <= cross_arm->right)
  {
    int cost_idx = blockIdx.x * gridDim.y * gridDim.z + arm_idx * gridDim.z + blockIdx.z;
    cost = src_costs[cost_idx];
  }

  #pragma unroll
  for (int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    cost += __shfl_down_sync(0xffffffff, cost, stride);
  }

  if (lane_id == 0)
  {
    costs_temp[warp_id] = cost;
  }

  __syncthreads();

  if (tid < 8)
  {
    cost = costs_temp[tid];

    #pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1)
    {
      cost += __shfl_down_sync(0xff, cost, stride);
    }

    if (tid == 0)
    {
      dst_costs[idx_dis] = cost / cnt;
    }
  }

  // grid_sync();
}

__device__ static void aggregate_costs_top_to_bottom(
  CuCrossArm * cross_arm,
  float * src_costs,
  float * dst_costs,
  float * costs_temp,
  int cnt
)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  int tid = threadIdx.x;
  int idx_dis = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;

  float cost = 0;

  int arm_idx = cross_arm->top + tid;
  if (arm_idx <= cross_arm->bottom)
  {
    int cost_idx = arm_idx * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    cost = src_costs[cost_idx];
  }

  #pragma unroll
  for (int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    cost += __shfl_down_sync(0xffffffff, cost, stride);
  }

  if (lane_id == 0)
  {
    costs_temp[warp_id] = cost;
  }

  __syncthreads();

  if (tid < 8)
  {
    cost = costs_temp[tid];

    #pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1)
    {
      cost += __shfl_down_sync(0xff, cost, stride);
    }

    if (tid == 0)
    {
      dst_costs[idx_dis] = cost / cnt;
    }
  }

  // grid_sync();
}

__global__ static void aggregate_costs_inner(
  CuCrossArm * cross_arms,
  CuArmPixelCnt * arms_pixel_cnts,
  float * init_costs,
  float * aggregate_costs)
{
  int idx = blockIdx.x * gridDim.y + blockIdx.y;
  int idx_dis = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
  int tid = threadIdx.x;

  CuCrossArm arm = cross_arms[idx];
  CuArmPixelCnt pixel_cnt = arms_pixel_cnts[idx];

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  __shared__ float costs_temp[8];

  bool horizon_first = true;

  for (int i = 0; i < AGGREGATE_ITER; i++)
  {
    if (horizon_first == true)
    {
      aggregate_costs_left_to_right(&arm, aggregate_costs, init_costs, costs_temp, 1);
      aggregate_costs_top_to_bottom(&arm, init_costs, aggregate_costs, costs_temp, pixel_cnt.cnts[0]);
    }
    else
    {
      aggregate_costs_top_to_bottom(&arm, aggregate_costs, init_costs, costs_temp, 1);
      aggregate_costs_left_to_right(&arm, init_costs, aggregate_costs, costs_temp, pixel_cnt.cnts[1]);
    }

    horizon_first = !horizon_first;
  }
}

__global__ static void aggregate_costs_left_to_right_inner(
  CuCrossArm * cross_arm,
  CuArmPixelCnt * cross_arm_pixels,
  float * src_costs,
  float * dst_costs
)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  int tid = threadIdx.x;
  int idx = blockIdx.x * gridDim.y + blockIdx.y;
  int idx_dis = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;

  CuCrossArm arm = cross_arm[idx];
  int arm_cnt = 1;
  if (cross_arm_pixels != NULL)
  {
    arm_cnt = cross_arm_pixels[idx].cnts[1];
  }

  __shared__ float costs_temp[8];
  float cost = 0;

  int arm_idx = arm.left + tid;
  if (arm_idx <= arm.right)
  {
    int cost_idx = blockIdx.x * gridDim.y * gridDim.z + arm_idx * gridDim.z + blockIdx.z;
    cost = src_costs[cost_idx];
  }

#pragma unroll
  for (unsigned int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    cost += __shfl_down_sync(0xffffffff, cost, stride);
  }

  if (lane_id == 0)
  {
    costs_temp[warp_id] = cost;
  }

  __syncthreads();

  if (tid < 8)
  {
    cost = costs_temp[tid];

#pragma unroll
    for (unsigned int stride = 1; stride < 8; stride <<= 1)
    {
      cost += __shfl_down_sync(0xff, cost, stride);
    }

    if (tid == 0)
    {
      dst_costs[idx_dis] = cost / arm_cnt;
    }
  }
}

__global__ static void aggregate_costs_top_to_bottom_inner(
  CuCrossArm * cross_arm,
  CuArmPixelCnt * cross_arm_pixels,
  float * src_costs,
  float * dst_costs
)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  int tid = threadIdx.x;
  int idx = blockIdx.x * gridDim.y + blockIdx.y;
  int idx_dis = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;

  CuCrossArm arm = cross_arm[idx];
  int arm_cnt = 1;
  if (cross_arm_pixels != NULL)
  {
    arm_cnt = cross_arm_pixels[idx].cnts[0];
  }

  __shared__ float costs_temp[8];
  float cost = 0;

  int arm_idx = arm.top + tid;
  if (arm_idx <= arm.bottom)
  {
    int cost_idx = arm_idx * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    cost = src_costs[cost_idx];
  }

#pragma unroll
  for (unsigned int stride = 1; stride < WARP_SIZE; stride <<= 1)
  {
    cost += __shfl_down_sync(0xffffffff, cost, stride);
  }

  if (lane_id == 0)
  {
    costs_temp[warp_id] = cost;
  }

  __syncthreads();

  if (tid < 8)
  {
    cost = costs_temp[tid];

#pragma unroll
    for (unsigned int stride = 1; stride < 8; stride <<= 1)
    {
      cost += __shfl_down_sync(0xff, cost, stride);
    }

    if (tid == 0)
    {
      dst_costs[idx_dis] = cost / arm_cnt;
    }
  }
}

void CuADCensus::aggregate_cost(void)
{
  dim3 grid(m_rows, m_cols, 4);
  dim3 block((DEV_CROSS_ARM_LEN >= 256) ? 256 : DEV_CROSS_ARM_LEN);
  dim3 grid2(m_rows, m_cols);
  dim3 grid3(m_rows, m_cols, DEV_MAX_DIS);

  HANDLE_ERROR(cudaBindTexture(0, tex1, (const void *)dev_image1));
  HANDLE_ERROR(cudaBindTexture(0, tex2, (const void *)dev_image2));

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  uint64_t t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  HANDLE_ERROR(cudaMemcpyAsync((void *)dev_left_aggregate_costs,
                               (const void *)dev_left_costs,
                               sizeof(float) * m_rows * m_cols * DEV_MAX_DIS,
                               cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                               streams[0]));
  HANDLE_ERROR(cudaMemcpyAsync((void *)dev_right_aggregate_costs,
                               (const void *)dev_right_costs,
                               sizeof(float) * m_rows * m_cols * DEV_MAX_DIS,
                               cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                               streams[1]));
  HANDLE_ERROR(cudaMemsetAsync((void *)dev_left_arms_pixel_cnts,
                               0,
                               sizeof(CuArmPixelCnt) * m_rows * m_cols,
                               streams[0]));
  HANDLE_ERROR(cudaMemsetAsync((void *)dev_right_arms_pixel_cnts,
                               0,
                               sizeof(CuArmPixelCnt) * m_rows * m_cols,
                               streams[1]));

  build_arms_left<<<grid, block, 0, streams[0]>>>(dev_left_cross_arms);
  compute_arms_pixel_count<<<grid2, block, 0, streams[0]>>>(dev_left_cross_arms, dev_left_arms_pixel_cnts);
  for (uint8_t i = 0; i < m_opt.aggregate_iter; i++)
  {
    if ((i & 0x01) == 0)
    {
      aggregate_costs_left_to_right_inner<<<grid3, block, 0, streams[0]>>>(dev_left_cross_arms, NULL, dev_left_aggregate_costs, dev_left_costs);
      aggregate_costs_top_to_bottom_inner<<<grid3, block, 0, streams[0]>>>(dev_left_cross_arms, dev_left_arms_pixel_cnts, dev_left_costs, dev_left_aggregate_costs);
    }
    else
    {
      aggregate_costs_top_to_bottom_inner<<<grid3, block, 0, streams[0]>>>(dev_left_cross_arms, NULL, dev_left_aggregate_costs, dev_left_costs);
      aggregate_costs_left_to_right_inner<<<grid3, block, 0, streams[0]>>>(dev_left_cross_arms, dev_left_arms_pixel_cnts, dev_left_costs, dev_left_aggregate_costs);
    }
  }

  build_arms_right<<<grid, block, 0, streams[1]>>>(dev_right_cross_arms);
  compute_arms_pixel_count<<<grid2, block, 0, streams[1]>>>(dev_right_cross_arms, dev_right_arms_pixel_cnts);
  for (uint8_t i = 0; i < m_opt.aggregate_iter; i++)
  {
    if ((i & 0x01) == 0)
    {
      aggregate_costs_left_to_right_inner<<<grid3, block, 0, streams[1]>>>(dev_right_cross_arms, NULL, dev_right_aggregate_costs, dev_right_costs);
      aggregate_costs_top_to_bottom_inner<<<grid3, block, 0, streams[1]>>>(dev_right_cross_arms, dev_right_arms_pixel_cnts, dev_right_costs, dev_right_aggregate_costs);
    }
    else
    {
      aggregate_costs_top_to_bottom_inner<<<grid3, block, 0, streams[1]>>>(dev_right_cross_arms, NULL, dev_right_aggregate_costs, dev_right_costs);
      aggregate_costs_left_to_right_inner<<<grid3, block, 0, streams[1]>>>(dev_right_cross_arms, dev_right_arms_pixel_cnts, dev_right_costs, dev_right_aggregate_costs);
    }
  }

  // HANDLE_ERROR(cudaMemcpyAsync((void *)m_left_aggregate_costs,
  //                              (const void *)dev_left_aggregate_costs,
  //                              sizeof(float) * m_rows * m_cols * DEV_MAX_DIS,
  //                              cudaMemcpyDeviceToHost,
  //                              streams[0]));
  // HANDLE_ERROR(cudaMemcpyAsync((void *)m_right_aggregate_costs,
  //                              (const void *)dev_right_aggregate_costs,
  //                              sizeof(float) * m_rows * m_cols * DEV_MAX_DIS,
  //                              cudaMemcpyDeviceToHost,
  //                              streams[1]));

  cudaEventRecord(end);

  // cudaStreamSynchronize(streams[0]);
  // cudaStreamSynchronize(streams[1]);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cout << "aggregate cost elapsed_time: " << elapsed_time << "ms" << std::endl;

  uint64_t t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "aggregate cost cpu time: " << (float)(t2 - t1) / 1000000 << " ms" << std::endl;

}