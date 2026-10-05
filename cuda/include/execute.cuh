#ifndef EXECUTE_CUH
#define EXECUTE_CUH

#define WARP_SIZE  32

texture<uint8_t, 1, cudaReadModeElementType> tex1;
texture<uint8_t, 1, cudaReadModeElementType> tex2;

__constant__ uint8_t dev_window_height;
__constant__ uint8_t dev_window_width;
__constant__ float dev_lamda_ad;
__constant__ float dev_lamda_census;

#define DEV_CENSUS_HEIGHT  5
#define DEV_CENSUS_WIDTH   5
#define DEV_LAMDA_AD       30
#define DEV_LAMDA_CENSUS   10
#define DEV_CROSS_T1       20
#define DEV_CROSS_T2       6
#define DEV_CROSS_L1       34
#define DEV_CROSS_L2       17
#define DEV_CROSS_ARM_LEN  255
#define DEV_MAX_DIS        210
#define AGGREGATE_ITER     4
#define DEV_TSO            15
#define DEV_P1             1.0
#define DEV_P2             3.0

__constant__ uint8_t dev_t1, dev_t2;
__constant__ int dev_l1, dev_l2;
__constant__ int dev_cross_arm_max_len;
__constant__ uint8_t dev_aggregate_iter;

#endif