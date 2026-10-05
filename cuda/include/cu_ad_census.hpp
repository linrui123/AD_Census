#ifndef CU_AD_CENSUS_CUH
#define CU_AD_CENSUS_CUH

#include "cu_ad_census_type.hpp"
#include <cooperative_groups.h>
#include "cu_util.hpp"
#include "cuda_runtime.h"
#include "cuda_texture_types.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <opencv2/opencv.hpp>
#include "option.hpp"
#include <math.h>


class CuADCensus
{
  friend class Compare;

public:
  CuADCensus();
  CuADCensus(const std::string & config_file);
  ~CuADCensus();

  textureReference m_tex1;
  textureReference m_tex2;

  void parse_config(const std::string & config_file);
  
  void calculate_cost();
  void aggregate_cost();
  void scanline_optimization();
  void multistep_refine();
  void calculate_disparity(); 

  void init();
  void load_image();
  void execute();
  void release();

private:
  ADCensusOption m_opt;

  int m_rows;
  int m_cols;
  int m_channels;

  cv::Mat m_left_image;
  cv::Mat m_right_image;
  cv::Mat m_left_disparity_image;
  cv::Mat m_right_disparity_image;

  float *m_left_init_costs;
  float *m_right_init_costs;
  CuCrossArm *m_left_cross_arms;
  CuCrossArm *m_right_cross_arms;
  CuArmPixelCnt *m_left_arms_pixel_cnt;
  CuArmPixelCnt *m_right_arms_pixel_cnt;

  float *m_left_aggregate_costs;
  float *m_right_aggregate_costs;
  uint8_t * m_dev_left_disparity_image;
};


#endif