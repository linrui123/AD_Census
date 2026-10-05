#ifndef AD_CENSUS_HPP
#define AD_CENSUS_HPP

#include <iostream>
#include <bitset>
#include "opencv2/opencv.hpp"
#include "ad_census_type.hpp"
#include "option.hpp"

class ADCensus
{
  friend class Compare;

  using costs_vector = std::vector<std::vector<std::vector<float>>>;

public:

  ADCensus();
  ADCensus(const std::string & config_file);
  ~ADCensus();
    
  void parse_config(const std::string & config_file);
  void calculate_cost();
  void aggregation_cost();
  void scanline_optimization();
  void multistep_refine();

  void draw_disparity_image(const std::string & file_name);

private:
  ADCensusOption m_opt;
    
  cv::Mat m_left_image;
  cv::Mat m_right_image;
    
  cv::Mat m_disparity_image;
  cv::Mat m_disparity_rgb;
  cv::Mat m_reconstruct_image;
    
  costs_vector m_left_disparity_costs;
  costs_vector m_right_disparity_costs;
  costs_vector m_left_aggregate_costs;
  costs_vector m_right_aggregate_costs;

  std::vector<std::vector<CrossArm>> m_left_cross_arms;
  std::vector<std::vector<CrossArm>> m_right_cross_arms;
  std::vector<std::vector<ArmPixelCount>> m_left_arms_pixel_cnts;
  std::vector<std::vector<ArmPixelCount>> m_right_arms_pixel_cnts;
  std::vector<std::vector<float>> m_left_final_disparity;
  std::vector<std::vector<float>> m_right_final_disparity;

  std::vector<std::pair<int, int>> m_occlusions;
  std::vector<std::pair<int, int>> m_mismatches;

  void costs_init(costs_vector & costs, int rows, int cols);

  /*--初始代价计算--*/
  float ad_cost(const cv::Mat & image1, const int & row1, const int & col1, 
                const cv::Mat & image2, const int & row2, const int & col2);
  float census_cost(const cv::Mat & image1, const int & row1, const int & col1, 
                    const cv::Mat & image2, const int & row2, const int & col2);
  void calculate_cost_inter(LeftOrRight flag);

  /*--代价聚合--*/
  uint8_t color_distance(const cv::Mat & image, const int & row1, const int & col1, const int & row2, const int & col2);
  int spatial_distance(const int & row1, const int & col1, const int & row2, const int & col2); 

  void find_horizontal_arm(const cv::Mat & image, const int & row1, const int & col1, int & left, int & right);
  void find_vertical_arm(const cv::Mat & image, const int & row1, const int & col1, int & top, int & bottom);
  void build_arms(LeftOrRight flag);
  void compute_arms_pixel_count(LeftOrRight flag);
  void aggregate_cost_in_arms(LeftOrRight flag);

  /*--扫描线优化--*/
  void scanline_optimize_left_right(LeftOrRight flag, ScanlineDirection scanline_dir);
  void scanline_optimize_up_down(LeftOrRight flag, ScanlineDirection scanline_dir);

  /*--多视差优化--*/
  void calculate_disparity(LeftOrRight flag);
  void outlier_detection();
  void iterative_region_voting();
  void proper_interpolation();
  void depth_discontinuity_adjustment();

  void median_filter(std::vector<std::vector<float>> & in, std::vector<std::vector<float>> & out, const int rows, const int cols, const int win_size);
};


#endif