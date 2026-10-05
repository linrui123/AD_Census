#include "../include/cu_ad_census.hpp"

CuADCensus::CuADCensus()
{
}

CuADCensus::CuADCensus(const std::string & config_file)
{
  parse_config(config_file);
}

CuADCensus::~CuADCensus()
{
  delete m_left_init_costs;
  delete m_right_init_costs;
  delete m_left_cross_arms;
  delete m_right_cross_arms;
  delete m_left_arms_pixel_cnt;
  delete m_right_arms_pixel_cnt;
  delete m_left_aggregate_costs;
  delete m_right_aggregate_costs;
}

void CuADCensus::parse_config(const std::string & config_file)
{
  m_opt.parse_config(config_file);

  m_left_image = cv::imread(m_opt.m_left_image_name, cv::IMREAD_COLOR);
  m_right_image = cv::imread(m_opt.m_right_image_name, cv::IMREAD_COLOR);

  assert(m_left_image.rows == m_right_image.rows);
  assert(m_left_image.cols == m_right_image.cols);
  assert(m_left_image.channels() == m_right_image.channels());

  cv::resize(m_left_image, m_left_image, cv::Size(m_opt.m_resize_width, m_opt.m_resize_height));
  cv::resize(m_right_image, m_right_image, cv::Size(m_opt.m_resize_width, m_opt.m_resize_height));

  m_rows = m_left_image.rows;
  m_cols = m_left_image.cols;
  m_channels = m_left_image.channels();

  m_left_init_costs = new float[m_rows * m_cols * m_opt.m_max_disparity];
  m_right_init_costs = new float[m_rows * m_cols * m_opt.m_max_disparity];
  
  m_left_cross_arms = new CuCrossArm[m_rows * m_cols];
  m_right_cross_arms = new CuCrossArm[m_rows * m_cols];

  m_left_arms_pixel_cnt = new CuArmPixelCnt[m_rows * m_cols];
  m_right_arms_pixel_cnt = new CuArmPixelCnt[m_rows * m_cols];

  m_left_aggregate_costs = new float[m_rows * m_cols * m_opt.m_max_disparity];
  m_right_aggregate_costs = new float[m_rows * m_cols * m_opt.m_max_disparity];

  m_left_disparity_image = cv::Mat(m_rows, m_cols, CV_8UC1);
  m_right_disparity_image = cv::Mat(m_rows, m_cols, CV_8UC1);

  std::cout << "========== AD-Census CONFIG ==========" << std::endl;
  std::cout << "left_image_name:" << m_opt.m_left_image_name << std::endl;
  std::cout << "right_image_name:" << m_opt.m_right_image_name << std::endl;
  std::cout << "groudtruth:" << m_opt.m_groudtruth << std::endl;
  std::cout << "separation:" << m_opt.m_separation << std::endl;
  std::cout << "max_disparity:" << m_opt.m_max_disparity << std::endl;
  std::cout << "census_size_width:" << m_opt.m_census_size_width << std::endl;
  std::cout << "census_size_height:" << m_opt.m_census_size_height << std::endl;
  std::cout << "cross_arm_max_len:" << m_opt.m_cross_arm_max_len << std::endl;
  std::cout << "lamda_ad:" << m_opt.lamda_ad << std::endl;
  std::cout << "lamda_census:" << m_opt.lamda_census << std::endl;
  std::cout << "cross_t1:" << m_opt.cross_t1 << std::endl;
  std::cout << "cross_t2:" << m_opt.cross_t2 << std::endl;
  std::cout << "cross_l1:" << m_opt.cross_l1 << std::endl;
  std::cout << "cross_l2:" << m_opt.cross_l2 << std::endl;
  std::cout << "tso:" << m_opt.tso << std::endl;
  std::cout << "P1:" << m_opt.p1 << std::endl;
  std::cout << "P2:" << m_opt.p2 << std::endl;
  std::cout << "aggregate_iter:" << m_opt.aggregate_iter << std::endl;
  std::cout << "disparity_cost_type:" << m_opt.cost_type << std::endl;
  std::cout << "is_resize:" << m_opt.is_resize << std::endl;
  std::cout << "resize_height:" << m_opt.m_resize_height << std::endl;
  std::cout << "resize_width:" << m_opt.m_resize_width << std::endl; 
  std::cout << "======================================" << std::endl;
}