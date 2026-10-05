#ifndef OPTION_HPP
#define OPTION_HPP

#include <iostream>
#include <string>
#include <atomic>
#include <mutex>

class ADCensusOption
{
  friend class ADCensus;
  friend class CuADCensus;

protected:
  std::string m_left_image_name;
  std::string m_right_image_name;
  std::string m_groudtruth;
  bool m_separation;
  uint16_t m_max_disparity;
  uint8_t m_census_size_width;
  uint8_t m_census_size_height;
  int m_cross_arm_max_len;
  float lamda_ad;
  float lamda_census;
  uint8_t cross_t1;
  uint8_t cross_t2;
  int cross_l1;
  int cross_l2;
  int tso;
  float p1;
  float p2;
  uint8_t aggregate_iter;
  uint8_t cost_type;
  int is_resize;
  uint16_t m_resize_height;
  uint16_t m_resize_width;

public:
  ADCensusOption();
  ~ADCensusOption();
  void parse_config(const std::string & config_file);
};

#define SET_CONFIG(cmpr, opt, key, value) \
if (!key.compare(cmpr))\
{\
  opt = value;\
}

#endif
