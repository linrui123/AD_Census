#ifndef AD_CENSUS_TYPE_HPP
#define AD_CENSUS_TYPE_HPP

#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <vector>

// typedef struct ADCensusOption
// {
//   std::string m_left_image_name;
//   std::string m_right_image_name;
//   bool m_separation;
//   uint16_t m_max_disparity;
//   uint8_t m_census_size_width;
//   uint8_t m_census_size_height;
//   int m_cross_arm_max_len;
//   float lamda_ad;
//   float lamda_census;
//   uint8_t cross_t1;
//   uint8_t cross_t2;
//   int cross_l1;
//   int cross_l2;
//   int tso;
//   float p1;
//   float p2;
//   uint8_t aggregate_iter;
//   uint8_t cost_type;
//   uint16_t m_resize_height;
//   uint16_t m_resize_width;
// } ADCensusOption;

enum CrossArmDirection 
{
  CROSS_ARM_HORIZON = 0,
  CROSS_ARM_VERTICAL,
  CROSS_ARM_DIR_NUM
};

enum class ScanlineDirection : uint8_t
{
  LEFT = 0,
  RIGHT,
  UP,
  DOWN
};

enum class CostType : uint8_t 
{
  INIT_COST,
  AGGREGATE_COST,
  FINAL_COST
};

enum class LeftOrRight : int8_t
{
  LEFT = -1,
  RIGHT = 1
};

struct CrossArm
{
  int left, right, top, bottom;
  CrossArm() : left(0), right(0), top(0), bottom(0) {}
};

struct ArmPixelCount
{
  int cnts[CROSS_ARM_DIR_NUM];
  ArmPixelCount()
  {
    int i;
    for (i = 0; i < CROSS_ARM_DIR_NUM; i++)
    {
      cnts[i] = 0;
    }
  }
};

#endif