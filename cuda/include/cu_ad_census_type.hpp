#ifndef CU_AD_CENSUS_TYPE_HPP
#define CU_AD_CENSUS_TYPE_HPP

#include <iostream>
#include <string>

#define CROSS_ARM_LEFT    (0)
#define CROSS_ARM_RIGHT   (1)
#define CROSS_ARM_UP      (2)
#define CROSS_ARM_DOWN    (3)

struct CuCrossArm
{
  int left, right, top, bottom;
};

struct CuArmPixelCnt
{
  int cnts[2];
};

#endif