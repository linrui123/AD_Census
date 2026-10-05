
#include <iostream>
#include "ad_census.hpp"
#include "cu_ad_census.hpp"

class Compare
{
public:
  Compare() {}
  ~Compare() {}

  void compare_init_costs(
    ADCensus & a, 
    CuADCensus & b
  )
  {
    int rows = a.m_left_disparity_costs.size();
    int cols = a.m_left_disparity_costs[0].size();
    int dis = a.m_left_disparity_costs[0][0].size();

    float error = 0;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        for (int k = 0; k < dis; k++)
        {
          float e = abs(a.m_left_disparity_costs[i][j][k] - b.m_left_init_costs[i * cols * dis + j * dis + k]);
          if (e > 0.000001)
          {
            std::cout << "cost: " << a.m_left_disparity_costs[i][j][k]
                      << ", " << b.m_left_init_costs[i * cols * dis + j * dis + k]
                      << ", i: " << i 
                      << ", j: " << j 
                      << ", k: " << k
                      << ", e: " << e << std::endl;
            error += e;
          }

          e = abs(a.m_right_disparity_costs[i][j][k] - b.m_right_init_costs[i * cols * dis + j * dis + k]);
          if (e > 0.000001)
          {
            std::cout << "cost: " << a.m_right_disparity_costs[i][j][k]
                      << ", " << b.m_right_init_costs[i * cols * dis + j * dis + k]
                      << ", i: " << i 
                      << ", j: " << j 
                      << ", k: " << k
                      << ", e: " << e << std::endl;
            error += e;
          }
        }
      }
    }

    std::cout << "error: " << error << std::endl;
  }

  void compare_cross_arm(
    ADCensus & a, 
    CuADCensus & b
  )
  {
    int rows = a.m_left_cross_arms.size();
    int cols = a.m_left_cross_arms[0].size();

    int error = 0;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        int left1 = a.m_left_cross_arms[i][j].left;
        int left2 = b.m_left_cross_arms[i * cols + j].left;

        int right1 = a.m_left_cross_arms[i][j].right;
        int right2 = b.m_left_cross_arms[i * cols + j].right;

        int top1 = a.m_left_cross_arms[i][j].top;
        int top2 = b.m_left_cross_arms[i * cols + j].top;

        int bottom1 = a.m_left_cross_arms[i][j].bottom;
        int bottom2 = b.m_left_cross_arms[i * cols + j].bottom;

        int e = abs(left1 - left2) + abs(right1 - right2) + abs(top1 - top2) + abs(bottom1 - bottom2);
        if (e != 0)
        {
          std::cout << "row:" << i << ", col:" << j 
                    << ", left1:" << left1 << ", left2:" << left2
                    << ", right1:" << right1 << ", right2:" << right2
                    << ", top1:" << top1 << ", top2:" << top2
                    << ", bottom1:" << bottom1 << ", bottom2:" << bottom2 << std::endl;
        }

        if (left1 != left2)
        {
          uint8_t color_dist = a.color_distance(a.m_left_image, i, j, i, left1 - 1);
          uint8_t color_dist2 = a.color_distance(a.m_left_image, i, j, i, left1);
          int spacial_dist = a.spatial_distance(i, j, i, left1 - 1);

          if (color_dist >= 20)
          {
            std::cout << "left1: color_dist:" << (int)color_dist << std::endl;
          }
          if (color_dist2 >= 20)
          {
            std::cout << "left1: color_dist2:" << (int)color_dist2 << std::endl;
          }
          if (spacial_dist > 34)
          {
            std::cout << "left1: spacial_dist:" << (int)spacial_dist << std::endl;
          }
          if (spacial_dist > 17)
          {
            if (color_dist >= 6)
            {
              std::cout << "left1: spacial_dist:" << (int)spacial_dist 
                        << ", color_dist:" << (int)color_dist << std::endl;
            }
          }
          
          color_dist = a.color_distance(a.m_left_image, i, j, i, left2 - 1);
          color_dist2 = a.color_distance(a.m_left_image, i, j, i, left2);
          spacial_dist = a.spatial_distance(i, j, i, left2 - 1);

          if (color_dist >= 20)
          {
            std::cout << "left2: color_dist:" << (int)color_dist << std::endl;
          }
          if (color_dist2 >= 20)
          {
            std::cout << "left2: color_dist2:" << (int)color_dist2 << std::endl;
          }
          if (spacial_dist > 34)
          {
            std::cout << "left2: spacial_dist:" << (int)spacial_dist << std::endl;
          }
          if (spacial_dist > 17)
          {
            if (color_dist >= 6)
            {
              std::cout << "left2: spacial_dist:" << (int)spacial_dist 
                        << ", color_dist:" << (int)color_dist << std::endl;
            }
          }
        }

        error += e;
      }
    }

    std::cout << "error: " << error << std::endl;
  }

  void compare_cross_arm_pixel_cnts(
    ADCensus & a, 
    CuADCensus & b
  )
  {
    int rows = a.m_left_arms_pixel_cnts.size();
    int cols = a.m_left_arms_pixel_cnts[0].size();

    int error = 0;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        int horizon_cnt1 = a.m_left_arms_pixel_cnts[i][j].cnts[0];
        int horizon_cnt2 = b.m_left_arms_pixel_cnt[i * cols + j].cnts[0];

        int vertical_cnt1 = a.m_left_arms_pixel_cnts[i][j].cnts[1];
        int vertical_cnt2 = b.m_left_arms_pixel_cnt[i * cols + j].cnts[1];

        error += abs(horizon_cnt1 - horizon_cnt2);
        error += abs(vertical_cnt1 - vertical_cnt2);

        if ((horizon_cnt1 != horizon_cnt2) || (vertical_cnt1 != vertical_cnt2))
        {
          std::cout << "row:" << i << ", col:" << j 
                    << ", horizon_cnt1:" << horizon_cnt1 << ", horizon_cnt2:" << horizon_cnt2
                    << ", vertical_cnt1:" << vertical_cnt1 << ", vertical_cnt2:" << vertical_cnt2
                    << std::endl;
        }
      }
    }

    std::cout << "error:" << error << std::endl;
  }

  void compare_aggregate_costs(
    ADCensus & a, 
    CuADCensus & b
  )
  {
    int rows = a.m_left_aggregate_costs.size();
    int cols = a.m_left_aggregate_costs[0].size();
    int dis = a.m_left_aggregate_costs[0][0].size();

    float error = 0;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        for (int k = 0; k < dis; k++)
        {
          float e = abs(a.m_left_aggregate_costs[i][j][k] - b.m_left_aggregate_costs[i * cols * dis + j * dis + k]);
          if (e > 0.000001)
          {
            std::cout << "left cost: " << a.m_left_aggregate_costs[i][j][k]
                      << ", " << b.m_left_aggregate_costs[i * cols * dis + j * dis + k]
                      << ", i: " << i 
                      << ", j: " << j 
                      << ", k: " << k
                      << ", e: " << e << std::endl;
            error += e;
          }

          e = abs(a.m_right_aggregate_costs[i][j][k] - b.m_right_aggregate_costs[i * cols * dis + j * dis + k]);
          if (e > 0.000001)
          {
            std::cout << "right cost: " << a.m_right_aggregate_costs[i][j][k]
                      << ", " << b.m_right_aggregate_costs[i * cols * dis + j * dis + k]
                      << ", i: " << i 
                      << ", j: " << j 
                      << ", k: " << k
                      << ", e: " << e << std::endl;
            error += e;
          }
        }
      }
    }

    std::cout << "error: " << error << std::endl;
  }
};