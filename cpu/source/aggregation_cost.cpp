#include "ad_census.hpp"

uint8_t ADCensus::color_distance(const cv::Mat & image, 
                                 const int & row1, 
                                 const int & col1, 
                                 const int & row2, 
                                 const int & col2)
{
  uint8_t chnl_num = image.channels();
  uint8_t i;
  uint8_t result = 0;
  uint8_t dis;
  uint8_t color1, color2;

  for (i = 0; i < chnl_num; i++)
  {
    if ((row1 < 0) || (row1 >= image.rows) || (col1 < 0) || (col1 >= image.cols))
    {
      color1 = 0;
    }
    else
    {
      const uint8_t * data = image.ptr<uint8_t>(row1);
      color1 = data[col1 * chnl_num + i];
    }
    if ((row2 < 0) || (row2 >= image.rows) || (col2 < 0) || (col2 >= image.cols))
    {
      color2 = 0;
    }
    else
    {
      const uint8_t * data = image.ptr<uint8_t>(row2);
      color2 = data[col2 * chnl_num + i];
    }

    dis = (uint8_t)abs(color1 - color2);
    result = std::max(result, dis);
  }

  return result;
}

int ADCensus::spatial_distance(const int & row1, 
                               const int & col1, 
                               const int & row2, 
                               const int & col2)
{
  return sqrt((row1 - row2) * (row1 - row2) + (col1 - col2) * (col1 - col2));
}

void ADCensus::find_horizontal_arm(const cv::Mat & image, 
                                   const int & row1, 
                                   const int & col1, 
                                   int & left, 
                                   int & right)
{
  uint8_t i, j;
  uint8_t color_dist;
  uint8_t color_dist2;
  int spatial_dist;
  int y;
  int last_y;

  left = 0;
  right = 0;

  for (j = 0; j < 2; j++)
  {
    last_y = col1;

    y = col1;
    
    for (i = 0; i < m_opt.m_cross_arm_max_len; i++)
    {
      last_y = y;
      if (j == 0)
      {
        left = y;
        if (y - 1 < 0)
        {
          break;
        }
        y--;
      }
      else
      {
        right = y;
        if (y + 1 >= image.cols)
        {
          break;
        }
        y++;
      }

      color_dist = this->color_distance(image, row1, col1, row1, y);
      if (color_dist >= m_opt.cross_t1)
      {
        break;
      }

      color_dist2 = this->color_distance(image, row1, last_y, row1, y);
      if (color_dist2 >= m_opt.cross_t1)
      {
        break;
      }

      spatial_dist = this->spatial_distance(row1, col1, row1, y);
      if (spatial_dist > m_opt.cross_l1)
      {
        break;
      }

      if (spatial_dist > m_opt.cross_l2)
      {
        if (color_dist >= m_opt.cross_t2)
        {
          break;
        }
      }
    }
  }
}

void ADCensus::find_vertical_arm(const cv::Mat & image, 
                                 const int & row1, 
                                 const int & col1, 
                                 int & top, 
                                 int & bottom)
{
  uint8_t i, j;
  uint8_t color_dist;
  uint8_t color_dist2;
  int spatial_dist;
  int x;
  int last_x;

  top = 0;
  bottom = 0;

  for (j = 0; j < 2; j++)
  {
    last_x = row1;

    x = row1;
    
    for (i = 0; i < m_opt.m_cross_arm_max_len; i++)
    {
      last_x = x;
      if (j == 0)
      {
        top = x;
        if (x - 1 < 0)
        {
          break;
        }
        x--;
      }
      else
      {
        bottom = x;
        if (x + 1 >= image.rows)
        {
          break;
        }
        x++;
      }

      color_dist = this->color_distance(image, row1, col1, x, col1);
      if (color_dist >= m_opt.cross_t1)
      {
        break;
      }

      color_dist2 = this->color_distance(image, last_x, col1, x, col1);
      if (color_dist2 >= m_opt.cross_t1)
      {
        break;
      }

      spatial_dist = this->spatial_distance(row1, col1, x, col1);
      if (spatial_dist > m_opt.cross_l1)
      {
        break;
      }

      if (spatial_dist > m_opt.cross_l2)
      {
        if (color_dist >= m_opt.cross_t2)
        {
          break;
        }
      }
    }
  }
}

void ADCensus::build_arms(LeftOrRight flag)
{
  int i, j;
  cv::Mat &image = (flag == LeftOrRight::LEFT) ? m_left_image : m_right_image;
  std::vector<std::vector<CrossArm>> &cross_arms = (flag == LeftOrRight::LEFT) ? m_left_cross_arms : m_right_cross_arms;

  for (i = 0; i < m_left_image.rows; i++)
  {
    for (j = 0; j < m_left_image.cols; j++)
    {
      this->find_horizontal_arm(image, 
                                i, j, 
                                cross_arms[i][j].left, 
                                cross_arms[i][j].right);

      this->find_vertical_arm(image, 
                              i, j, 
                              cross_arms[i][j].top, 
                              cross_arms[i][j].bottom);
    }
  }
}

void ADCensus::compute_arms_pixel_count(LeftOrRight flag)
{
  int i, j, t;
  uint16_t count;
  CrossArm * arm = nullptr;

  std::vector<std::vector<CrossArm>> &cross_arms = (flag == LeftOrRight::LEFT) ? m_left_cross_arms : m_right_cross_arms;
  std::vector<std::vector<ArmPixelCount>> &dst_arms_pixel_cnts = (flag == LeftOrRight::LEFT) ? m_left_arms_pixel_cnts : m_right_arms_pixel_cnts;

  int h = m_left_image.rows;
  int w = m_left_image.cols;

  uint16_t arms_pixel_cnts[h * w];

  for (i = 0; i < h; i++)
  {
    for (j = 0; j < w; j++)
    {
      arm = &cross_arms[i][j];
      count = arm->right - arm->left + 1;
      arms_pixel_cnts[i * w + j] = count;
    }
  }

  for (i = 0; i < h; i++)
  {
    for (j = 0; j < w; j++)
    {
      arm = &cross_arms[i][j];
      count = 0;
      for (t = arm->top; t <= arm->bottom; t++)
      {
        count += arms_pixel_cnts[t * w + j];
      }
      dst_arms_pixel_cnts[i][j].cnts[CROSS_ARM_HORIZON] = count;
    }
  }

  for (i = 0; i < h; i++)
  {
    for (j = 0; j < w; j++)
    {
      arm = &cross_arms[i][j];
      count = arm->bottom - arm->top + 1;
      arms_pixel_cnts[i * w + j] = count;
    }
  }

  for (i = 0; i < h; i++)
  {
    for (j = 0; j < w; j++)
    {
      arm = &cross_arms[i][j];
      count = 0;
      for (t = arm->left; t <= arm->right; t++)
      {
        count += arms_pixel_cnts[i * w + t];
      }
      dst_arms_pixel_cnts[i][j].cnts[CROSS_ARM_VERTICAL] = count;
    }
  }  
}

void ADCensus::aggregate_cost_in_arms(LeftOrRight flag)
{
  int i, j, k, l, d;
  float cost;
  std::vector<CrossArmDirection> dirs(CROSS_ARM_DIR_NUM);
  bool horizon_first = true;

  std::vector<std::vector<CrossArm>> &cross_arms = (flag == LeftOrRight::LEFT) ? m_left_cross_arms : m_right_cross_arms;
  std::vector<std::vector<ArmPixelCount>> &arms_pixel_cnts = (flag == LeftOrRight::LEFT) ? m_left_arms_pixel_cnts : m_right_arms_pixel_cnts;
  costs_vector &dst_aggregate_costs = (flag == LeftOrRight::LEFT) ? m_left_aggregate_costs : m_right_aggregate_costs;

  int h = m_left_image.rows;
  int w = m_left_image.cols;
  float costs_tmp[2][h * w];

  if (flag == LeftOrRight::LEFT)
  {
    std::copy(m_left_disparity_costs.begin(), m_left_disparity_costs.end(), dst_aggregate_costs.begin());
  }
  else
  {
    std::copy(m_right_disparity_costs.begin(), m_right_disparity_costs.end(), dst_aggregate_costs.begin());
  }
  

  for (l = 0; l < m_opt.aggregate_iter; l++)
  {
    if (horizon_first == true)
    {
      dirs[0] = CROSS_ARM_HORIZON;
      dirs[1] = CROSS_ARM_VERTICAL;
    }
    else
    {
      dirs[0] = CROSS_ARM_VERTICAL;
      dirs[1] = CROSS_ARM_HORIZON;
    }

    for (d = 0; d < m_opt.m_max_disparity; d++)
    {
      for (i = 0; i < h; i++)
      {
        for (j = 0; j < w; j++)
        {
          costs_tmp[0][i * w + j] = dst_aggregate_costs[i][j][d];
        }
      }

      for (auto & dir : dirs)
      {
        for (i = 0; i < h; i++)
        {
          for (j = 0; j < w; j++)
          {
            cost = 0;
            if (dir == CROSS_ARM_HORIZON)
            {
              for (k = cross_arms[i][j].left; k <= cross_arms[i][j].right; k++)
              {
                if (horizon_first == true)
                {
                  cost += costs_tmp[0][i * w + k];
                }
                else
                {
                  cost += costs_tmp[1][i * w + k];
                }
              }
            }
            else
            {
              for (k = cross_arms[i][j].top; k <= cross_arms[i][j].bottom; k++)
              {
                if (horizon_first == true)
                {
                  cost += costs_tmp[1][k * w + j];
                }
                else
                {
                  cost += costs_tmp[0][k * w + j];
                }
              }
            }

            if (((dir == CROSS_ARM_HORIZON) && (horizon_first == true)) || ((dir == CROSS_ARM_VERTICAL) && (horizon_first == false)))
            {
              costs_tmp[1][i * w + j] = cost;
            }
            else
            {
              dst_aggregate_costs[i][j][d] = cost / arms_pixel_cnts[i][j].cnts[dirs[0]];
            }
          }
        }
      }
    }

    horizon_first = !horizon_first;  
  }
}

void ADCensus::aggregation_cost()
{
  this->build_arms(LeftOrRight::LEFT);
  this->compute_arms_pixel_count(LeftOrRight::LEFT);
  this->aggregate_cost_in_arms(LeftOrRight::LEFT);

  this->build_arms(LeftOrRight::RIGHT);
  this->compute_arms_pixel_count(LeftOrRight::RIGHT);
  this->aggregate_cost_in_arms(LeftOrRight::RIGHT);
}