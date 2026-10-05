#include "ad_census.hpp"

float ADCensus::ad_cost(const cv::Mat & image1, const int & row1, const int & col1, 
                        const cv::Mat & image2, const int & row2, const int & col2)
{
  assert(image1.channels() == image2.channels());

  float ad_cost = 0;
  uint8_t left_pixel_rgb;
  uint8_t right_pixel_rgb;
  uint8_t i;
  uint8_t chnl_num;
  
  chnl_num = image1.channels();

  for (i = 0; i < chnl_num; i++)
  {
    const uint8_t * data1 = image1.ptr<uint8_t>(row1);
    left_pixel_rgb = data1[col1 * chnl_num + i];
    
    const uint8_t * data2 = image2.ptr<uint8_t>(row2);
    right_pixel_rgb = data2[col2 * chnl_num + i];
        
    ad_cost += abs(left_pixel_rgb - right_pixel_rgb);
    if ((row1 == 319) && (col1 == 62) && (row2 == 319) && (col2 == -1))
    {
      printf("cpu pixel: %d, %d\n", left_pixel_rgb, right_pixel_rgb);
    }
  }
  
  return ad_cost / chnl_num;
}

float ADCensus::census_cost(const cv::Mat & image1, const int & row1, const int & col1, 
                            const cv::Mat & image2, const int & row2, const int & col2)
{
  assert(image1.channels() == image2.channels());

  float census_cost = 0;
  int8_t i = 0, j = 0;
  float pixel_gray;
  float mid_pixel_gray;
  const uint8_t * data1;
  const uint8_t * data2;
  int count;
  std::vector<bool> census_val1(m_opt.m_census_size_height * m_opt.m_census_size_width, false);
  std::vector<bool> census_val2(m_opt.m_census_size_height * m_opt.m_census_size_width, false);

  count = 0;
  data1 = image1.ptr<uint8_t>(row1);
  mid_pixel_gray = 0.2126 * data1[col1 * 3] + 0.7152 * data1[col1 * 3 + 1] + 0.0722 * data1[col1 * 3 + 2];
  for (i = -m_opt.m_census_size_height / 2; i <= m_opt.m_census_size_height / 2; i++)
  {
    for (j = -m_opt.m_census_size_width / 2; j <= m_opt.m_census_size_width / 2; j++)
    {
      if ((row1 + i < 0) || (row1 + i >= image1.rows) || (col1 + j < 0) || (col1 + j >= image1.cols))
      {
        count++;
        continue;
      }
      data1 = image1.ptr<uint8_t>(row1 + i);
      pixel_gray = 0.2126 * data1[(col1 + j) * 3] + 0.7152 * data1[(col1 + j) * 3 + 1] + 0.0722 * data1[(col1 + j) * 3 + 2];
      if (pixel_gray < mid_pixel_gray)
      {
        census_val1[count] = true;
      }
      count++;
    }
  }

  count = 0;
  data2 = image2.ptr<uint8_t>(row2);
  mid_pixel_gray = 0.2126 * data2[col2 * 3] + 0.7152 * data2[col2 * 3 + 1] + 0.0722 * data2[col2 * 3 + 2];
  for (i = -m_opt.m_census_size_height / 2; i <= m_opt.m_census_size_height / 2; i++)
  {
    for (j = -m_opt.m_census_size_width / 2; j <= m_opt.m_census_size_width / 2; j++)
    {
      if ((row2 + i < 0) || (row2 + i >= image2.rows) || (col2 + j < 0) || (col2 + j >= image2.cols))
      {
        count++;
        continue;
      }
      data2 = image2.ptr<uint8_t>(row2 + i);
      pixel_gray = 0.2126 * data2[(col2 + j) * 3] + 0.7152 * data2[(col2 + j) * 3 + 1] + 0.0722 * data2[(col2 + j) * 3 + 2];
      if (pixel_gray < mid_pixel_gray)
      {
        census_val2[count] = true;
      }
      count++;
    }
  }

  for (i = 0; i < m_opt.m_census_size_height * m_opt.m_census_size_width; i++)
  {
    if (census_val1[i] != census_val2[i])
    {
      census_cost++;
    }
  }

  return census_cost;
}

void ADCensus::calculate_cost_inter(LeftOrRight flag)
{
  float ad_cost = 0;
  float census_cost = 0;
  int i, j, d;

  int dir = (flag == LeftOrRight::LEFT) ? -1 : 1;
  cv::Mat &image1 = (flag == LeftOrRight::LEFT) ? m_left_image : m_right_image;
  cv::Mat &image2 = (flag == LeftOrRight::LEFT) ? m_right_image : m_left_image;
  costs_vector &init_costs = (flag == LeftOrRight::LEFT) ? m_left_disparity_costs : m_right_disparity_costs;

  for (i = 0; i < m_left_image.rows; i++)
  {
    for (j = 0; j < m_left_image.cols; j++)
    {
      for (d = 0; d < m_opt.m_max_disparity; d++)
      {
        if ((j + d * dir < 0) || (j + d * dir >= m_left_image.cols))
        {
          init_costs[i][j][d] = 1.0;
        }
        else
        {
          ad_cost = this->ad_cost(image1, i, j, image2, i, j + d * dir);
          census_cost = this->census_cost(image1, i, j, image2, i, j + d * dir);
          init_costs[i][j][d] = 2 - exp(-ad_cost / m_opt.lamda_ad) - exp(-census_cost / m_opt.lamda_census);
        }
      }
    }
  }
}

void ADCensus::calculate_cost()
{
  this->calculate_cost_inter(LeftOrRight::LEFT);
  this->calculate_cost_inter(LeftOrRight::RIGHT);
}
