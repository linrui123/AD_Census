#include "ad_census.hpp"

void ADCensus::calculate_disparity(LeftOrRight flag)
{
  int i, j, d;
  float min_cost;

  costs_vector & final_costs = (flag == LeftOrRight::RIGHT) ? m_left_aggregate_costs : m_right_aggregate_costs;
  std::vector<std::vector<float>> & final_disparity = (flag == LeftOrRight::LEFT) ? m_left_final_disparity : m_right_final_disparity;
  std::vector<float> cost_local(m_opt.m_max_disparity);

  for (i = 0; i < m_left_image.rows; i++)
  {
    for (j = 0; j < m_left_image.cols; j++)
    {
      min_cost = FLT_MAX;
      int best_disp_t = 0;
      for (d = 0; d < m_opt.m_max_disparity; d++)
      {
        cost_local[d] = final_costs[i][j][d];
        if (final_costs[i][j][d] < min_cost)
        {
          min_cost = final_costs[i][j][d];
          best_disp_t = d;
        }
      }

      if ((best_disp_t == 0) || (best_disp_t == m_opt.m_max_disparity - 1))
      {
        final_disparity[i][j] = -1;
        continue;
      }

      const int idx1 = best_disp_t - 1;
      const int idx2 = best_disp_t + 1;
      const float cost1 = cost_local[idx1];
      const float cost2 = cost_local[idx2];
      const float denom = cost1 + cost2 - 2 * min_cost;
      
      if (denom != 0)
      {
        final_disparity[i][j] = (float)best_disp_t - (cost2 - cost1) / (2 * denom);
      }
      else
      {
        final_disparity[i][j] = (float)best_disp_t;
      }
    }
  }
}

void ADCensus::outlier_detection()
{
  int i, j;
  int threshold = 1;
  int col_r, col_l;
  float disp, disp_l, disp_r;

  m_occlusions.clear();
  m_mismatches.clear();

  for (i = 0; i < m_left_image.rows; i++)
  {
    for (j = 0; j < m_left_image.cols; j++)
    {
      float disp = m_left_final_disparity[i][j];
      if (disp == -1)
      {
        m_mismatches.push_back(std::pair<int, int>(i, j));
        continue;
      }

      col_r = lround(j - disp);  // 指向右图像
      if ((col_r >= 0) && (col_r < m_left_image.cols))
      {
        disp_r = m_right_final_disparity[i][col_r];
        if (abs(disp - disp_r) > threshold)
        {
          col_l = lround(col_r + disp_r);  // 指向左图像
          if ((col_l >= 0) && (col_l < m_left_image.cols))
          {
            disp_l = m_left_final_disparity[i][col_l];
            if (disp_l > disp)
            {
              m_occlusions.push_back(std::pair<int, int>(i, j));
            }
            else
            {
              m_mismatches.push_back(std::pair<int, int>(i, j));
            }
          }
          else
          {
            m_mismatches.push_back(std::pair<int, int>(i, j));
          }

          disp = -1;
        }
      }
      else
      {
        disp = -1;
        m_mismatches.push_back(std::pair<int, int>(i, j));
      }
    }
  }
}

void ADCensus::iterative_region_voting()
{
  int i, j, k, l;
  int x, y;
  float disp;
  float d;
  int num_iters = 5;

  std::vector<int> histogram(m_opt.m_max_disparity, 0);

  for (i = 0; i < num_iters; i++)
  {
    for (j = 0; j < 2; j++)
    {
      std::vector<std::pair<int, int>> & outliers = (j == 0) ? m_mismatches : m_occlusions;
      for (auto & pix : outliers)
      {
        x = pix.first;
        y = pix.second;
        disp = m_left_final_disparity[x][y];
        if (disp != -1)
        {
          continue;
        }

        memset((void *)&histogram[0], 0, m_opt.m_max_disparity);

        CrossArm & arm = m_left_cross_arms[x][y];
        for (k = arm.top; k <= arm.bottom; k++)
        {
          CrossArm & arm2 = m_left_cross_arms[k][y];
          for (l = arm2.left; l <= arm2.right; l++)
          {
            d = m_left_final_disparity[k][l];
            if (d != -1)
            {
              histogram[int(d)]++;
            }
          }
        }

        float best_disp = 0;
        int max_ht = 0;
        for (k = 0; k < m_opt.m_max_disparity; k++)
        {
          l = histogram[k];
          if (max_ht < l)
          {
            max_ht = l;
            best_disp = (float)k;
          }
        }

        m_left_final_disparity[x][y] = best_disp;
      }

      for (auto it = outliers.begin(); it != outliers.end();)
      {
        x = it->first;
        y = it->second;
        if (m_left_final_disparity[x][y] != -1)
        {
          it = outliers.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
  }
}

void ADCensus::proper_interpolation()
{
  const float pi = 3.141592654;
  int i, j, k, l;
  int x, y;
  int xx, yy;
  double angle;
  float sina, cosa;
  float d;
  uint8_t dist, min_dist;
  std::vector<std::pair<std::pair<int, int>, float>> disp_collects;

  for (i = 0; i < 2; i++)
  {
    auto & outliers = (i == 0) ? m_mismatches : m_occlusions;
    if (outliers.empty())
    {
      std::cout << "outliers is empty" << std::endl;
      continue;
    }
    std::vector<float> fill_disps(outliers.size());

    for (j = 0; j < outliers.size(); j++)
    {
      x = outliers[j].first;
      y = outliers[j].second;
      disp_collects.clear();

      angle = 0.0;
      for (k = 0; k < 16; k++)
      {
        sina = sin(angle);
        cosa = cos(angle);
        for (l = 0; l < m_opt.m_max_disparity; l++)
        {
          xx = lround(x + l * sina);
          yy = lround(y + l * cosa);
          if ((xx < 0) || (x >= m_left_image.rows) || (yy < 0) || (yy >= m_left_image.cols))
          {
            break;
          }
          d = m_left_final_disparity[xx][yy];
          if (d != -1)
          {
            disp_collects.emplace_back(std::pair<std::pair<int, int>, float>(std::pair<int, int>(xx, yy), d));
            break;
          }
        }
        angle += pi / 16;
      }

      if (disp_collects.empty())
      {
        continue;
      }

      if (k == 0)
      {
        min_dist = 255;
        for (auto & dc : disp_collects)
        {
          dist = this->color_distance(m_left_image, x, y, dc.first.first, dc.first.second);
          if (min_dist > dist)
          {
            min_dist = dist;
            d = dc.second;
          }
        }
        fill_disps[j] = d;
      }
      else
      {
        d = FLT_MAX;
        for (auto & dc : disp_collects)
        {
          d = std::min(d, dc.second);
        }
        fill_disps[j] = d;
      }
    }
    for (j = 0; j < outliers.size(); j++)
    {
      x = outliers[j].first;
      y = outliers[j].second;
      m_left_final_disparity[x][y] = fill_disps[j];
    }
  }
}

void ADCensus::depth_discontinuity_adjustment()
{
  int i, j, k;
  const float edge_thres = 5.0f;
  std::vector<std::vector<uint8_t>> edge_mask;

  edge_mask.resize(m_left_image.rows);
  for (i = 0; i < m_left_image.rows; i++)
  {
    edge_mask[i].resize(m_left_image.cols);
  }

  for (i = 1; i < m_left_image.rows - 1; i++)
  {
    for (j = 1; j < m_left_image.cols - 1; j++)
    {
      const float grad_x = (-m_left_final_disparity[i - 1][j - 1] + m_left_final_disparity[i - 1][j + 1]) + 
                           (-2 * m_left_final_disparity[i][j - 1] + 2 * m_left_final_disparity[i][j + 1]) + 
                           (-m_left_final_disparity[i + 1][j - 1] + m_left_final_disparity[i + 1][j + 1]);
      const float grad_y = (-m_left_final_disparity[i - 1][j - 1] - 2 * m_left_final_disparity[i - 1][j] - m_left_final_disparity[i - 1][j + 1]) + 
                           ( m_left_final_disparity[i + 1][j - 1] + 2 * m_left_final_disparity[i + 1][j] + m_left_final_disparity[i + 1][j + 1]);
      const float grad = abs(grad_x) + abs(grad_y);
      if (grad > edge_thres)
      {
        edge_mask[i][j] = 1;
      }
    }
  }

  for (i = 0; i < m_left_image.rows; i++)
  {
    for (j = 1; j < m_left_image.cols - 1; j++)
    {
      const float mask = edge_mask[i][j];
      if (mask == 1)
      {
        if (m_left_final_disparity[i][j] != -1)
        {
          const int d = lround(m_left_final_disparity[i][j]);
          float c = m_left_aggregate_costs[i][j][d];
          for (k = 0; k < 2; k++)
          {
            const int x2 = (k == 0) ? j - 1 : j + 1;
            if (m_left_final_disparity[i][x2] != -1)
            {
              const int d2 = lround(m_left_final_disparity[i][x2]);
              const float c2 = m_left_aggregate_costs[i][x2][d2];
              if (c2 < c)
              {
                m_left_final_disparity[i][j] = d2;
                c = c2;
              }
            }
          }
        }
      }
    }
  }
}

void ADCensus::median_filter(std::vector<std::vector<float>> & in, 
                             std::vector<std::vector<float>> & out, 
                             const int rows, const int cols, const int win_size)
{
  const int radius = win_size / 2;
  const int size = win_size * win_size;

  std::vector<float> win_data;
  win_data.reserve(size);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      win_data.clear();
      for (int k = -radius; k <= radius; k++)
      {
        for (int l = -radius; l <= radius; l++)
        {
          const int row = i + k;
          const int col = j + l;
          if ((row >= 0) && (row < rows) && (col >= 0) && (col < cols))
          {
            win_data.push_back(in[row][col]);
          }
        }
      }
      std::sort(win_data.begin(), win_data.end());
      if (win_data.size() > 0)
      {
        out[i][j] = win_data[win_data.size() / 2];
        
      }
    }
  }
}

void ADCensus::multistep_refine()
{
  calculate_disparity(LeftOrRight::LEFT);
  calculate_disparity(LeftOrRight::RIGHT);

  outlier_detection();
  // iterative_region_voting();
  // proper_interpolation();
  // depth_discontinuity_adjustment();
  median_filter(m_left_final_disparity, m_left_final_disparity, m_left_image.rows, m_left_image.cols, 3);
}