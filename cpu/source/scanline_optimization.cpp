#include "ad_census.hpp"


void ADCensus::scanline_optimize_left_right(LeftOrRight flag,  ScanlineDirection scanline_dir)
{
  int i, j, d;
  int dir;
  int y, yr;
  float color_dis, color_dis2;
  float p1, p2;
  float l1, l2, l3, l4;
  int disparity_dir;
  costs_vector * costs_init;
  costs_vector * costs_aggr;

  cv::Mat &image1 = (flag == LeftOrRight::LEFT) ? m_left_image : m_right_image;
  cv::Mat &image2 = (flag == LeftOrRight::LEFT) ? m_right_image : m_left_image;
  
  switch (flag)
  {
  case LeftOrRight::LEFT:
    disparity_dir = -1;
    switch (scanline_dir)
    {
    case ScanlineDirection::RIGHT :
      dir = 1;
      costs_init = &m_left_aggregate_costs;
      costs_aggr = &m_left_disparity_costs;
      break;
    case ScanlineDirection::LEFT:
      dir = -1;
      costs_init = &m_left_disparity_costs;
      costs_aggr = &m_left_aggregate_costs;
      break;
    default:
      break;
    }
    break;
  case LeftOrRight::RIGHT:
    disparity_dir = 1;
    switch (scanline_dir)
    {
    case ScanlineDirection::RIGHT :
      dir = 1;
      costs_init = &m_right_aggregate_costs;
      costs_aggr = &m_right_disparity_costs;
      break;
    case ScanlineDirection::LEFT :
      dir = -1;
      costs_init = &m_right_disparity_costs;
      costs_aggr = &m_right_aggregate_costs;
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }

  for (i = 0; i < image1.rows; i++)
  {
    y = (scanline_dir == ScanlineDirection::RIGHT) ? 0 : image1.cols - 1;
    std::vector<float> last_path_costs(m_opt.m_max_disparity + 2, FLT_MAX);
    costs_aggr->at(i).at(y).assign(costs_init->at(i).at(y).begin(), costs_init->at(i).at(y).end());

    for (d = 0; d < m_opt.m_max_disparity; d++)
    {
      last_path_costs[d + 1] = costs_aggr->at(i).at(y).at(d);
    }

    float mincost_last_path = FLT_MAX;
    for (auto & cost : last_path_costs)
    {
      mincost_last_path = std::min(cost, mincost_last_path);
    }

    for (j = 0; j < image1.cols - 1; j++)
    {
      yr = y;
      y += dir;
      float min_cost = FLT_MAX;

      for (d = 0; d < m_opt.m_max_disparity; d++)
      {
        color_dis = this->color_distance(image1, i, y, i, yr);
        color_dis2 = this->color_distance(image2, i, y + d * disparity_dir, i, yr + d * disparity_dir);

        if ((color_dis < m_opt.tso) && (color_dis2 < m_opt.tso))
        {
          p1 = m_opt.p1;
          p2 = m_opt.p2;
        }
        else if ((color_dis < m_opt.tso) && (color_dis2 >= m_opt.tso))
        {
          p1 = m_opt.p1 / 4;
          p2 = m_opt.p2 / 4;
        }
        else if ((color_dis >= m_opt.tso) && (color_dis2 < m_opt.tso))
        {
          p1 = m_opt.p1 / 4;
          p2 = m_opt.p2 / 4;
        }
        else if ((color_dis >= m_opt.tso) && (color_dis2 >= m_opt.tso))
        {
          p1 = m_opt.p1 / 10;
          p2 = m_opt.p2 / 10;
        }

        float cost = costs_init->at(i).at(y).at(d);
        l1 = last_path_costs[d + 1];
        l2 = last_path_costs[d] + p1;
        l3 = last_path_costs[d + 2] + p1;
        l4 = mincost_last_path + p2;

        float cost_s = cost + std::min(std::min(l1, l2), std::min(l3, l4));
        cost_s /= 2;

        costs_aggr->at(i).at(y).at(d) = cost_s;
        min_cost = std::min(min_cost, cost_s);
      }

      mincost_last_path = min_cost;
      for (d = 0; d < m_opt.m_max_disparity; d++)
      {
        last_path_costs[d + 1] = costs_aggr->at(i).at(y).at(d);
      }
    }
  }
}

void ADCensus::scanline_optimize_up_down(LeftOrRight flag, ScanlineDirection scanline_dir)
{
  int i, j, d;
  int dir;
  int x, xr;
  float color_dis, color_dis2;
  float p1, p2;
  float l1, l2, l3, l4;
  int disparity_dir;
  costs_vector * costs_init;
  costs_vector * costs_aggr;

  cv::Mat &image1 = (flag == LeftOrRight::LEFT) ? m_left_image : m_right_image;
  cv::Mat &image2 = (flag == LeftOrRight::LEFT) ? m_right_image : m_left_image;

  switch (flag)
  {
  case LeftOrRight::LEFT:
    disparity_dir = -1;
    switch (scanline_dir)
    {
    case ScanlineDirection::DOWN :
      dir = 1;
      costs_init = &m_left_aggregate_costs;
      costs_aggr = &m_left_disparity_costs;
      break;
    case ScanlineDirection::UP :
      dir = -1;
      costs_init = &m_left_disparity_costs;
      costs_aggr = &m_left_aggregate_costs;
      break;
    default:
      break;
    }
    break;
  case LeftOrRight::RIGHT:
    disparity_dir = 1;
    switch (scanline_dir)
    {
    case ScanlineDirection::DOWN :
      dir = 1;
      costs_init = &m_right_aggregate_costs;
      costs_aggr = &m_right_disparity_costs;
      break;
    case ScanlineDirection::UP :
      dir = -1;
      costs_init = &m_right_disparity_costs;
      costs_aggr = &m_right_aggregate_costs;
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }

  for (j = 0; j < image1.cols - 1; j++)
  {
    x = (scanline_dir == ScanlineDirection::DOWN) ? 0 : image1.rows - 1;
    std::vector<float> last_path_costs(m_opt.m_max_disparity + 2, FLT_MAX);
    (*costs_aggr)[x][j].assign((*costs_init)[x][j].begin(), (*costs_init)[x][j].end());
 
    for (d = 0; d < m_opt.m_max_disparity; d++)
    {
      last_path_costs[d + 1] = (*costs_aggr)[x][j][d];
    }

    float mincost_last_path = FLT_MAX;
    for (auto & cost : last_path_costs)
    {
      mincost_last_path = std::min(cost, mincost_last_path);
    }

    for (i = 0; i < image1.rows - 1; i++)
    {
      xr = x;
      x += dir;
      float min_cost = FLT_MAX;

      for (d = 0; d < m_opt.m_max_disparity; d++)
      {
        color_dis = this->color_distance(image1, x, j, xr, j);
        color_dis2 = this->color_distance(image2, x, j + d * disparity_dir, xr, j + d * disparity_dir);

        if ((color_dis < m_opt.tso) && (color_dis2 < m_opt.tso))
        {
          p1 = m_opt.p1;
          p2 = m_opt.p2;
        }
        else if ((color_dis < m_opt.tso) && (color_dis2 >= m_opt.tso))
        {
          p1 = m_opt.p1 / 4;
          p2 = m_opt.p2 / 4;
        }
        else if ((color_dis >= m_opt.tso) && (color_dis2 < m_opt.tso))
        {
          p1 = m_opt.p1 / 4;
          p2 = m_opt.p2 / 4;
        }
        else if ((color_dis >= m_opt.tso) && (color_dis2 >= m_opt.tso))
        {
          p1 = m_opt.p1 / 10;
          p2 = m_opt.p2 / 10;
        }

        float cost = (*costs_init)[x][j][d];
        l1 = last_path_costs[d + 1];
        l2 = last_path_costs[d] + p1;
        l3 = last_path_costs[d + 2] + p1;
        l4 = mincost_last_path + p2;

        float cost_s = cost + std::min(std::min(l1, l2), std::min(l3, l4));
        cost_s /= 2;

        (*costs_aggr)[x][j][d] = cost_s;
        min_cost = std::min(min_cost, cost_s);
      }

      mincost_last_path = min_cost;
      for (d = 0; d < m_opt.m_max_disparity; d++)
      {
        last_path_costs[d + 1] = (*costs_aggr)[x][j][d];
      }
    }
  }
}

void ADCensus::scanline_optimization()
{
  this->scanline_optimize_left_right(LeftOrRight::LEFT, ScanlineDirection::RIGHT);
  this->scanline_optimize_left_right(LeftOrRight::LEFT, ScanlineDirection::LEFT);
  this->scanline_optimize_up_down(LeftOrRight::LEFT, ScanlineDirection::DOWN);
  this->scanline_optimize_up_down(LeftOrRight::LEFT, ScanlineDirection::UP);

  // this->scanline_optimize_left_right(RIGHT, SCANLINE_RIGHT);
  // this->scanline_optimize_left_right(RIGHT, SCANLINE_LEFT);
  // this->scanline_optimize_up_down(RIGHT, SCANLINE_DOWN);
  // this->scanline_optimize_up_down(RIGHT, SCANLINE_UP);
}