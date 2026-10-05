#include "ad_census.hpp"

ADCensus::ADCensus()
{
}

ADCensus::ADCensus(const std::string & config_file)
{
  parse_config(config_file);
}

ADCensus::~ADCensus()
{
}

void ADCensus::parse_config(const std::string & config_file)
{
  int i, w, h;

  m_opt.parse_config(config_file);

  if (m_opt.m_separation == 0)
  {
    m_left_image = cv::imread(m_opt.m_left_image_name, cv::IMREAD_COLOR);
    m_right_image = cv::imread(m_opt.m_right_image_name, cv::IMREAD_COLOR);
  }
  else
  {
    if (m_opt.m_left_image_name.compare("NULL"))
    {
      cv::Mat img = cv::imread(m_opt.m_left_image_name, cv::IMREAD_COLOR);

      w = img.cols;
      h = img.rows;

      m_left_image = img(cv::Rect(0, 0, w / 2, h));
      m_right_image = img(cv::Rect(w / 2, 0, w / 2, h));
    }
    else if (m_opt.m_right_image_name.compare("NULL"))
    {
      cv::Mat img = cv::imread(m_opt.m_right_image_name, cv::IMREAD_COLOR);
      
      w = img.cols;
      h = img.rows;

      m_left_image = img(cv::Rect(0, 0, w / 2, h / 2));
      m_right_image = img(cv::Rect(w / 2, h / 2, w, h));
    }
  }

  assert((m_left_image.rows > 0) && (m_left_image.cols > 0));
  assert((m_right_image.rows > 0) && (m_right_image.cols > 0));
  assert((m_left_image.rows == m_right_image.rows) && (m_left_image.cols == m_right_image.cols));

  if (m_opt.is_resize > 0)
  {
    cv::resize(m_left_image, m_left_image, cv::Size(m_opt.m_resize_width, m_opt.m_resize_height));
    cv::resize(m_right_image, m_right_image, cv::Size(m_opt.m_resize_width, m_opt.m_resize_height));
  }

  m_disparity_image = cv::Mat(m_left_image.rows, m_left_image.cols, CV_8UC1);
  m_disparity_rgb = cv::Mat(m_left_image.rows, m_left_image.cols, CV_8UC3);
  m_reconstruct_image = cv::Mat(m_left_image.rows, m_left_image.cols, CV_8UC3);

  this->costs_init(m_left_disparity_costs, m_left_image.rows, m_left_image.cols);
  this->costs_init(m_right_disparity_costs, m_right_image.rows, m_right_image.cols);
  this->costs_init(m_left_aggregate_costs, m_left_image.rows, m_left_image.cols);
  this->costs_init(m_right_aggregate_costs, m_right_image.rows, m_right_image.cols);

  m_left_cross_arms.resize(m_left_image.rows);
  for (i = 0; i < m_left_image.rows; i++)
  {
    m_left_cross_arms[i].resize(m_left_image.cols);
  }

  m_right_cross_arms.resize(m_right_image.rows);
  for (i = 0; i < m_right_image.rows; i++)
  {
    m_right_cross_arms[i].resize(m_right_image.cols);
  }

  m_left_arms_pixel_cnts.resize(m_left_image.rows);
  for (i = 0; i < m_left_image.rows; i++)
  {
    m_left_arms_pixel_cnts[i].resize(m_left_image.cols);
  }

  m_right_arms_pixel_cnts.resize(m_right_image.rows);
  for (i = 0; i < m_right_image.rows; i++)
  {
    m_right_arms_pixel_cnts[i].resize(m_right_image.cols);
  }

  m_left_final_disparity.resize(m_left_image.rows);
  for (i = 0; i < m_left_image.rows; i++)
  {
    m_left_final_disparity[i].resize(m_left_image.cols);
  }

  m_right_final_disparity.resize(m_right_image.rows);
  for (i = 0; i < m_right_image.rows; i++)
  {
    m_right_final_disparity[i].resize(m_right_image.cols);
  }

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

void ADCensus::costs_init(costs_vector & costs, int rows, int cols)
{
  int i, j;

  costs.resize(rows);
  for (i = 0; i < rows; i++)
  {
    costs[i].resize(cols);
    for (j = 0; j < cols; j++)
    {
      costs[i][j].resize(m_opt.m_max_disparity);
    }
  }
}

void ADCensus::draw_disparity_image(const std::string & file_name)
{
  int i, j, k, d;

  float cost, disparity_cost;
  int dis;
  CostType type = static_cast<CostType>(m_opt.cost_type);

  for (i = 0; i < m_left_image.rows; i++)
  {
    for (j = 0; j < m_left_image.cols; j++)
    {
      cost = FLT_MAX;
      dis = 0;
      for (k = 0; k < m_opt.m_max_disparity; k++)
      {
        switch (type)
        {
        case CostType::INIT_COST:
          /* code */
          disparity_cost = m_left_disparity_costs[i][j][k];
          break;

        case CostType::AGGREGATE_COST:
          disparity_cost = m_left_aggregate_costs[i][j][k];
          break;
        
        case CostType::FINAL_COST:
          disparity_cost = m_left_aggregate_costs[i][j][k];
        
        default:
          break;
        }

        if (cost > disparity_cost)
        {
          cost = disparity_cost;
          d = k * 255 / m_opt.m_max_disparity;
          dis = k;
        }
      }
      
      m_disparity_image.at<uint8_t>(i, j) = (uint8_t)d;
      if ((j - dis < 0) || (j - dis >= m_left_image.cols))
      {
        m_reconstruct_image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
      }
      else
      {
        m_reconstruct_image.at<cv::Vec3b>(i, j) = m_right_image.at<cv::Vec3b>(i, j - dis);
      }
    }
  }

  cv::applyColorMap(m_disparity_image, m_disparity_rgb, cv::COLORMAP_JET);

  cv::imwrite(file_name + ".jpg", m_disparity_image);
  cv::imwrite(file_name + "_rgb.jpg", m_disparity_rgb);
  cv::imwrite(file_name + "_rebuild.jpg", m_reconstruct_image);

  cv::Mat output = cv::Mat(m_left_image.rows, m_left_image.cols, CV_8UC1);
  cv::compare(m_right_image, m_reconstruct_image, output, cv::CmpTypes::CMP_NE);

  cv::imwrite(file_name + "_compare.png", output);
}