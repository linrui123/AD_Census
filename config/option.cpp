#include "option.hpp"
#include "ad_census_type.hpp"

// 配置
ADCensusOption::ADCensusOption()
{
  m_left_image_name = "/home/librui/code/AD_Census/image/000000_10_l.png";
  m_right_image_name = "/home/librui/code/AD_Census/image/000000_10_r.png";
  m_separation = false;
  m_max_disparity = 64;
  m_census_size_width = 5;
  m_census_size_height = 5;
  m_cross_arm_max_len = 255;
  lamda_ad = 30;
  lamda_census = 10;
  cross_t1 = 10;
  cross_t2 = 10;
  cross_l1 = 12;
  cross_l2 = 10;
  tso = 15;
  p1 = 1.0;
  p2 = 1.0;
  aggregate_iter = 4;
  cost_type = static_cast<uint8_t>(CostType::FINAL_COST);
  m_resize_height = 320;
  m_resize_width = 480;
}

ADCensusOption::~ADCensusOption()
{
  std::cout << "~ADCensusOption()" << std::endl;
}

void ADCensusOption::parse_config(const std::string & config_file)
{
  std::ifstream fin;
  std::string line;
  std::string key;
  std::string value;

  fin.open(config_file, std::ios::in);
  while(std::getline(fin, line))
  {
    key.clear();
    value.clear();

    size_t pos = line.find(":");
    if (pos == std::string::npos)
    {
      continue;
    }
    
    key.assign(line.begin(), line.begin() + pos);
    value.assign(line.begin() + pos + 1, line.end());

    SET_CONFIG("left_image_name", m_left_image_name, key, value);
    SET_CONFIG("right_image_name", m_right_image_name, key, value);
    SET_CONFIG("groudtruth", m_groudtruth, key, value);
    SET_CONFIG("max_disparity", m_max_disparity, key, std::stoi(value));
    SET_CONFIG("census_size_width", m_census_size_width, key, std::stoi(value));
    SET_CONFIG("census_size_height", m_census_size_height, key, std::stoi(value));
    SET_CONFIG("lamda_ad", lamda_ad, key, std::stoi(value));
    SET_CONFIG("lamda_census", lamda_census, key, std::stoi(value));
    SET_CONFIG("cross_t1", cross_t1, key, std::stoi(value));
    SET_CONFIG("cross_t2", cross_t2, key, std::stoi(value));
    SET_CONFIG("cross_l1", cross_l1, key, std::stoi(value));
    SET_CONFIG("cross_l2", cross_l2, key, std::stoi(value));
    SET_CONFIG("cross_arm_max_len", m_cross_arm_max_len, key, std::stoi(value));
    SET_CONFIG("aggregate_iter", aggregate_iter, key, std::stoi(value));
    SET_CONFIG("tso", tso, key, std::stoi(value));
    SET_CONFIG("P1", p1, key, std::stof(value));
    SET_CONFIG("P2", p2, key, std::stof(value));
    SET_CONFIG("disparity_cost_type", cost_type, key, std::stoi(value));
    SET_CONFIG("separation", m_separation, key, std::stoi(value));
    SET_CONFIG("is_resize", is_resize, key, std::stod(value));
    SET_CONFIG("resize_height", m_resize_height, key, std::stoi(value));
    SET_CONFIG("resize_width", m_resize_width, key, std::stoi(value));
  }
}

