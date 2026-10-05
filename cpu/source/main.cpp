#include <iostream>
#include <memory.h>
// #include <opencv2/opencv.hpp>
#include "ad_census.hpp"
#include "ad_census_type.hpp"
#include "option.hpp"


int main(int argc, char ** argv)
{
  std::shared_ptr<ADCensus> p_ad_census = std::make_shared<ADCensus>(std::string(argv[1]));
  uint64_t t1, t2;

  std::cout << "calculate cost ... " << std::endl;
  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  p_ad_census->calculate_cost();                     // 计算代价
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << (t2 - t1) / 1000000 << " ms" << std::endl;

  std::cout << "aggregation cost ... " << std::endl;
  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  p_ad_census->aggregation_cost();                   // 代价聚合
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << (t2 - t1) / 1000000 << " ms" << std::endl;

  std::cout << "scanline optimization ... " << std::endl;
  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  p_ad_census->scanline_optimization();              // 扫描线优化
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << (t2 - t1) / 1000000 << " ms" << std::endl;

  std::cout << "multistep refine ... " << std::endl;
  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  p_ad_census->multistep_refine();                   // 多步骤优化
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << (t2 - t1) / 1000000 << " ms" << std::endl;

  p_ad_census->draw_disparity_image("disparity");   // 生成视差图
  
  return 0;
}
