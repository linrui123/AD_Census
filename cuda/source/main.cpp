#include <iostream>
#include <fstream>
#include <istream>
#include <memory.h>
#include <opencv2/opencv.hpp>
#include "cu_ad_census.hpp"
#include "cu_ad_census_type.hpp"
#include "compare.hpp"

int main(int argc, char ** argv)
{
  std::shared_ptr<CuADCensus> p_cu_ad_census = std::make_shared<CuADCensus>();
  if (argc > 1)
  {
    p_cu_ad_census->parse_config(std::string(argv[1]));
  }

  p_cu_ad_census->init();
  p_cu_ad_census->load_image();
  p_cu_ad_census->execute();

  // std::shared_ptr<ADCensus> p_ad_census = std::make_shared<ADCensus>(std::string(argv[1]));
  // p_ad_census->calculate_cost();
  // p_ad_census->aggregation_cost();

  Compare comp;
  // comp.compare_init_costs(*p_ad_census, *p_cu_ad_census);
  // comp.compare_cross_arm(*p_ad_census, *p_cu_ad_census);
  // comp.compare_cross_arm_pixel_cnts(*p_ad_census, *p_cu_ad_census);
  // comp.compare_aggregate_costs(*p_ad_census, *p_cu_ad_census);

  p_cu_ad_census->release();
  
  return 0;
}

