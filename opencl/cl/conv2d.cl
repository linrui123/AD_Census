
__kernel void conv2d(
  __global const float * input
  , __global const float * weights
  , __global const float * bias
  , __global float * output
  , __private int input_h
  , __private int input_w
  , __private int input_c
  , __private int kernel_h
  , __private int kernel_w
  , __private int padding
  , __private int stride
  , __private float clip_min
  , __private float clip_max)
{
  size_t c = get_global_id(0);
  size_t h = get_global_id(1);
  size_t w = get_global_id(2);

  size_t output_h = get_global_size(1);
  size_t output_w = get_global_size(2);

  size_t hidx = kernel_h / 2 - padding + h * stride;
  size_t widx = kernel_w / 2 - padding + w * stride;

  size_t woffset = c * input_c * kernel_h * kernel_w;

  float sum = 0;
  for (size_t i = 0; i < input_c; i++)
  {
    for (size_t j = -kernel_h / 2; j <= kernel_h / 2; j++)
    {
      for (size_t k = -kernel_w / 2; k <= kernel_w / 2; k++)
      {
        if ((hidx + j >= 0) && (hidx + j < input_h) && (widx + k >= 0) && (widx + k < input_w))
        {
          sum += weights[woffset + i * kernel_h * kernel_w + (j + kernel_h / 2) * kernel_w + k + kernel_w / 2] * 
                 input[i * input_h * input_w + (hidx + j) * input_w + widx + k];
        }
      }
    }
  }
  sum += bias[c];

  if (sum < clip_min)
  {
    sum = clip_min;
  }
  else if (sum > clip_max)
  {
    sum = clip_max;
  }

  output[c * output_h * output_w + h * output_w + w] = sum; 
}