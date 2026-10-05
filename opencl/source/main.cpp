#include "CL/cl.h"
#include "CL/Utils/Context.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <random>

const int INPUT_H = 224;
const int INPUT_W = 224;
const int INPUT_C = 3;
const int OUTPUT_C = 32;
const int KERNEL_H = 3;
const int KERNEL_W = 3;
const int PADDING = 1;
const int STRIDE = 2;

#define CHECK_ERROR(err, code, func) func;\
{\
  if (err)\
  {\
    std::cout << #func << ": " << code \
              << ", " << __LINE__ \
              << "," << __FUNCTION__ << std::endl;\
    return 1;\
  }\
}

void program_source(std::string & file_name, std::string & program_code)
{
  std::fstream istream;
  try
  {
    istream.open(file_name, std::ios::in);
    std::string strs;
    while (getline(istream, strs))
    {
      program_code += strs;
      program_code.push_back('\n');
    }
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
}

/// @brief 
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char ** argv)
{

  std::string cl_file;
  if (argc > 1)
  {
    cl_file.assign(argv[1]);
  }
  else
  {
    cl_file.assign("../src/conv2d.cl");
  }

  cl_uint platform_nums;
  cl_int errcode;

  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    errcode = clGetPlatformIDs(0, NULL, &platform_nums));
  
  std::vector<cl_platform_id> platforms(platform_nums);
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    errcode = clGetPlatformIDs(platform_nums, platforms.data(), NULL));

  int platform_id = -1;
  for (int i = 0; i < platform_nums; i++)
  {
    size_t size;
    CHECK_ERROR(errcode != CL_SUCCESS, 
      errcode, 
      errcode = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &size));
    
    std::string name;
    name.reserve(size);
    CHECK_ERROR(errcode != CL_SUCCESS, 
      errcode, 
      errcode = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, (void *)name.data(), NULL));

    if (strstr(name.c_str(), "NVIDIA"))
    {
      platform_id = i;
      break;
    }
  }

  cl_uint device_nums;
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    errcode = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_GPU, 0, NULL, &device_nums));
  
  std::vector<cl_device_id> devices(device_nums);
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    errcode = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_GPU, device_nums, devices.data(), NULL));

  // for (int i = 0; i < device_nums; i++)
  // {
  //   size_t size;
  //   clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &size);

  //   char * name = new char [size];
  //   clGetDeviceInfo(devices[i], CL_DEVICE_NAME, size, name, NULL);

  //   std::cout << name << std::endl;
  // }

  cl_context context;
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    context = clCreateContext(0, device_nums, devices.data(), NULL, NULL, &errcode));

  cl_command_queue command_queue;
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    command_queue = clCreateCommandQueue(context, devices[0], 0, &errcode));

  std::vector<float> a(INPUT_C * INPUT_H * INPUT_W);
  std::vector<float> b(OUTPUT_C * INPUT_C * KERNEL_H * KERNEL_W);
  std::vector<float> c(OUTPUT_C);

  std::default_random_engine e;
  std::uniform_real_distribution<float> u(0, 1);
  e.seed(time(0));
  for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++)
  {
    a[i] = u(e);
  }
  for (int i = 0; i < OUTPUT_C * INPUT_C * KERNEL_H * KERNEL_W; i++)
  {
    b[i] = u(e);
  }
  for (int i = 0; i < OUTPUT_C; i++)
  {
    c[i] = u(e);
  }

  cl_mem input, weights, bias, output;
  size_t output_height = (INPUT_H + 2 * PADDING - KERNEL_H) / STRIDE + 1;
  size_t output_width = (INPUT_W + 2 * PADDING - KERNEL_W) / STRIDE + 1;


  uint64_t t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    input = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_C * INPUT_H * INPUT_W, a.data(), &errcode));
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    weights = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float) * OUTPUT_C * INPUT_C * KERNEL_H * KERNEL_W, b.data(), &errcode));
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    bias = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float) * OUTPUT_C, c.data(), &errcode));
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_C * output_height * output_width, NULL, &errcode));
  uint64_t t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  std::string program_code;
  program_source(cl_file, program_code);
  if (program_code.empty())
  {
    return 1;
  }

  const char * code = program_code.c_str();
  size_t code_len = program_code.length();

  cl_program program;
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    program = clCreateProgramWithSource(context, 1, (const char **)&code, &code_len, &errcode));

  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    errcode = clBuildProgram(program, device_nums, devices.data(), NULL, NULL, NULL));

  cl_kernel kernel;
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode, 
    kernel = clCreateKernel(program, "conv2d", &errcode));

  float clip_min = 0, clip_max = 6;

  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 4, sizeof(int), &INPUT_H));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 5, sizeof(int), &INPUT_W));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 6, sizeof(int), &INPUT_C));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 7, sizeof(int), &KERNEL_H));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 8, sizeof(int), &KERNEL_W));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 9, sizeof(int), &PADDING));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 10, sizeof(int), &STRIDE));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 11, sizeof(float), &clip_min));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clSetKernelArg(kernel, 12, sizeof(float), &clip_max));

  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  size_t global_size[] {OUTPUT_C, output_height, output_width};
  CHECK_ERROR(errcode != CL_SUCCESS, 
    errcode,
    errcode = clEnqueueNDRangeKernel(command_queue, kernel, 3, nullptr, global_size, nullptr, 0, nullptr, nullptr));
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "gpu inference time: " << (float)(t2 - t1) / 1000000 << " ms " << std::endl;

  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::vector<float> o(OUTPUT_C * output_height * output_width);
  CHECK_ERROR(errcode !=  CL_SUCCESS,
    errcode,
    errcode = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(float) * OUTPUT_C * output_height * output_width, o.data(), 0, nullptr, nullptr));
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "copy time: " << (float)(t2 - t1) / 1000000 << " ms" << std::endl;
  
  t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  float error = 0;
  for (int i = 0; i < OUTPUT_C; i++)
  {
    for (int j = 0; j < output_height; j++)
    {
      for (int k = 0; k < output_width; k++)
      {
        size_t hidx = KERNEL_H / 2 - PADDING + j * STRIDE;
        size_t widx = KERNEL_W / 2 - PADDING + k * STRIDE;
        size_t woffset = i * INPUT_C * KERNEL_H * KERNEL_W;
        float sum = 0;

        for (size_t si = 0; si < INPUT_C; si++)
        {
          for (size_t sj = -KERNEL_H / 2; sj <= KERNEL_H / 2; sj++)
          {
            for (size_t sk = -KERNEL_W / 2; sk <= KERNEL_W / 2; sk++)
            {
              if ((hidx + sj >= 0) && (hidx + sj < INPUT_H) && (widx + sk >= 0) && (widx + sk < INPUT_W))
              {
                sum += b[woffset + si * KERNEL_H * KERNEL_W + (sj + KERNEL_H / 2) * KERNEL_W + sk + KERNEL_W / 2] * 
                      a[si * INPUT_H * INPUT_W + (hidx + sj) * INPUT_W + widx + sk];
              }
            }
          }
        }
        sum += c[i];

        error += std::abs(sum - o[i * output_height * output_width + j * output_width + k]);
      }
    }
  }
  t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "cpu inference time: " << (float)(t2 - t1) / 1000000 << " ms" << std::endl;
  
  std::cout << "error: " << error << std::endl;

  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseMemObject(input));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseMemObject(weights));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseMemObject(bias));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseMemObject(output));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseContext(context));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseCommandQueue(command_queue));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseKernel(kernel));
  CHECK_ERROR(errcode != CL_SUCCESS, errcode, errcode = clReleaseProgram(program));

  return 0;
}