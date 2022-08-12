/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Hangkun
 */

#include <cuda.h>
#include <filesystem/path.h>
#include <nerf-cuda/nerf_render.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>
#include <args/args.hxx>
#include <iostream>
#include <string>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

void printDeviceProp(const cudaDeviceProp& prop) {
  cout << "Device Name : " << prop.name << "\n";
  cout << "totalGlobalMem : " << prop.totalGlobalMem << "\n";
  cout << "sharedMemPerBlock " << prop.sharedMemPerBlock << "\n";
  cout << "regsPerBlock : " << prop.regsPerBlock << "\n";
  cout << "warpSize :" << prop.warpSize << "\n";
  cout << "memPitch : " << prop.memPitch << "\n";
  cout << "maxThreadsPerBlock " << prop.maxThreadsPerBlock << "\n";
  cout << "maxThreadsDim[0 - 2] : " << prop.maxThreadsDim[0] << " "
       << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << "\n";
  cout << "maxGridSize[0 - 2] " << prop.maxGridSize[0] << " "
       << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << "\n";
  cout << "totalConstMem : " << prop.totalConstMem << "\n";
  cout << "major.minor : " << prop.major << "." << prop.minor << "\n";
  cout << "clockRate : " << prop.clockRate << "\n";
  cout << "textureAlignment :" << prop.textureAlignment << "\n";
  cout << "deviceOverlap : " << prop.deviceOverlap << "\n";
  cout << "multiProcessorCount : " << prop.multiProcessorCount << "\n";
}

__global__ void add_one(float* data, const int N = 5) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > N) {
    return;
  }
  data[index] += 1;
}

__global__ void matrix_add_one(MatrixView<float> data, const int M = 5,
                               const int N = 5) {
  const uint32_t encoded_index_x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t encoded_index_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (encoded_index_x > M || encoded_index_y > N) {
    return;
  }
  data(encoded_index_x, encoded_index_y) += 1;
}

int main(int argc, char** argv) {
  cout << "Hello, Metavese!" << endl;
  NerfRender* render = new NerfRender();
  string config_path = "./configs/nerf/base.json";
  render->reload_network_from_file(config_path);  // Init Model
  render
      ->generate_density_grid();  // generate densiy grid if we do not load it.
  Camera cam = {1920, 1920, 0, 0};
  Eigen::Matrix<float, 4, 4> pos;
  pos << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  Eigen::Vector2i resolution(1920, 1080);
  render->render_frame(cam, pos, resolution);

  int deviceId;
  cudaGetDevice(&deviceId);  // `deviceId` now points to the id of the currently
                             // active GPU.

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  printDeviceProp(props);

  // // GPU memory reading and write
  // tcnn::GPUMemory<float> mom(5); // length is equal to 5.
  // std::vector<float> mom_h; // length is equal to 5.
  // mom_h.resize(5,0);
  // mom.copy_from_host(mom_h);
  // cout << "Second Value of GPUMemory<float> : " << mom_h[1] << endl;
  // add_one<<<1,5>>> (mom.data(), 5); // error operation : mom.data()[1] += 1;
  // device variable should be operated in __device__ function !!!
  // mom.copy_to_host(mom_h);
  // cout << "Second Value of GPUMemory<float> (changed!) : " << mom_h[1] <<
  // endl;

  // // GPU Matrix reading and write
  // tcnn::GPUMatrixDynamic<float> matrix(5,5);
  // matrix.initialize_constant(1.0);
  // tcnn::MatrixView<float> view = matrix.view();
  // float host_data[1] = {0};
  // cudaMemcpy(host_data, &view(1,1), 1 * sizeof(float),
  // cudaMemcpyDeviceToHost);    // copy one data to host, you can also copy a
  // list of data to host! cout << "Matrix[1,1] : " <<*host_data << endl; dim3
  // dim_grid(1,1); dim3 dim_thread(5,5); matrix_add_one<<<dim_grid,
  // dim_thread>>>(view); cudaMemcpy(host_data, &view(1,1), 1 * sizeof(float),
  // cudaMemcpyDeviceToHost); cout << "Matrix[1,1] (Changed): " <<*host_data <<
  // endl;
}
