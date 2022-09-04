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
#include <nerf-cuda/common.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>
#include <args/args.hxx>
#include <iostream>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

int main(int argc, char** argv) {

  cudaSetDevice(2);
  cout << "Hello, Metavese!" << endl;
  NerfRender* render = new NerfRender();
  string config_path = "../result/nerf/test.msgpack";
  render->reload_network_from_file(config_path);  // Init Model
  Camera cam = {1375.52, 1374.49, 554.558, 965.268};
  Eigen::Matrix<float, 4, 4> pos;
  pos << 0.8926439112348871, 0.08799600283226543, 0.4420900262071262, 3.168359405609479,
      0.4464189982715247, -0.03675452191179031, -0.8940689141475064, -5.4794898611466945,
      -0.062425682580756266, 0.995442519072023, -0.07209178487538156, -0.9791660699008925,
      0.0, 0.0, 0.0, 1.0;
  Eigen::Vector2i resolution(1080, 1080);
  render -> set_resolution(resolution);
  Image img = render->render_frame(cam, pos);
  // store images
  char const* deep_file_name = "./deep.png";
  char const* image_file_name = "./image.png";
  stbi_write_png(deep_file_name, img.W, img.H, 1, img.depth, img.W * 1);
  stbi_write_png(image_file_name, img.W, img.H, 3, img.rgb, img.W * 3);
}
