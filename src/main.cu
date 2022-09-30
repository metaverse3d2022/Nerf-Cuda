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
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

int main(int argc, char** argv) {

  cout << "Hello, Metavese!" << endl;
  NerfRender* render = new NerfRender();
  string config_path = "/home/xuhangkun/result/msgpacks/freality.msgpack";
  render->reload_network_from_file(config_path);  // Init Model
  Camera cam = {3550.115/4, 3554.515/4, 3010.45/4, 1996.027/4};
  Eigen::Matrix<float, 4, 4> pos;
  pos << -0.5575427361517304, -0.11682263918046752, 0.8218871992959822, 3.9673954052389253,
      0.8300327085486383, -0.094966079921629, 0.5495699649760266, 2.667431152445114,
      0.013849191732089516, 0.9886020001326434, 0.14991425965987268, 0.45955395816033995,
      0.0, 0.0, 0.0, 1.0;
  Eigen::Vector2i resolution(4000/4, 4000/4);
  assert(resolution[0]*resolution[1]%NGPU==0);
  render -> set_resolution(resolution);
  clock_t start_t, end_t;
  start_t = clock();
  Image img = render->render_frame(cam, pos);
  end_t = clock();
  double total_time = static_cast<double> (end_t - start_t) / 1 / CLOCKS_PER_SEC;
  printf("Process time : %f s / frame", total_time);
  // store images
  char const* deep_file_name = "./deep.png";
  char const* image_file_name = "./image.png";
  stbi_write_png(deep_file_name, img.W, img.H, 1, img.depth, img.W * 1);
  stbi_write_png(image_file_name, img.W, img.H, 3, img.rgb, img.W * 3);
}