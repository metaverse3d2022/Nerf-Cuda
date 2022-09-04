/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_render.h
 *  @author Hangkun Xu
 */

#pragma once
#include <filesystem/path.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/nerf_network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/random.h>

#include <json/json.hpp>
#include <nerf-cuda/common_device.cuh>

NGP_NAMESPACE_BEGIN

class NerfRender {
 public:
  NerfRender();
  ~NerfRender();

  // load network
  // need to do : load pretrained model !
  void reload_network_from_file(const std::string& network_config_path);
  nlohmann::json load_network_config(
      const filesystem::path& network_config_path);
  void reset_network();  // reset the network according to the network config.

  // render !
  void set_resolution(Eigen::Vector2i resolution);
  Image render_frame(
      struct Camera cam, Eigen::Matrix<float, 4, 4> pos);  // render an image according to camera inner
                                    // parameters and outer parameters.

  void generate_rays(struct Camera cam, Eigen::Matrix<float, 4, 4> pos,
                     Eigen::Vector2i resolution,
                     tcnn::GPUMatrixDynamic<float>& rays_o,
                     tcnn::GPUMatrixDynamic<float>& rays_d);

  void generate_density_grid();
  void load_snapshot(const std::string& filepath_string);

 private:
  tcnn::GPUMemory<float> m_aabb;
  // Scene parameters
  float m_bound = 1;
  float m_scale = 0.33;

  // Random Number
  uint32_t m_seed = 42;
  tcnn::pcg32 m_rng;

  // density grid parameter !
  float m_density_scale=1;
  int m_dg_cascade = 1;
  int m_dg_h = 128;
  float m_dg_threshould_l = 1.e-4;
  float m_mean_density = 1.e-4;
  float m_dt_gamma = 1.0/128;
  tcnn::GPUMemory<float> m_density_grid;
  // CASCADE * H * H * H * size_of(float),
  // index calculation : cascade_level * H * H * H + nx * H * H + ny * H + nz

  // infer parameters
  int m_bg_color = 1;
  bool m_perturb = false;
  float m_min_near = 0.2;
  int m_num_thread = 128;
  int m_max_infer_steps = 1024;

  // middle variable
  Eigen::Vector2i resolution;
  // initial points corresponding to pixels, in world coordination
  tcnn::GPUMatrixDynamic<float> rays_o;
  // direction corresponding to pixels,in world coordination
  tcnn::GPUMatrixDynamic<float> rays_d;
  // Calculate rays' intersection time (near and far) with aabb
  tcnn::GPUMatrixDynamic<float> nears;
  tcnn::GPUMatrixDynamic<float> fars;
  //  allocate outputs
  tcnn::GPUMatrixDynamic<float> weight_sum;  // the accumlate weight of each ray
  tcnn::GPUMatrixDynamic<float> depth;  // output depth img
  tcnn::GPUMatrixDynamic<float> image;  // output rgb image
  // store the alive rays number
  tcnn::GPUMatrixDynamic<int> alive_counter;
  // the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
  // 2 is used to loop old/new
  tcnn::GPUMatrixDynamic<int> rays_alive;
  // the alive rays' time, we only use the first n_alive.
  // dead rays are marked by rays_t < 0
  //  2 is used to loop old/new
  tcnn::GPUMatrixDynamic<float> rays_t;
  tcnn::GPUMatrixDynamic<float> xyzs;
  // all generated points' view dirs.
  tcnn::GPUMatrixDynamic<float> dirs;
  // all generated points' deltas
  //(here we record two deltas, the first is for RGB, the second for depth).
  tcnn::GPUMatrixDynamic<float> deltas;

  // volume density
  tcnn::GPUMatrixDynamic<float> sigmas;
  // emitted color
  tcnn::GPUMatrixDynamic<float> rgbs;

  // concated input
  tcnn::GPUMatrixDynamic<float> network_input;
  // concated output
  tcnn::GPUMatrixDynamic<precision_t> network_output;

  float* deep_h;
  float* image_h;
  unsigned char* us_image;
  unsigned char* us_depth;


  // Cuda Stuff
  cudaStream_t m_inference_stream;

  // network variable
  filesystem::path m_network_config_path;
  nlohmann::json m_network_config;
  std::shared_ptr<NerfNetwork<precision_t>> m_nerf_network;
};

NGP_NAMESPACE_END