/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_utils.h
 *  @author Xu Hangkun, NVIDIA
 *  @brief  functions used by nerf render
 */

#pragma once
#include <nerf-cuda/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <nerf-cuda/common_device.cuh>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

// Initialization function !

// Ray Generation
__global__ void set_rays_d(MatrixView<float> rays_d, struct Camera cam,
                           Eigen::Matrix<float, 3, 3> pose, int W, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float i = (tid % W) + 0.5;
  float j = (tid / W) + 0.5;

  float zs = 1;
  float xs = (i - cam.cx) / cam.fl_x * zs;
  float ys = (j - cam.cy) / cam.fl_y * zs;
  Eigen::Vector3f directions(xs, ys, zs);
  directions = directions / directions.norm();
  Eigen::Vector3f ray_d = pose * directions;

  if (tid < N) {
    rays_d(tid, 0) = ray_d[0];
    rays_d(tid, 1) = ray_d[1];
    rays_d(tid, 2) = ray_d[2];
  }
}

__global__ void set_rays_o(MatrixView<float> rays_o, Eigen::Vector3f ray_o,
                           int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3] @ function
  // get_rays
  if (tid < N) {
    rays_o(tid, 0) = ray_o[0];
    rays_o(tid, 1) = ray_o[1];
    rays_o(tid, 2) = ray_o[2];
  }
}

Eigen::Matrix<float, 4, 4> nerf_matrix_to_ngp(
    Eigen::Matrix<float, 4, 4> pose, float scale = 0.33,
    Eigen::Vector3f offset = Eigen::Vector3f(0, 0, 0)) {
  Eigen::Matrix<float, 4, 4> new_pose;
  new_pose << pose(1, 0), -pose(1, 1), -pose(1, 2),
      pose(1, 3) * scale + offset[0], pose(2, 0), -pose(2, 1), -pose(2, 2),
      pose(2, 3) * scale + offset[1], pose(0, 0), -pose(0, 1), -pose(0, 2),
      pose(0, 3) * scale + offset[2], 0, 0, 0, 1;
  return new_pose;
}

// Ray Marching functions

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}
inline __host__ __device__ void swapf(float& a, float& b) {
  float c = a;
  a = b;
  b = c;
}
inline __host__ __device__ float clamp(const float x, const float min,
                                       const float max) {
  return fminf(max, fmaxf(min, x));
}
inline __host__ __device__ float signf(const float x) {
  return copysignf(1.0, x);
}
inline __device__ int mip_from_pos(const float x, const float y, const float z,
                                   const float max_cascade) {
  const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
  int exponent;
  frexpf(mx, &exponent);  // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2,
                          // 4) --> 2, ...
  return fminf(max_cascade - 1, fmaxf(0, exponent));
}
inline __device__ int mip_from_dt(const float dt, const float H,
                                  const float max_cascade) {
  const float mx = dt * H * 0.5;
  int exponent;
  frexpf(mx, &exponent);
  return fminf(max_cascade - 1, fmaxf(0, exponent));
}
inline __host__ __device__ uint32_t __expand_bits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}
inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y,
                                               uint32_t z) {
  uint32_t xx = __expand_bits(x);
  uint32_t yy = __expand_bits(y);
  uint32_t zz = __expand_bits(z);
  return xx | (yy << 1) | (zz << 2);
}

inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }

__global__ void get_aabb(tcnn::MatrixView<float> aabb_in, const float bound_in,
                         const int N = 6) {
  // prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
  const uint32_t index_x = threadIdx.x + blockIdx.x * blockDim.x;
  if (index_x > N) {
    return;
  }
  if (index_x < 3) {
    aabb_in(0, index_x) = -bound_in;
  } else {
    aabb_in(0, index_x) = bound_in;
  }
}

__global__ void init_step0(tcnn::MatrixView<int> rays_alive_view,
                           tcnn::MatrixView<float> rays_t_view, int n_alive,
                           tcnn::MatrixView<float> near_view) {
  // init the rays_alive and rays_t at the first step of the loop
  const uint32_t index_x = threadIdx.x + blockIdx.x * blockDim.x;
  if (index_x > n_alive) {
    return;
  }
  rays_alive_view(0, index_x) = index_x;
  rays_t_view(0, index_x) = near_view(0, index_x);
}

__global__ void get_image_and_depth(
    tcnn::MatrixView<float> image_view,        // n,3
    tcnn::MatrixView<float> depth_view,        // 1,n
    tcnn::MatrixView<float> nears_view,        // 1,n
    tcnn::MatrixView<float> fars_view,         // 1,n
    tcnn::MatrixView<float> weights_sum_view,  // 1,n
    int bg_color, const uint32_t N) {
  // get the final image and depth from render results
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;
  auto inf = std::numeric_limits<float>::infinity();
  image_view(n, 0) = image_view(n, 0) + (1 - weights_sum_view(0, n)) * bg_color;
  depth_view(0, n) = clamp(depth_view(0, n) - nears_view(0, n), 0, inf) /
                     (fars_view(0, n) - nears_view(0, n));
}

__global__ void matrix_multiply_1x1n(int a, const uint32_t N,
                                     tcnn::MatrixView<float> b) {
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  b(0, n) = a * b(0, n);
}

__global__ void concat_network_in_and_out(
    const tcnn::MatrixView<float> a,        //(c1,b)
    const tcnn::MatrixView<float> b,        //(c2,b)
    tcnn::MatrixView<float> concat_result,  //(c1+c2,b)
    const uint32_t N, const uint32_t rows_a, const uint32_t rows_b) {
  // concat two MatrixView to one
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  for (int i = 0; i < rows_a + rows_b; ++i) {
    if (i < rows_a) {
      concat_result(i, n) = a(i, n);
    } else {
      concat_result(i, n) = b(i - rows_a, n);
    }
  }
}

__global__ void decompose_network_in_and_out(
    tcnn::MatrixView<float> a,                    //(1,b)
    tcnn::MatrixView<float> b,                    //(b,c2)
    tcnn::MatrixView<precision_t> concat_result,  //(1+c2,b)
    const uint32_t N, const uint32_t rows_a, const uint32_t cols_b) {
  // decompose one MatrixView to two
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  for (int i = 0; i < rows_a + cols_b; ++i) {
    if (i < rows_a) {
      a(i, n) = concat_result(i, n);
    } else {
      b(n, i - rows_a) = concat_result(i, n);
    }
  }
}

// rays_o/d: [N, 3]
// nears/fars: [N]
__global__ void kernel_near_far_from_aabb(
    tcnn::MatrixView<float> rays_o, tcnn::MatrixView<float> rays_d,
    tcnn::MatrixView<float> aabb, const uint32_t N, const float min_near,
    tcnn::MatrixView<float> nears, tcnn::MatrixView<float> fars) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  // rays_o += n * 3;
  // rays_d += n * 3;

  const float ox = rays_o(n, 0), oy = rays_o(n, 1), oz = rays_o(n, 2);
  const float dx = rays_d(n, 0), dy = rays_d(n, 1), dz = rays_d(n, 2);
  const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

  // get near far (assume cube scene)
  float near = (aabb(0, 0) - ox) * rdx;
  float far = (aabb(0, 3) - ox) * rdx;
  if (near > far) swapf(near, far);

  float near_y = (aabb(0, 1) - oy) * rdy;
  float far_y = (aabb(0, 4) - oy) * rdy;
  if (near_y > far_y) swapf(near_y, far_y);

  if (near > far_y || near_y > far) {
    nears(0, n) = fars(0, n) = std::numeric_limits<float>::max();
    return;
  }

  if (near_y > near) near = near_y;
  if (far_y < far) far = far_y;

  float near_z = (aabb(0, 2) - oz) * rdz;
  float far_z = (aabb(0, 5) - oz) * rdz;
  if (near_z > far_z) swapf(near_z, far_z);

  if (near > far_z || near_z > far) {
    nears(0, n) = fars(0, n) = std::numeric_limits<float>::max();
    return;
  }

  if (near_z > near) near = near_z;
  if (far_z < far) far = far_z;

  if (near < min_near) near = min_near;

  nears(0, n) = near;
  fars(0, n) = far;
}

__global__ void kernel_compact_rays(const int n_alive,
                                    tcnn::MatrixView<int> rays_alive_view,
                                    tcnn::MatrixView<float> rays_t_view,
                                    tcnn::MatrixView<int> alive_counter_view,
                                    const int i1, const int i2) {
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= n_alive) return;

  // rays_t_old[n] < 0 means ray died in last composite kernel.
  if (rays_alive_view(i2, n) >= 0) {
    const int index = alive_counter_view(0, 0) + 1;
    rays_alive_view(i1, index) = rays_alive_view(i2, n);
    rays_t_view(i1, index) = rays_t_view(i2, n);
  }
}

__global__ void kernel_march_rays(
    const uint32_t n_alive, const uint32_t n_step,
    tcnn::MatrixView<int> rays_alive_view, tcnn::MatrixView<float> rays_t_view,
    tcnn::MatrixView<float> rays_o_view, tcnn::MatrixView<float> rays_d_view,
    const float bound, const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H, tcnn::MatrixView<uint8_t> grid,
    tcnn::MatrixView<float> nears_view, tcnn::MatrixView<float> fars_view,
    tcnn::MatrixView<float> xyzs_view, tcnn::MatrixView<float> dirs_view,
    tcnn::MatrixView<float> deltas_view, const uint32_t perturb,
    tcnn::pcg32 rng, const int i) {
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= n_alive) return;

  const int index = rays_alive_view(i % 2, n);  // ray id
  float t = rays_t_view(i % 2, n);              // current ray's t

  // locate
  // rays_o += index * 3;
  // rays_d += index * 3;
  int xyzs_loc = n * n_step;
  int dirs_loc = n * n_step;
  int deltas_loc = n * n_step;

  const float ox = rays_o_view(n, 0), oy = rays_o_view(n, 1),
              oz = rays_o_view(n, 2);
  const float dx = rays_d_view(n, 0), dy = rays_d_view(n, 1),
              dz = rays_d_view(n, 2);
  const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
  const float rH = 1 / (float)H;

  const float near = nears_view(0, index), far = fars_view(0, index);

  const float dt_min = 2 * SQRT3() / max_steps;
  const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

  // march for n_step steps, record points
  uint32_t step = 0;

  // introduce some randomness (pass in spp as perturb here)
  if (perturb) {
    rng.advance(n);
    t += dt_min * rng.next_float();
  }

  float last_t = t;

  while (t < far && step < n_step) {
    // current point
    const float x = clamp(ox + t * dx, -bound, bound);
    const float y = clamp(oy + t * dy, -bound, bound);
    const float z = clamp(oz + t * dz, -bound, bound);

    const float dt = clamp(t * dt_gamma, dt_min, dt_max);

    // get mip level
    const int level = max(mip_from_pos(x, y, z, C),
                          mip_from_dt(dt, H, C));  // range in [0, C - 1]

    const float mip_bound = fminf((float)(1 << level), bound);
    const float mip_rbound = 1 / mip_bound;

    // convert to nearest grid position
    const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
    const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
    const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

    const uint32_t index = level * H * H * H + __morton3D(nx, ny, nz);
    const bool occ = grid(0, index / 8) & (1 << (index % 8));

    // if occpuied, advance a small step, and write to output
    if (occ) {
      // write step
      xyzs_view(xyzs_loc, 0) = x;
      xyzs_view(xyzs_loc, 1) = y;
      xyzs_view(xyzs_loc, 2) = z;
      dirs_view(dirs_loc, 0) = dx;
      dirs_view(dirs_loc, 1) = dy;
      dirs_view(dirs_loc, 2) = dz;
      // calc dt
      t += dt;
      deltas_view(deltas_loc, 0) = dt;
      deltas_view(deltas_loc, 1) = t - last_t;  // used to calc depth
      last_t = t;
      // step
      xyzs_loc += 1;
      dirs_loc += 1;
      deltas_loc += 1;
      step++;

      // else, skip a large step (basically skip a voxel grid)
    } else {
      // calc distance to next voxel
      const float tx =
          (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
      const float ty =
          (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
      const float tz =
          (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
      const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
      // step until next voxel
      do {
        t += clamp(t * dt_gamma, dt_min, dt_max);
      } while (t < tt);
    }
  }
}

// Volume Render functions
__global__ void kernel_composite_rays(
    const uint32_t n_alive, const uint32_t n_step,
    tcnn::MatrixView<int> rays_alive, tcnn::MatrixView<float> rays_t,
    tcnn::MatrixView<float> sigmas, tcnn::MatrixView<float> rgbs,
    tcnn::MatrixView<float> deltas, tcnn::MatrixView<float> weights_sum,
    tcnn::MatrixView<float> depth, tcnn::MatrixView<float> image, const int i) {
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= n_alive) return;

  const int index = rays_alive(i % 2, n);  // ray id
  float t = rays_t(i % 2, n);              // current ray's t

  // locate
  int sigmas_loc = n * n_step;
  int rgbs_loc = n * n_step;
  int deltas_loc = n * n_step;

  //   weights_sum += index;
  //   depth += index;
  //   image += index * 3;

  float weight_sum = weights_sum(0, index);
  float d = depth(0, index);
  float r = image(index, 0);
  float g = image(index, 1);
  float b = image(index, 2);

  // accumulate
  uint32_t step = 0;
  while (step < n_step) {
    // ray is terminated if delta == 0
    if (deltas(deltas_loc, 0) == 0) break;

    const float alpha =
        1.0f - __expf(-sigmas(sigmas_loc, 0) * deltas(sigmas_loc, 0));

    /*
    T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
    w_i = alpha_i * T_i
    -->
    T_i = 1 - \sum_{j=0}^{i-1} w_j
    */
    const float T = 1 - weight_sum;
    const float weight = alpha * T;
    weight_sum += weight;

    t += deltas(deltas_loc, 1);  // real delta
    d += weight * t;
    r += weight * rgbs(rgbs_loc, 0);
    g += weight * rgbs(rgbs_loc, 1);
    b += weight * rgbs(rgbs_loc, 2);

    // printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n,
    // step, alpha, weight, T, sum_delta, d);

    // ray is terminated if T is too small
    // NOTE: can significantly accelerate inference!
    if (T < 1e-4) break;

    // locate
    sigmas_loc++;
    rgbs_loc += 1;
    deltas_loc += 1;
    step++;
  }

  // printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

  // rays_t = -1 means ray is terminated early.
  if (step < n_step) {
    rays_t(i % 2, n) = -1;
  } else {
    rays_t(i % 2, n) = t;
  }

  weights_sum(0, index) = weight_sum;  // this is the thing I needed!
  depth(0, index) = d;
  image(index, 0) = r;
  image(index, 1) = g;
  image(index, 2) = b;
}

NGP_NAMESPACE_END