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
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int tid = indexWithinTheGrid; tid < N; tid += gridStride) {
    float i = (tid % W) + 0.5;
    float j = (tid / W) + 0.5;

    float zs = 1;
    float xs = (i - cam.cx) / cam.fl_x * zs;
    float ys = (j - cam.cy) / cam.fl_y * zs;
    Eigen::Vector3f directions(xs, ys, zs);
    directions = directions / directions.norm();
    Eigen::Vector3f ray_d = pose * directions;

    rays_d(0, tid) = ray_d[0];
    rays_d(1, tid) = ray_d[1];
    rays_d(2, tid) = ray_d[2];
  }
}

__global__ void set_rays_o(MatrixView<float> rays_o, Eigen::Vector3f ray_o,
                           int N) {
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int tid = indexWithinTheGrid; tid < N; tid += gridStride) {
    // rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3] @ function
    // get_rays
    rays_o(0, tid) = ray_o[0];
    rays_o(1, tid) = ray_o[1];
    rays_o(2, tid) = ray_o[2];
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

template <typename T>
__global__ void dd_scale(const T* d_s, float *d_d, const uint32_t N, const float k=1, const float b=0)
{
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid>=N) return;

	if(!d_s) d_d[tid] = k*float(d_s[tid])+b;
	else d_d[tid] = k*d_d[tid]+b;
}

template <typename T>
__global__ void init_xyzs(T* dd, const uint32_t N, const uint32_t H = 128)
{
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid>=N) return;

	const uint32_t posid = tid/3;
	const uint32_t xyzid = tid%3;
	uint32_t id;
	if(xyzid==2){
		id = posid%H;
	}else if(xyzid==1){
		id = posid%(H*H)/H;
	}else{
		id = posid/(H*H);
	}
	dd[tid] = -1.f+2.f/(H-1)*id;
}

template <typename T>
__global__ void add_random(T *dd, default_rng_t rng, const uint32_t N, const T k=1, const T b=0)
{
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid>=N) return;

	// tmp in -1~1
	// const float tmp = 2.f*random_val(rng)-1.f;
	dd[tid] += k*(2.f*random_val(rng)-1.f)+b;
}

template <typename T>
__global__ void dg_update(T *dg, T *tmp_dg, T decay, const uint32_t N)
{
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid>=N) return;

	if(dg[tid]>=0){
		dg[tid] = (dg[tid]*decay > tmp_dg[tid]) ? dg[tid]*decay : tmp_dg[tid];
	}
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

inline constexpr __device__ float DENSITY_THRESH() { return 0.01f; }
inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ int MAX_STEPS() { return 1024; }
inline constexpr __device__ float MIN_STEPSIZE() {
  return 2 * SQRT3() / MAX_STEPS();
}
inline constexpr __device__ float MIN_NEAR() { return 0.05f; }
inline constexpr __device__ float DT_GAMMA() {
  return 1.0f / 128.0f;
}  // accelerate if bound > 1 (very significant effect...)

inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }

__global__ void get_aabb(tcnn::MatrixView<float> aabb_in, const float bound_in,
                         const int N = 6) {
  // prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
  // aabb_in: float [6] (xmin, ymin, zmin, xmax, ymax, zmax)
  // bound_in: float  half of the cube side length
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
__global__ void get_aabb0(tcnn::MatrixView<float> aabb_in, const float a, const float b,
                         const int N = 6) {
  // prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
  const uint32_t index_x = threadIdx.x + blockIdx.x * blockDim.x;
  if (index_x > N) {
    return;
  }
  if (index_x < 3) {
    aabb_in(0, index_x) = a;
  } else {
    aabb_in(0, index_x) = b;
  }
}

__global__ void init_step0(tcnn::MatrixView<int> rays_alive_view,
                           tcnn::MatrixView<float> rays_t_view, int n_alive,
                           tcnn::MatrixView<float> near_view) {
  // init the rays_alive and rays_t at the first step of the loop
  // rays_alive_view: the alive rays's ID   int, [2,N]
  // rays_t_view: the alive rays's time float, [2,N]
  // n_alive: the alive rays number int
  // near_view: rays' intersection time (near) with aabb float, [N]
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int index_x = indexWithinTheGrid; index_x < n_alive;
       index_x += gridStride) {
    // initialize rays_alive[0] with a constant value
    rays_alive_view(0, index_x) = index_x;
    // initialize rays_t[0] by nears
    rays_t_view(0, index_x) = near_view(0, index_x);
  }
}

__global__ void get_image_and_depth(
    tcnn::MatrixView<float> image_view,        // n,3
    tcnn::MatrixView<float> depth_view,        // 1,n
    tcnn::MatrixView<float> nears_view,        // 1,n
    tcnn::MatrixView<float> fars_view,         // 1,n
    tcnn::MatrixView<float> weights_sum_view,  // 1,n
    int bg_color, const uint32_t N) {
  // get the final image and depth from render results
  // image_view: float [n,3]
  // depth_view: float [1,n]
  // nears_view: float [1,n]
  // fars_view: float [1,n]
  // weights_sum_view: float [1,n]
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < N; n += gridStride) {
    auto inf = std::numeric_limits<float>::infinity();
    image_view(n, 0) =
        image_view(n, 0) + (1 - weights_sum_view(0, n)) * bg_color;
    depth_view(0, n) = clamp(depth_view(0, n) - nears_view(0, n), 0, inf) /
                       (fars_view(0, n) - nears_view(0, n));
  }
}

__global__ void matrix_multiply_1x1n(int a, const uint32_t N,
                                     tcnn::MatrixView<float> b) {
  // a:the multiplier
  // N:the value number in the Matrix
  // b:the Multiplicand Matrix  float [1,n]
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < N; n += gridStride) {
    b(0, n) = a * b(0, n);
  }
}

__global__ void concat_network_in_and_out(
    const tcnn::MatrixView<float> a,        //(c1,b)
    const tcnn::MatrixView<float> b,        //(c2,b)
    tcnn::MatrixView<float> concat_result,  //(c1+c2,b)
    const uint32_t N, const uint32_t rows_a, const uint32_t rows_b) {
  // concat two MatrixView to one
  // a: float [rows_a,b] the matrix that needed to be concated
  // b: float [rows_b,b] the matrix that needed to be concated
  // concat_result: float [rows_a+rows_b,b] the matrix that needed to be
  // concated
  // N: num elements per row
  // rows_a: the rows of Matrix a
  // rows_b: the rows of Matrix b

  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < N; n += gridStride) {
    for (int i = 0; i < rows_a + rows_b; ++i) {
      if (i < rows_a) {
        concat_result(i, n) = a(i, n);
      } else {
        concat_result(i, n) = b(i - rows_a, n);
      }
    }
  }
}

__global__ void decompose_network_in_and_out(
    tcnn::MatrixView<float> a,                    //(1,b)
    tcnn::MatrixView<float> b,                    //(b,c2)
    tcnn::MatrixView<precision_t> concat_result,  //(1+c2,b)
    const uint32_t N, const uint32_t rows_a, const uint32_t cols_b) {
  // decompose one MatrixView to two
  // a: float [1,b] the matrix that get from decomposed results
  // b: float [b,cols_b] the matrix that get from decomposed results
  // concat_result: float [1+cols_b,b] the matrix that needed to be
  // decomposed
  // N: num elements per row
  // rows_a: the rows of Matrix a
  // cols_b: the cols of Matrix b

  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < N; n += gridStride) {
    for (int i = 0; i < rows_a + cols_b; ++i) {
      if (i < cols_b) {
        b(n, i) = concat_result(i, n);
      } else {
        a(i - cols_b, n) = concat_result(i, n);
      }
    }
  }
}

// rays_o/d: [N, 3]
// nears/fars: [N]
__global__ void kernel_near_far_from_aabb(
    tcnn::MatrixView<float> rays_o, tcnn::MatrixView<float> rays_d,
    float* aabb, const uint32_t N, const float min_near,
    tcnn::MatrixView<float> nears, tcnn::MatrixView<float> fars) {
  // rays_o:float, [N, 3]
  // rays_d: float, [N, 3]
  // aabb:float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
  // N: rays number
  // min_near: float, scalar
  // nears: float, [N]
  // fars: float, [N]
  //  parallel per ray
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < N; n += gridStride) {
    const float ox = rays_o(0, n), oy = rays_o(1, n), oz = rays_o(2, n);
    const float dx = rays_d(0, n), dy = rays_d(1, n), dz = rays_d(2, n);
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
      nears(0, n) = fars(0, n) = std::numeric_limits<float>::max();
      return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
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
}

__global__ void kernel_compact_rays(const int n_alive,
                                    tcnn::MatrixView<int> rays_alive_view,
                                    tcnn::MatrixView<float> rays_t_view,
                                    tcnn::MatrixView<int> alive_counter_view,
                                    const int i1, const int i2) {
  // rays_alive_view: the alive rays's ID   int, [2,N]
  // rays_t_view: the alive rays's time float, [2,N]
  // n_alive: the alive rays number int
  // alive_counter_view: alive rays number int [1,1]
  // i1,i2: int the index for new/old
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < n_alive; n += gridStride) {
    // rays_t_old[n] < 0 means ray died in last composite kernel.
    if (rays_t_view(i2, n) >= 0) {
      const int index = atomicAdd(&alive_counter_view(0, 0), 1);
      rays_alive_view(i1, index) = rays_alive_view(i2, n);
      rays_t_view(i1, index) = rays_t_view(i2, n);
    }
  }
}

__global__ void kernel_march_rays0(
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

__global__ void kernel_march_rays(
    const uint32_t n_alive, const uint32_t n_step,
    tcnn::MatrixView<int> rays_alive_view, tcnn::MatrixView<float> rays_t_view,
    tcnn::MatrixView<float> rays_o_view, tcnn::MatrixView<float> rays_d_view,
    const float bound, const float dt_gamma, const uint32_t C, const uint32_t H,
    float* grid,  // CASCADE * H * H * H * size_of(float)
    const float mean_density, tcnn::MatrixView<float> nears_view,
    tcnn::MatrixView<float> fars_view, tcnn::MatrixView<float> xyzs_view,
    tcnn::MatrixView<float> dirs_view, tcnn::MatrixView<float> deltas_view,
    const uint32_t perturb, const int i) {
  // n_alive: the alive rays number int
  // n_step: the compact steps  int
  // rays_alive_view: the alive rays's ID   int, [2,N]
  // rays_t_view: the alive rays's time float, [2,N]
  // rays_o_view:float, [3, N]
  // rays_d_view: float, [3, N]
  // bound: float, scalar
  // dt_gamma: the density threshould float
  // C: grid cascade int
  // H: grid resolution
  // grid: density grid float, [C* H* H* H]
  // mean_density: float, scalar
  // nears_view/fars_view: float, [N]
  //  xyzs: float, [3, n_alive * n_step], all generated points' coords
  //  dirs: float, [3, n_alive * n_step], all generated points' view dirs.
  //  deltas: float, [2, n_alive * n_step], all generated points' deltas
  // perturb: bool/int, int > 0 is used as the random seed.
  // i: used for index new/old  int
  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < n_alive; n += gridStride) {
    const int index = rays_alive_view(i % 2, n);  // ray id
    float t = rays_t_view(i % 2, n);              // current ray's t

    //   printf("%d kernel_march_rays : index:%d \n", n, index);
    const float density_thresh = fminf(DENSITY_THRESH(), mean_density);

    // locate
    //   rays_o += index * 3;
    //   rays_d += index * 3;
    int xyzs_loc = n * n_step;
    int dirs_loc = n * n_step;
    int deltas_loc = n * n_step;
    //   xyzs += n * n_step * 3;
    //   dirs += n * n_step * 3;
    //   deltas += n * n_step * 2;

    const float ox = rays_o_view(0, index), oy = rays_o_view(1, index),
                oz = rays_o_view(2, index);
    const float dx = rays_d_view(0, index), dy = rays_d_view(1, index),
                dz = rays_d_view(2, index);
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = nears_view(0, index), far = fars_view(0, index);

    const float dt_min = MIN_STEPSIZE();
    const float dt_max = 2 * bound / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness (pass in spp as perturb here)
    if (perturb) {
      pcg32 rng((uint64_t)n, (uint64_t)perturb);
      t += MIN_STEPSIZE() * rng.next_float();
    }

    float last_t = t;

    while (t < far && step < n_step) {
      // current point
      const float x = clamp(ox + t * dx, -bound, bound);
      const float y = clamp(oy + t * dy, -bound, bound);
      const float z = clamp(oz + t * dz, -bound, bound);

      // get mip level
      // TODO: check why using mip_from_dt...
      const int level = mip_from_pos(x, y, z, C);  // range in [0, C - 1]
      const float mip_bound = fminf(exp2f((float)level), bound);
      const float mip_rbound = 1 / mip_bound;

      // convert to nearest grid position
      const int nx =
          clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
      const int ny =
          clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
      const int nz =
          clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
      // printf("yes 2.5\n");
      const uint32_t index_density =
          level * H * H * H + nx * H * H + ny * H + nz;
      // printf("index_density:%d\n", index_density);
      const float density = grid[index_density];
      // printf("yes 3\n");
      // if occpuied, advance a small step, and write to output
      if (density > density_thresh) {
        // write step
        xyzs_view(0, xyzs_loc) = x;
        xyzs_view(1, xyzs_loc) = y;
        xyzs_view(2, xyzs_loc) = z;
        dirs_view(0, dirs_loc) = dx;
        dirs_view(1, dirs_loc) = dy;
        dirs_view(2, dirs_loc) = dz;
        // calc dt
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);
        t += dt;
        deltas_view(0, deltas_loc) = dt;
        deltas_view(1, deltas_loc) = t - last_t;  // used to calc depth
        last_t = t;
        // step
        xyzs_loc += 1;
        dirs_loc += 1;
        deltas_loc += 1;
        step++;
        //   printf("%d kernel_march_rays : xyzs_loc:%d \n", n, xyzs_loc);

        // else, skip a large step (basically skip a voxel grid)
      } else {
        // calc distance to next voxel
        const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * mip_bound - x) * rdx;
        const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * mip_bound - y) * rdy;
        const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * mip_bound - z) * rdz;
        const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
        // step until next voxel
        do {
          const float dt = clamp(t * dt_gamma, dt_min, dt_max);
          t += dt;
        } while (t < tt);
      }
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
  // n_alive: the alive rays number int
  // n_step: the compact steps  int
  // rays_alive_view: the alive rays's ID   int, [2,N]
  // rays_t_view: the alive rays's time float, [2,N]
  // sigmas: float, [n_alive * n_step,]
  // rgbs: float, [n_alive * n_step, 3]
  // deltas: float, [2, n_alive * n_step], all generated points' deltas
  // weights_sum: float, [N,], the alpha channel
  // depth: float, [N,], the depth value
  // image: float, [N, 3], the RGB channel (after multiplying alpha!)

  const uint32_t indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  // use the grid-stride-loops
  for (int n = indexWithinTheGrid; n < n_alive; n += gridStride) {
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
      if (deltas(0, deltas_loc) == 0) break;

      const float alpha =
          1.0f - __expf(-sigmas(0, sigmas_loc) * deltas(sigmas_loc, 0));

      /*
      T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
      w_i = alpha_i * T_i
      -->
      T_i = 1 - \sum_{j=0}^{i-1} w_j
      */
      const float T = 1 - weight_sum;
      const float weight = alpha * T;
      weight_sum += weight;

      t += deltas(1, deltas_loc);  // real delta
      d += weight * t;
      r += weight * rgbs(rgbs_loc, 0);
      g += weight * rgbs(rgbs_loc, 1);
      b += weight * rgbs(rgbs_loc, 2);

      // printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n",
      // n, step, alpha, weight, T, sum_delta, d);

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
}
NGP_NAMESPACE_END