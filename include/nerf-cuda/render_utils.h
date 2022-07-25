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
#include <nerf-cuda/common_device.cuh>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

// Initialization function !

// Ray Generation
__global__ void set_rays_d(MatrixView<float> rays_d, struct Camera cam, Eigen::Matrix<float, 3, 3> pose, int W, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float i = (tid % W) + 0.5;
    float j = (tid / W) + 0.5;
    
    float zs = 1;
    float xs = (i - cam.cx) / cam.fl_x * zs;
    float ys = (j - cam.cy) / cam.fl_y * zs;
    Eigen::Vector3f directions(xs, ys, zs);
    directions = directions / directions.norm();
    Eigen::Vector3f ray_d = pose * directions;
    
    if (tid < N){
        rays_d(tid, 0) = ray_d[0];
        rays_d(tid, 1) = ray_d[1];
        rays_d(tid, 2) = ray_d[2];
    }
}

__global__ void set_rays_o(MatrixView<float> rays_o, Eigen::Vector3f ray_o, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3] @ function get_rays
    if (tid < N){
        rays_o(tid, 0) = ray_o[0];
        rays_o(tid, 1) = ray_o[1];
        rays_o(tid, 2) = ray_o[2];
    }
}

Eigen::Matrix<float, 4, 4> nerf_matrix_to_ngp(Eigen::Matrix<float, 4, 4> pose, float scale = 0.33, Eigen::Vector3f offset = Eigen::Vector3f(0, 0, 0)) {
    Eigen::Matrix<float, 4, 4> new_pose;
    new_pose << pose(1, 0), -pose(1, 1), -pose(1, 2), pose(1, 3) * scale + offset[0],
	              pose(2, 0), -pose(2, 1), -pose(2, 2), pose(2, 3) * scale + offset[1],
                pose(0, 0), -pose(0, 1), -pose(0, 2), pose(0, 3) * scale + offset[2],
	                       0,          0 ,           0,                              1;
    return new_pose;
}

// Ray Marching functions


// Volume Render functions


NGP_NAMESPACE_END