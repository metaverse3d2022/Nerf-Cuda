/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** @file   common.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Shared functionality among multiple neural-graphics-primitives components.
 */

#pragma once

#include <nerf-cuda/common.h>
#include <nerf-cuda/random_val.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>

NGP_NAMESPACE_BEGIN

using precision_t = tcnn::network_precision_t;
// using precision_t = float;

template <typename T>
__global__ void linear_transformer(const uint32_t n_elements, const T weight, const T bias, const T* input, T* n_input) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_elements) return;
  n_input[i] = (T) ((T) weight * (T) input[i] + (T) bias);
}

NGP_NAMESPACE_END
