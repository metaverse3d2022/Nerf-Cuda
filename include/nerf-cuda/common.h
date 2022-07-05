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

// lightweight log
#include <tinylogger/tinylogger.h>

// Eigen uses __device__ __host__ on a bunch of defaulted constructors.
// This doesn't actually cause unwanted behavior, but does cause NVCC
// to emit this diagnostic.
// nlohmann::json produces a comparison with zero in one of its templates,
// which can also safely be ignored.
#if defined(__NVCC__)
#  if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#    pragma nv_diag_suppress = unsigned_compare_with_zero
#  else
#    pragma diag_suppress = esa_on_defaulted_function_ignored
#    pragma diag_suppress = unsigned_compare_with_zero
#  endif
#endif
// Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
#include <Eigen/Dense>

#define NGP_NAMESPACE_BEGIN namespace ngp {
#define NGP_NAMESPACE_END }

#if defined(__CUDA_ARCH__)
	#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
		#define NGP_PRAGMA_UNROLL _Pragma("unroll")
		#define NGP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
	#else
		#define NGP_PRAGMA_UNROLL #pragma unroll
		#define NGP_PRAGMA_NO_UNROLL #pragma unroll 1
	#endif
#else
	#define NGP_PRAGMA_UNROLL
	#define NGP_PRAGMA_NO_UNROLL
#endif

NGP_NAMESPACE_BEGIN

using Vector2i32 = Eigen::Matrix<uint32_t, 2, 1>;
using Vector3i16 = Eigen::Matrix<uint16_t, 3, 1>;
using Vector4i16 = Eigen::Matrix<uint16_t, 4, 1>;
using Vector4i32 = Eigen::Matrix<uint32_t, 4, 1>;

struct Camera {
    // Linear camera parameter !
    float fl_x;
    float fl_y;
    float cx;
    float cy;
};

NGP_NAMESPACE_END
