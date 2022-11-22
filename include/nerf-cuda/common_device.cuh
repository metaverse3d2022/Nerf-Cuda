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

inline NGP_HOST_DEVICE float srgb_to_linear(float srgb) {
	if (srgb <= 0.04045f) {
		return srgb / 12.92f;
	} else {
		return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f srgb_to_linear(const Eigen::Array3f& x) {
	return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

inline NGP_HOST_DEVICE float linear_to_srgb(float linear) {
	if (linear < 0.0031308f) {
		return 12.92f * linear;
	} else {
		return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f linear_to_srgb(const Eigen::Array3f& x) {
	return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline NGP_HOST_DEVICE Eigen::Vector2i image_pos(const Eigen::Vector2f& pos, const Eigen::Vector2i& resolution) {
	return pos.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMin(resolution - Eigen::Vector2i::Constant(1)).cwiseMax(0);
}

inline NGP_HOST_DEVICE uint64_t pixel_idx(const Eigen::Vector2i& pos, const Eigen::Vector2i& resolution, uint32_t img) {
	return pos.x() + pos.y() * resolution.x() + img * (uint64_t)resolution.x() * resolution.y();
}

inline NGP_HOST_DEVICE uint64_t pixel_idx(const Eigen::Vector2f& xy, const Eigen::Vector2i& resolution, uint32_t img) {
	return pixel_idx(image_pos(xy, resolution), resolution, img);
}

inline NGP_HOST_DEVICE Eigen::Array4f read_rgba(Eigen::Vector2i px, const Eigen::Vector2i& resolution, const void* pixels, EImageDataType image_data_type, uint32_t img = 0) {
	switch (image_data_type) {
		default:
			// This should never happen. Bright red to indicate this.
			return Eigen::Array4f{5.0f, 0.0f, 0.0f, 1.0f};
		case EImageDataType::Byte: {
			uint8_t val[4];
			*(uint32_t*)&val[0] = ((uint32_t*)pixels)[pixel_idx(px, resolution, img)];
			if (*(uint32_t*)&val[0] == 0x00FF00FF) {
				return Eigen::Array4f::Constant(-1.0f);
			}

			float alpha = (float)val[3] * (1.0f/255.0f);
			return Eigen::Array4f{
				srgb_to_linear((float)val[0] * (1.0f/255.0f)) * alpha,
				srgb_to_linear((float)val[1] * (1.0f/255.0f)) * alpha,
				srgb_to_linear((float)val[2] * (1.0f/255.0f)) * alpha,
				alpha,
			};
		}
		case EImageDataType::Half: {
			__half val[4];
			*(uint64_t*)&val[0] = ((uint64_t*)pixels)[pixel_idx(px, resolution, img)];
			return Eigen::Array4f{val[0], val[1], val[2], val[3]};
		}
		case EImageDataType::Float:
			return ((Eigen::Array4f*)pixels)[pixel_idx(px, resolution, img)];
	}
}

inline NGP_HOST_DEVICE Eigen::Array4f read_rgba(Eigen::Vector2f pos, const Eigen::Vector2i& resolution, const void* pixels, EImageDataType image_data_type, uint32_t img = 0) {
	return read_rgba(image_pos(pos, resolution), resolution, pixels, image_data_type, img);
}

inline NGP_HOST_DEVICE float4 to_float4(const Eigen::Array4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline NGP_HOST_DEVICE float4 to_float4(const Eigen::Vector4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

NGP_NAMESPACE_END
