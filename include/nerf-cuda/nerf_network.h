/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <nerf-cuda/common_device.cuh>

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void extract_density(const uint32_t n_elements,
                                const uint32_t density_stride,
                                const uint32_t rgbd_stride,
                                const T* __restrict__ density,
                                T* __restrict__ rgbd) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_elements) return;

  rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void extract_rgb(const uint32_t n_elements,
                            const uint32_t rgb_stride,
                            const uint32_t output_stride,
                            const T* __restrict__ rgbd, T* __restrict__ rgb) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_elements) return;

  const uint32_t elem_idx = i / 3;
  const uint32_t dim_idx = i - elem_idx * 3;

  rgb[elem_idx * rgb_stride + dim_idx] =
      rgbd[elem_idx * output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient(const uint32_t n_elements,
                                     const uint32_t rgbd_stride,
                                     const T* __restrict__ rgbd,
                                     const uint32_t density_stride,
                                     T* __restrict__ density) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_elements) return;

  density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
 public:
  using json = nlohmann::json;

  NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims,
              uint32_t dir_offset, const json& pos_encoding,
              const json& dir_encoding, const json& density_network,
              const json& rgb_network)
      : m_n_pos_dims{n_pos_dims},
        m_n_dir_dims{n_dir_dims},
        m_dir_offset{dir_offset},
        m_n_extra_dims{n_extra_dims} {
    m_pos_encoding.reset(tcnn::create_encoding<T>(
        n_pos_dims, pos_encoding,
        density_network.contains("otype") &&
                (tcnn::equals_case_insensitive(density_network["otype"],
                                               "FullyFusedMLP") ||
                 tcnn::equals_case_insensitive(density_network["otype"],
                                               "MegakernelMLP"))
            ? 16u
            : 8u));
    uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
    m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims,
                                                  dir_encoding, rgb_alignment));

    json local_density_network_config = density_network;
    local_density_network_config["n_input_dims"] =
        m_pos_encoding->padded_output_width();
    if (!density_network.contains("n_output_dims")) {
      local_density_network_config["n_output_dims"] = 16;
    }
    m_density_network.reset(
        tcnn::create_network<T>(local_density_network_config));

    m_rgb_network_input_width =
        tcnn::next_multiple(m_dir_encoding->padded_output_width() +
                                m_density_network->padded_output_width(),
                            rgb_alignment);

    json local_rgb_network_config = rgb_network;
    local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
    local_rgb_network_config["n_output_dims"] = 3;
    m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));

    m_infer_params_full_precision = tcnn::GPUMemory<float>(n_params());
    m_infer_params = tcnn::GPUMemory<T>(n_params());
  }

  virtual ~NerfNetwork() {}

  void inference_mixed_precision_impl(
      cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input,
      tcnn::GPUMatrixDynamic<T>& output,
      bool use_inference_params = true) override {
    uint32_t batch_size = input.n();
    tcnn::GPUMatrixDynamic<T> density_network_input{
        m_pos_encoding->padded_output_width(), batch_size, stream,
        m_pos_encoding->preferred_output_layout()};
    tcnn::GPUMatrixDynamic<T> rgb_network_input{
        m_rgb_network_input_width, batch_size, stream,
        m_dir_encoding->preferred_output_layout()};

    tcnn::GPUMatrixDynamic<T> density_network_output =
        rgb_network_input.slice_rows(0,
                                     m_density_network->padded_output_width());
    tcnn::GPUMatrixDynamic<T> rgb_network_output{
        output.data(), m_rgb_network->padded_output_width(), batch_size,
        output.layout()};

    m_pos_encoding->inference_mixed_precision(
        stream, input.slice_rows(0, m_pos_encoding->input_width()),
        density_network_input, use_inference_params);

    m_density_network->inference_mixed_precision(stream, density_network_input,
                                                 density_network_output,
                                                 use_inference_params);

    auto dir_out =
        rgb_network_input.slice_rows(m_density_network->padded_output_width(),
                                     m_dir_encoding->padded_output_width());
    m_dir_encoding->inference_mixed_precision(
        stream, input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
        dir_out, use_inference_params);

    m_rgb_network->inference_mixed_precision(
        stream, rgb_network_input, rgb_network_output, use_inference_params);

    tcnn::linear_kernel(
        extract_density<T>, 0, stream, batch_size,
        density_network_output.layout() == tcnn::AoS
            ? density_network_output.stride()
            : 1,
        output.layout() == tcnn::AoS ? padded_output_width() : 1,
        density_network_output.data(),
        output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size));
  }

  uint32_t padded_density_output_width() const {
    return m_density_network->padded_output_width();
  }

  std::unique_ptr<tcnn::Context> forward_impl(
      cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input,
      tcnn::GPUMatrixDynamic<T>* output = nullptr,
      bool use_inference_params = false,
      bool prepare_input_gradients = false) override {
    // Make sure our temporary buffers have the correct size for the given batch
    // size
    auto forward = std::make_unique<ForwardContext>();

    return forward;
  }

  void backward_impl(cudaStream_t stream, const tcnn::Context& ctx,
                     const tcnn::GPUMatrixDynamic<float>& input,
                     const tcnn::GPUMatrixDynamic<T>& output,
                     const tcnn::GPUMatrixDynamic<T>& dL_doutput,
                     tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
                     bool use_inference_params = false,
                     tcnn::EGradientMode param_gradients_mode =
                         tcnn::EGradientMode::Overwrite) override {}

  void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input,
               tcnn::GPUMatrixDynamic<T>& output,
               bool use_inference_params = true) {
    if (input.layout() != tcnn::CM) {
      throw std::runtime_error(
          "NerfNetwork::density input must be in column major format.");
    }

    uint32_t batch_size = output.n();
    tcnn::GPUMatrixDynamic<T> density_network_input{
        m_pos_encoding->padded_output_width(), batch_size, stream,
        m_pos_encoding->preferred_output_layout()};

    m_pos_encoding->inference_mixed_precision(
        stream, input.slice_rows(0, m_pos_encoding->input_width()),
        density_network_input, use_inference_params);

    m_density_network->inference_mixed_precision(stream, density_network_input,
                                                 output, use_inference_params);
  }

  std::unique_ptr<tcnn::Context> density_forward(
      cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input,
      tcnn::GPUMatrixDynamic<T>* output = nullptr,
      bool use_inference_params = false, bool prepare_input_gradients = false) {
    if (input.layout() != tcnn::CM) {
      throw std::runtime_error(
          "NerfNetwork::density_forward input must be in column major format.");
    }
    auto forward = std::make_unique<ForwardContext>();

    return forward;
  }

  void density_backward(cudaStream_t stream, const tcnn::Context& ctx,
                        const tcnn::GPUMatrixDynamic<float>& input,
                        const tcnn::GPUMatrixDynamic<T>& output,
                        const tcnn::GPUMatrixDynamic<T>& dL_doutput,
                        tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
                        bool use_inference_params = false,
                        tcnn::EGradientMode param_gradients_mode =
                            tcnn::EGradientMode::Overwrite) {
    if (input.layout() != tcnn::CM ||
        (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
      throw std::runtime_error(
          "NerfNetwork::density_backward input must be in column major "
          "format.");
    }
  }

  void set_params(T* params, T* inference_params, T* backward_params,
                  T* gradients) override {
    size_t offset = 0;
    m_density_network->set_params(params + offset, inference_params + offset,
                                  backward_params + offset, gradients + offset);
    offset += m_density_network->n_params();

    m_rgb_network->set_params(params + offset, inference_params + offset,
                              backward_params + offset, gradients + offset);
    offset += m_rgb_network->n_params();

    m_pos_encoding->set_params(params + offset, inference_params + offset,
                               backward_params + offset, gradients + offset);
    offset += m_pos_encoding->n_params();

    m_dir_encoding->set_params(params + offset, inference_params + offset,
                               backward_params + offset, gradients + offset);
    offset += m_dir_encoding->n_params();
  }

  void initialize_params(tcnn::pcg32& rnd, float* params_full_precision,
                         T* params, T* inference_params, T* backward_params,
                         T* gradients, float scale = 1) override {
    size_t offset = 0;
    m_density_network->initialize_params(
        rnd, params_full_precision + offset, params + offset,
        inference_params + offset, backward_params + offset, gradients + offset,
        scale);
    offset += m_density_network->n_params();

    m_rgb_network->initialize_params(rnd, params_full_precision + offset,
                                     params + offset, inference_params + offset,
                                     backward_params + offset,
                                     gradients + offset, scale);
    offset += m_rgb_network->n_params();

    m_pos_encoding->initialize_params(
        rnd, params_full_precision + offset, params + offset,
        inference_params + offset, backward_params + offset, gradients + offset,
        scale);
    offset += m_pos_encoding->n_params();

    m_dir_encoding->initialize_params(
        rnd, params_full_precision + offset, params + offset,
        inference_params + offset, backward_params + offset, gradients + offset,
        scale);
    offset += m_dir_encoding->n_params();
  }

  void initialize_xavier_uniform(tcnn::pcg32& rng, float scale = 1) {
    // now, we just initialize it with a constant!
    int num_params = n_params();
    std::vector<float> pfp_h(num_params, 1.0 / 32);
    std::vector<precision_t> p_h(num_params, 1.0 / 32);
    // for (int i = 0; i < num_params; ++i) {
    //   pfp_h[i] = (float)(rng.next_float());
    //   p_h[i] = (float)(rng.next_float());
    // }
    m_infer_params_full_precision.copy_from_host(pfp_h);
    m_infer_params.copy_from_host(p_h);
    initialize_params(rng, m_infer_params_full_precision.data(),
                      m_infer_params.data(), m_infer_params.data(),
                      m_infer_params.data(), m_infer_params.data());
  }

  size_t n_params() const override {
    return m_pos_encoding->n_params() + m_density_network->n_params() +
           m_dir_encoding->n_params() + m_rgb_network->n_params();
  }

  uint32_t padded_output_width() const override {
    return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
  }

  uint32_t input_width() const override {
    return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
  }

  uint32_t output_width() const override { return 4; }

  uint32_t n_extra_dims() const { return m_n_extra_dims; }

  uint32_t required_input_alignment() const override {
    return 1;  // No alignment required due to encoding
  }

  std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
    auto layers = m_density_network->layer_sizes();
    auto rgb_layers = m_rgb_network->layer_sizes();
    layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
    return layers;
  }

  uint32_t width(uint32_t layer) const override {
    if (layer == 0) {
      return m_pos_encoding->padded_output_width();
    } else if (layer < m_density_network->num_forward_activations() + 1) {
      return m_density_network->width(layer - 1);
    } else if (layer == m_density_network->num_forward_activations() + 1) {
      return m_rgb_network_input_width;
    } else {
      return m_rgb_network->width(layer - 2 -
                                  m_density_network->num_forward_activations());
    }
  }

  uint32_t num_forward_activations() const override {
    return m_density_network->num_forward_activations() +
           m_rgb_network->num_forward_activations() + 2;
  }

  std::pair<const T*, tcnn::MatrixLayout> forward_activations(
      const tcnn::Context& ctx, uint32_t layer) const override {
    const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
    if (layer == 0) {
      return {forward.density_network_input.data(),
              m_pos_encoding->preferred_output_layout()};
    } else if (layer < m_density_network->num_forward_activations() + 1) {
      return m_density_network->forward_activations(
          *forward.density_network_ctx, layer - 1);
    } else if (layer == m_density_network->num_forward_activations() + 1) {
      return {forward.rgb_network_input.data(),
              m_dir_encoding->preferred_output_layout()};
    } else {
      return m_rgb_network->forward_activations(
          *forward.rgb_network_ctx,
          layer - 2 - m_density_network->num_forward_activations());
    }
  }

  const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
    return m_pos_encoding;
  }

  const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
    return m_dir_encoding;
  }

  tcnn::json hyperparams() const override {
    json density_network_hyperparams = m_density_network->hyperparams();
    density_network_hyperparams["n_output_dims"] =
        m_density_network->padded_output_width();
    return {
        {"otype", "NerfNetwork"},
        {"pos_encoding", m_pos_encoding->hyperparams()},
        {"dir_encoding", m_dir_encoding->hyperparams()},
        {"density_network", density_network_hyperparams},
        {"rgb_network", m_rgb_network->hyperparams()},
    };
  }

  void deserialize(const json& data) {
		json::binary_t params = data["params_binary"];
    if (params.size()/sizeof(T) != n_params()) {
			throw std::runtime_error{"Can't set params because CPU buffer has the wrong size."};
		}
    std::cout << "nerf network n_params: " << n_params() << std::endl;
    std::cout << "snapshot size: " << params.size()/sizeof(T) << std::endl;
    std::cout << "size size: T: " << sizeof(T) << std::endl;
    // params allocated: params_full_precision(float) params(__half) m_params_backward(__half) m_param_gradients(__half)
    // CUDA_CHECK_THROW(cudaMemcpy(m_infer_params_full_precision.data(), params.data(), sizeof(T)*n_params(), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(m_infer_params.data(), params.data(), sizeof(T)*n_params(), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::cout << "check pos 0" << std::endl;
    parallel_for_gpu(n_params(), [params_fp=m_infer_params_full_precision.data(), params_inference=m_infer_params.data()] __device__ (size_t i) {
			params_fp[i] = (float)params_inference[i];
		});
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    tcnn::pcg32 rng = tcnn::pcg32{(uint64_t)42};
    initialize_params(rng, m_infer_params_full_precision.data(), m_infer_params.data(), 
      m_infer_params.data(), m_infer_params.data(), m_infer_params.data());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

 private:
  std::unique_ptr<tcnn::Network<T>> m_density_network;
  std::unique_ptr<tcnn::Network<T>> m_rgb_network;
  std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
  std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

  // parameters for network
  tcnn::GPUMemory<float> m_infer_params_full_precision;
  tcnn::GPUMemory<T> m_infer_params;

  uint32_t m_rgb_network_input_width;
  uint32_t m_n_pos_dims;
  uint32_t m_n_dir_dims;
  uint32_t m_n_extra_dims;  // extra dimensions are assumed to be part of a
                            // compound encoding with dir_dims
  uint32_t m_dir_offset;

  // // Storage of forward pass data
  struct ForwardContext : public tcnn::Context {
    tcnn::GPUMatrixDynamic<T> density_network_input;
    tcnn::GPUMatrixDynamic<T> density_network_output;
    tcnn::GPUMatrixDynamic<T> rgb_network_input;
    tcnn::GPUMatrix<T> rgb_network_output;

    std::unique_ptr<Context> pos_encoding_ctx;
    std::unique_ptr<Context> dir_encoding_ctx;

    std::unique_ptr<Context> density_network_ctx;
    std::unique_ptr<Context> rgb_network_ctx;
  };
};

NGP_NAMESPACE_END
