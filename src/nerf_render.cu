#include <cuda.h>
#include <filesystem/directory.h>
#include <filesystem/path.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/nerf_network.h>
#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/render_utils.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <json/json.hpp>
#include <nerf-cuda/common_device.cuh>
#include <set>
#include <typeinfo>
#include <vector>

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

json merge_parent_network_config(const json& child,
                                 const fs::path& child_filename) {
  if (!child.contains("parent")) {
    return child;
  }
  fs::path parent_filename =
      child_filename.parent_path() / std::string(child["parent"]);
  tlog::info() << "Loading parent network config from: "
               << parent_filename.str();
  std::ifstream f{parent_filename.str()};
  json parent = json::parse(f, nullptr, true, true);
  parent = merge_parent_network_config(parent, parent_filename);
  parent.merge_patch(child);
  return parent;
}

NerfRender::NerfRender() {
  m_network_config = {
      {"encoding",
       {
           {"otype", "HashGrid"},
           {"n_levels", 16},
           {"n_features_per_level", 2},
           {"log2_hashmap_size", 19},
           {"base_resolution", 16},
       }},
      {"network",
       {
           {"otype", "FullyFusedMLP"},
           {"n_neurons", 64},
           {"n_layers", 2},
           {"activation", "ReLU"},
           {"output_activation", "None"},
       }},
  };
  CUDA_CHECK_THROW(cudaStreamCreate(&m_inference_stream));
}

NerfRender::~NerfRender() {}

json NerfRender::load_network_config(const fs::path& network_config_path) {
  if (!network_config_path.empty()) {
    m_network_config_path = network_config_path;
  }

  tlog::info() << "Loading network config from: " << network_config_path;

  if (network_config_path.empty() || !network_config_path.exists()) {
    throw std::runtime_error{std::string{"Network config \""} +
                             network_config_path.str() + "\" does not exist."};
  }

  json result;
  if (equals_case_insensitive(network_config_path.extension(), "json")) {
    std::ifstream f{network_config_path.str()};
    result = json::parse(f, nullptr, true, true);
    result = merge_parent_network_config(result, network_config_path);
  } else if (equals_case_insensitive(network_config_path.extension(),
                                     "msgpack")) {
    std::ifstream f{network_config_path.str(), std::ios::in | std::ios::binary};
    result = json::from_msgpack(f);
    // we assume parent pointers are already resolved in snapshots.
  }

  return result;
}

void NerfRender::reload_network_from_file(
    const std::string& network_config_path) {
  if (!network_config_path.empty()) {
    m_network_config_path = network_config_path;
  }
  m_network_config = load_network_config(m_network_config_path);
  reset_network();

  // Haoran
  // Need to do
  // 1. Load pretrained model
  // 2. Generate Density Grid, which will be used by ray sampler strategy!!!
  tcnn::pcg32 rng = tcnn::pcg32{(uint64_t)42};
  m_nerf_network->initialize_xavier_uniform(rng);
}

void NerfRender::reset_network() {
  // reset the random seed
  m_rng = default_rng_t{m_seed};

  // Default config
  json config = m_network_config;
  json& encoding_config = config["encoding"];
  json& network_config = config["network"];
  json& dir_encoding_config = config["dir_encoding"];
  json& rgb_network_config = config["rgb_network"];
  uint32_t n_dir_dims = 3;
  uint32_t n_pos_dims = 3;
  uint32_t n_extra_dims = 0;  // Now, it's set to zero but it needs furture
                              // check! By Hangkun, 2022/06/30

  // Automatically determine certain parameters if we're dealing with the
  // (hash)grid encoding
  if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") !=
      std::string::npos) {
    encoding_config["n_pos_dims"] = n_pos_dims;  // 3 dimenison input

    const uint32_t n_features_per_level =
        encoding_config.value("n_features_per_level", 2u);
    uint32_t m_num_levels = 16u;

    if (encoding_config.contains("n_features") &&
        encoding_config["n_features"] > 0) {
      m_num_levels =
          (uint32_t)encoding_config["n_features"] / n_features_per_level;
    } else {
      m_num_levels = encoding_config.value("n_levels", 16u);
    }

    // m_level_stats.resize(m_num_levels);
    // m_first_layer_column_stats.resize(m_num_levels);

    const uint32_t log2_hashmap_size =
        encoding_config.value("log2_hashmap_size", 15);

    uint32_t m_base_grid_resolution =
        encoding_config.value("base_resolution", 0);
    if (!m_base_grid_resolution) {
      m_base_grid_resolution = 1u << ((log2_hashmap_size) / n_pos_dims);
      encoding_config["base_resolution"] = m_base_grid_resolution;
    }

    float desired_resolution = 2048.0f;  // Desired resolution of the finest
                                         // hashgrid level over the unit cube

    // Automatically determine suitable per_level_scale
    float m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
    if (m_per_level_scale <= 0.0f && m_num_levels > 1) {
      m_per_level_scale =
          std::exp(std::log(desired_resolution * (float)m_bound /
                            (float)m_base_grid_resolution) /
                   (m_num_levels - 1));
      encoding_config["per_level_scale"] = m_per_level_scale;
    }

    tlog::info() << "GridEncoding: "
                 << " Nmin=" << m_base_grid_resolution
                 << " b=" << m_per_level_scale << " F=" << n_features_per_level
                 << " T=2^" << log2_hashmap_size << " L=" << m_num_levels;
  }

  // Generate the network
  m_nerf_network = std::make_shared<NerfNetwork<precision_t>>(
      n_pos_dims, n_dir_dims, n_extra_dims,
      n_pos_dims + 1,  // The offset of 1 comes from the dt member variable of
                       // NerfCoordinate. HACKY
      encoding_config, dir_encoding_config, network_config, rgb_network_config);
}

void NerfRender::render_frame(struct Camera cam, Eigen::Matrix<float, 4, 4> pos,
                              Eigen::Vector2i resolution) {
  // cam : parameters of cam
  // pos : camera external parameters
  // resolution : [Width, Height]

  int N = resolution[0] * resolution[1];  // number of pixels
  // initial points corresponding to pixels, in world coordination
  tcnn::GPUMatrixDynamic<float> rays_o(N, 3, tcnn::RM);
  // direction corresponding to pixels,in world coordination
  tcnn::GPUMatrixDynamic<float> rays_d(N, 3, tcnn::RM);

  generate_rays(cam, pos, resolution, rays_o, rays_d);

  // std::cout << "test" << std::endl;
  rays_o.initialize_constant(1.0);
  // float tmp_h[1]={0};
  // std::cout << *tmp_h << std::endl;
  // cudaMemcpy(tmp_h, &rays_o.view()(0,0), 1 * sizeof(float),
  // cudaMemcpyDeviceToHost); std::cout << *tmp_h << std::endl;

  // // calucate near and far
  tcnn::GPUMatrixDynamic<float> nears(1, N, tcnn::RM);
  tcnn::GPUMatrixDynamic<float> fars(1, N, tcnn::RM);

  int bg_color = 1;
  float bound = 1;  // scalar
  int cascade = 1 + ceil(log2(bound));
  int grid_size = 128;
  float dt_gamma = 0;  // called cone_angle in instant-ngp, exponentially
                       // accelerate ray marching if > 0. (very significant
                       // effect, but generally lead to worse performance)
  bool perturb = false;
  int density_scale =
      1;  // scale up deltas (or sigmas), to make the density grid more sharp.
          // larger value than 1 usually improves performance.
  tcnn::GPUMatrixDynamic<uint8_t> density_bitfield(
      1, cascade * pow(grid_size, 3) / 8, tcnn::RM);
  density_bitfield.initialize_constant(0);

  tcnn::GPUMatrixDynamic<float> aabb(1, 6, tcnn::RM);
  get_aabb<<<1, 6>>>(aabb.view(), bound);

  const float min_near = 0.2;
  int N_THREAD = 128;
  kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(
      rays_o.view(), rays_d.view(), aabb.view(), N, min_near, nears.view(),
      fars.view());

  // // allocate outputs
  tcnn::GPUMatrixDynamic<float> weight_sum(1, N, tcnn::RM);
  tcnn::GPUMatrixDynamic<float> depth(1, N, tcnn::RM);
  tcnn::GPUMatrixDynamic<float> image(N, 3, tcnn::RM);
  // initial weight_sum, image and depth with 0
  weight_sum.initialize_constant(0);
  depth.initialize_constant(0);
  image.initialize_constant(0);

  std::cout << "initial weight_sum, image and depth with 0" << std::endl;
  int num_alive = N;
  tcnn::GPUMatrixDynamic<int> alive_counter(1, 1, tcnn::RM);
  tcnn::GPUMatrixDynamic<int> rays_alive(2, num_alive, tcnn::RM);
  // 2 is used to loop old/new
  tcnn::GPUMatrixDynamic<float> rays_t(2, num_alive, tcnn::RM);
  // initial with 0
  alive_counter.initialize_constant(0);
  rays_alive.initialize_constant(0);
  rays_t.initialize_constant(0);
  std::cout << "initial alive_counter rays_alive rays_t with 0" << std::endl;

  int max_steps = 1024;
  int step = 0;
  int i = 0;
  while (step < max_steps) {
    std::cout << "step " << step << std::endl;
    if (step == 0) {
      // init rays at first step
      init_step0<<<1, num_alive>>>(rays_alive.view(), rays_t.view(), num_alive,
                                   nears.view());
    } else {
      int tmp_value = 0;
      cudaMemcpy(&alive_counter.view()(0, 0), &tmp_value, 1 * sizeof(int),
                 cudaMemcpyHostToDevice);
      // alive_counter(0) = 0;
      // remove dead rays and reallocate alive rays, to accelerate next ray
      // marching
      int new_i = i % 2;
      int old_i = (i + 1) % 2;
      kernel_compact_rays<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
          num_alive, rays_alive.view(), rays_t.view(), alive_counter.view(),
          new_i, old_i);

      cudaMemcpy(&num_alive, &alive_counter.view()(0, 0), 1 * sizeof(int),
                 cudaMemcpyDeviceToHost);
      // num_alive = alive_counter(0);
      std::cout << "num_alives " << num_alive << std::endl;
    }

    if (num_alive <= 0) {
      break;  // exit loop if no alive rays
    }

    // decide compact_steps
    int num_step = max(min(N / num_alive, 8), 1);

    tcnn::GPUMatrixDynamic<float> xyzs(num_alive * num_step, 3, tcnn::RM);
    tcnn::GPUMatrixDynamic<float> dirs(num_alive * num_step, 3, tcnn::RM);
    tcnn::GPUMatrixDynamic<float> deltas(num_alive * num_step, 2, tcnn::RM);
    // 2 vals, one for rgb, one for depth
    // init it
    xyzs.initialize_constant(0);
    dirs.initialize_constant(0);
    deltas.initialize_constant(0);

    tcnn::pcg32 rng = tcnn::pcg32{(uint64_t)42};  // hard coded random seed
    kernel_march_rays<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
        num_alive, num_step, rays_alive.view(), rays_t.view(), rays_o.view(),
        rays_d.view(), bound, dt_gamma, max_steps, cascade, grid_size,
        density_bitfield.view(), nears.view(), fars.view(), xyzs.view(),
        dirs.view(), deltas.view(), perturb, rng, i);

    // Forward through the network
    tcnn::GPUMatrixDynamic<float> sigmas(1, num_alive * num_step, tcnn::RM);
    tcnn::GPUMatrixDynamic<float> rgbs(num_alive * num_step, 3, tcnn::RM);

    // concat input
    tcnn::GPUMatrixDynamic<float> network_input(m_nerf_network->input_width(),
                                                num_alive * num_step, tcnn::RM);
    // concat output
    tcnn::GPUMatrixDynamic<precision_t> network_output(
        m_nerf_network->padded_output_width(), num_alive * num_step, tcnn::RM);

    concat_network_in_and_out<<<div_round_up(num_alive * num_step, N_THREAD),
                                N_THREAD>>>(
        xyzs.transposed().view(), dirs.transposed().view(),
        network_input.view(), num_alive * num_step, xyzs.cols(), dirs.cols());

    m_nerf_network->inference_mixed_precision_impl(
        m_inference_stream, network_input, network_output);

    // decompose
    decompose_network_in_and_out<<<div_round_up(num_alive * num_step, N_THREAD),
                                   N_THREAD>>>(
        sigmas.view(), rgbs.view(), network_output.view(), num_alive * num_step,
        sigmas.rows(), rgbs.cols());

    matrix_multiply_1x1n<<<div_round_up(num_alive * num_step, N_THREAD),
                           N_THREAD>>>(density_scale, num_alive * num_step,
                                       sigmas.view());
    // // sigmas = density_scale * sigmas;

    kernel_composite_rays<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
        num_alive, num_step, rays_alive.view(), rays_t.view(), sigmas.view(),
        rgbs.view(), deltas.view(), weight_sum.view(), depth.view(),
        image.view(), i);

    step += num_step;
    i += 1;
  }

  get_image_and_depth<<<div_round_up(N, N_THREAD), N_THREAD>>>(
      image.view(), depth.view(), nears.view(), fars.view(), weight_sum.view(),
      bg_color, N);

  //   char* deep_im = new char[N];
  float* deep_h = new float[N];
  std::cout << N * 3 << std::endl;
  std::cout << "image width " << resolution[0] << " image height "
            << resolution[1] << std::endl;
  float* image_h = new float[N * 3];

  cudaMemcpy(deep_h, &depth.view()(0, 0), N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(image_h, &image.view()(0, 0), N * sizeof(float) * 3,
             cudaMemcpyDeviceToHost);

  std::cout << "deep  " << deep_h[1] << std::endl;
  std::cout << "image " << image_h[1] << std::endl;

  char* const deep_file_name = "./deep.png";
  char* const image_file_name = "./image.png";
  stbi_write_png(deep_file_name, resolution[0], resolution[1], 1, deep_h,
                 sizeof(float) * resolution[0]);
  stbi_write_png(image_file_name, resolution[0], resolution[1], 3, image_h,
                 sizeof(float) * resolution[0] * 3);

  // Zilong
  // Need to do
  // 1. ray sample. refer the instant-ngp paper for details of the sample
  // strategy. You can do it from the easiest equally spaced sampling strategy
  // to the ray marching strategy.
  // 2. infer and volume rendering.
  // please refer to
  //     1. https://github.com/ashawkey/torch-ngp/blob/main/nerf/renderer.py
  //     line 318 ~ 380. espically the functions raymarching.compact_rays,
  //     raymarching.march_rays and raymarching.composite_rays. These function
  //     are places in
  //     https://github.com/ashawkey/torch-ngp/tree/main/raymarching/src.
  // 这里没有指定返回，后续可以再讨论返回图片是以什么格式。先渲染出来再说
}

void NerfRender::generate_rays(struct Camera cam,
                               Eigen::Matrix<float, 4, 4> pos,
                               Eigen::Vector2i resolution,
                               tcnn::GPUMatrixDynamic<float>& rays_o,
                               tcnn::GPUMatrixDynamic<float>& rays_d) {
  // Weixuan
  // Generate rays according to the input
  // please refer to
  //     1. https://github.com/ashawkey/torch-ngp/blob/main/nerf/provider.py
  //     function nerf_matrix_to_ngp
  //     2. https://github.com/ashawkey/torch-ngp/blob/main/nerf/utils.py
  //     function get_rays
  // use cuda to speed up

  int N = resolution[0] * resolution[1];  // number of pixels
  std::cout << "N: " << N << std::endl;

  Eigen::Matrix<float, 4, 4> new_pose = nerf_matrix_to_ngp(pos);

  int block_size = 256;
  int grid_size = ((N + block_size) / block_size);

  tcnn::MatrixView<float> rays_o_view = rays_o.view();
  set_rays_o<<<grid_size, block_size>>>(rays_o_view, new_pose.block<1, 3>(0, 3),
                                        N);

  tcnn::MatrixView<float> rays_d_view = rays_d.view();
  set_rays_d<<<grid_size, block_size>>>(
      rays_d_view, cam, new_pose.block<3, 3>(0, 0), resolution[0], N);

  float host_data[3] = {0, 0, 0};

  cudaMemcpy(host_data, &rays_o_view(0, 0), 3 * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "rays_o[0, :]: " << host_data[0] << ", " << host_data[1] << ", "
            << host_data[2] << std::endl;

  cudaMemcpy(host_data, &rays_d_view(0, 0), 3 * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "rays_d[0, :]: " << host_data[0] << ", " << host_data[1] << ", "
            << host_data[2] << std::endl;
}

void NerfRender::generate_density_grid() {
  // Jiang Wei
  // once the pretrained model is loaded! we can generate the density grid.
}

NGP_NAMESPACE_END
