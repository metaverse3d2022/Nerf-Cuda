#include <cuda.h>
#include <filesystem/directory.h>
#include <filesystem/path.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/nerf_network.h>
#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/render_utils.h>
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
#include "omp.h"

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
  m_network_config = {};
  m_inference_stream.resize(NGPU);
  for(uint64_t gpu=0;gpu<NGPU;gpu++){
    cudaSetDevice(gpu);
    cudaStreamCreate(&m_inference_stream[gpu]);

    m_density_grid.emplace_back(GPUMemory<float>(1));
    m_aabb.emplace_back(GPUMemory<float>(1));
    m_rng.emplace_back(default_rng_t{m_seed});
  }
}

NerfRender::~NerfRender() {
  delete [] deep_h;
  delete [] image_h;
  delete [] us_image;
  delete [] us_depth;
}

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
    tlog::info() << m_network_config_path.extension() ;
    if (equals_case_insensitive(m_network_config_path.extension(), "msgpack")) {
      load_snapshot(network_config_path);
      reset_network();
      for(uint64_t gpu=0;gpu<NGPU;gpu++){
        cudaSetDevice(gpu);
        m_nerf_network[gpu]->deserialize(m_network_config["snapshot"]);
      }
    } else {
      throw std::runtime_error{"Input file with wrong extension!"};
    }
  }
}

void NerfRender::reset_network() {
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

  for(uint64_t gpu=0;gpu<NGPU;gpu++){
    cudaSetDevice(gpu);
    // reset the random seed
    m_rng[gpu] = default_rng_t{m_seed};
    // Generate the network
    m_nerf_network.emplace_back(std::make_shared<NerfNetwork<precision_t>>(
        n_pos_dims, n_dir_dims, n_extra_dims,
        n_pos_dims,  // The offset of 1 comes from the dt member variable of
                        // NerfCoordinate. HACKY
        encoding_config, dir_encoding_config, network_config, rgb_network_config));
  }
}

void NerfRender::set_resolution(Eigen::Vector2i res)
{
  resolution = res;
  int N = res[0] * res[1] / NGPU;  // number of pixels
  for(uint64_t gpu=0;gpu<NGPU;gpu++){
    cudaSetDevice(gpu);
    // initial points corresponding to pixels, in world coordination
    rays_o.emplace_back(tcnn::GPUMatrixDynamic<float>(3, N, m_inference_stream[gpu], tcnn::RM));
    // direction corresponding to pixels,in world coordination
    rays_d.emplace_back(tcnn::GPUMatrixDynamic<float>(3, N, m_inference_stream[gpu], tcnn::RM));
    // Calculate rays' intersection time (near and far) with aabb
    nears.emplace_back(tcnn::GPUMatrixDynamic<float>(1, N, m_inference_stream[gpu], tcnn::RM));
    fars.emplace_back(tcnn::GPUMatrixDynamic<float>(1, N, m_inference_stream[gpu], tcnn::RM));
    //  allocate outputs
    weight_sum.emplace_back(tcnn::GPUMatrixDynamic<float>(1, N, m_inference_stream[gpu], tcnn::RM));  // the accumlate weight of each ray
    depth.emplace_back(tcnn::GPUMatrixDynamic<float>(1, N, m_inference_stream[gpu], tcnn::RM));  // output depth img
    image.emplace_back(tcnn::GPUMatrixDynamic<float>(N, 3, m_inference_stream[gpu], tcnn::RM));  // output rgb image
    // store the alive rays number
    alive_counter.emplace_back(tcnn::GPUMatrixDynamic<int>(1, 1, m_inference_stream[gpu], tcnn::RM));
    // the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
    // 2 is used to loop old/new
    rays_alive.emplace_back(tcnn::GPUMatrixDynamic<int>(2, N, m_inference_stream[gpu], tcnn::RM));
    // the alive rays' time, we only use the first n_alive.
    // dead rays are marked by rays_t < 0
    //  2 is used to loop old/new
    rays_t.emplace_back(tcnn::GPUMatrixDynamic<float>(2, N, m_inference_stream[gpu], tcnn::RM));

    xyzs.emplace_back(tcnn::GPUMatrixDynamic<float>(3, div_round_up(N, 128) * 128, m_inference_stream[gpu], tcnn::CM));
    // all generated points' view dirs.
    dirs.emplace_back(tcnn::GPUMatrixDynamic<float>(3, div_round_up(N, 128) * 128, m_inference_stream[gpu], tcnn::CM));
    // all generated points' deltas
    //(here we record two deltas, the first is for RGB, the second for depth).
    deltas.emplace_back(tcnn::GPUMatrixDynamic<float>(2, div_round_up(N, 128) * 128, m_inference_stream[gpu], tcnn::CM));

    // volume density
    sigmas.emplace_back(tcnn::GPUMatrixDynamic<float>(1, div_round_up(N, 128) * 128, m_inference_stream[gpu], tcnn::RM));
    // emitted color
    rgbs.emplace_back(tcnn::GPUMatrixDynamic<float>(div_round_up(N, 128) * 128, 3, m_inference_stream[gpu], tcnn::RM));

    // concated input
    network_input.emplace_back(tcnn::GPUMatrixDynamic<float>(m_nerf_network[gpu]->input_width(),
                                                  div_round_up(N, 128) * 128, m_inference_stream[gpu], tcnn::RM));
    // concated output
    network_output.emplace_back(tcnn::GPUMatrixDynamic<precision_t>(
          m_nerf_network[gpu]->padded_output_width(), div_round_up(N, 128) * 128, m_inference_stream[gpu], tcnn::RM));
  }
  deep_h = new float[N*NGPU];
  image_h = new float[N*NGPU * 3];
  us_image = new unsigned char [N*NGPU*3];
  us_depth = new unsigned char [N*NGPU];
}

Image NerfRender::render_frame(struct Camera cam, Eigen::Matrix<float, 4, 4> pos) {
  // cam : parameters of cam
  // pos : camera external parameters
  // resolution : [Width, Height]

  int N = resolution[0] * resolution[1] / NGPU;  // number of pixels
  std::vector<int> step(NGPU,0);          // the current march step
  std::vector<int> i(NGPU,0);             // the flag to index old and new rays
  std::vector<bool> running(NGPU,true);
  std::vector<int> num_alive(NGPU,N);  // initialize the initial number alive as N
  threads.clear();

  // functions to generate rays_o and rays_d, it takes camera parameters and
  // resolution as input
  for(uint64_t gpu=0;gpu<NGPU;gpu++){
    threads.emplace_back(std::thread([&,gpu](){
    cudaSetDevice(gpu);
    generate_rays(cam, pos, int(gpu));
    // caliucate nears and fars
    kernel_near_far_from_aabb<<<div_round_up(N, m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
        rays_o[gpu].view(), rays_d[gpu].view(), m_aabb[gpu].data(), N, m_min_near, nears[gpu].view(),
        fars[gpu].view());

    // initial weight_sum, image and depth with 0
    parallel_for_gpu(m_inference_stream[gpu], weight_sum[gpu].n_elements(), [params_fp=weight_sum[gpu].data()] __device__ (size_t i) { params_fp[i] = 0; });
    parallel_for_gpu(m_inference_stream[gpu], depth[gpu].n_elements(), [params_fp=depth[gpu].data()] __device__ (size_t i) { params_fp[i] = 0;});
    parallel_for_gpu(m_inference_stream[gpu], image[gpu].n_elements(), [params_fp=image[gpu].data()] __device__ (size_t i) { params_fp[i] = 0;});
    parallel_for_gpu(m_inference_stream[gpu], alive_counter[gpu].n_elements(), [params_fp=alive_counter[gpu].data()] __device__ (size_t i) { params_fp[i] = 0;});
    parallel_for_gpu(m_inference_stream[gpu], rays_alive[gpu].n_elements(), [params_fp=rays_alive[gpu].data()] __device__ (size_t i) {params_fp[i] = 0;});
    parallel_for_gpu(m_inference_stream[gpu], rays_t[gpu].n_elements(), [params_fp=rays_t[gpu].data()] __device__ (size_t i) { params_fp[i] = 0; });

      while(running[gpu]){
        if(step[gpu]>=m_max_infer_steps){
          running[gpu] = false;
          continue;
        }
        if (step[gpu] == 0) {
          // init rays at first step
          init_step0<<<div_round_up(num_alive[gpu], m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
              rays_alive[gpu].view(), rays_t[gpu].view(), num_alive[gpu], nears[gpu].view());
        } else {
          // initialize alive_couter's value with 0
          int tmp_value = 0;
          cudaMemcpyAsync(&alive_counter[gpu].view()(0, 0), &tmp_value, 1 * sizeof(int),
                    cudaMemcpyHostToDevice, m_inference_stream[gpu]);

          // remove dead rays and reallocate alive rays, to accelerate next ray
          // marching
          int new_i = i[gpu] % 2;
          int old_i = (i[gpu] + 1) % 2;
          kernel_compact_rays<<<div_round_up(num_alive[gpu], m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
              num_alive[gpu], rays_alive[gpu].view(), rays_t[gpu].view(), alive_counter[gpu].view(),
              new_i, old_i);
          cudaMemcpyAsync(&num_alive[gpu], &alive_counter[gpu].view()(0, 0), 1 * sizeof(int),
                    cudaMemcpyDeviceToHost, m_inference_stream[gpu]);
        }
        if (num_alive[gpu] <= 0) {
          running[gpu] = false;  // exit loop if no alive rays
          continue;
        }

        // decide compact_steps
        int num_step = max(min(N / num_alive[gpu], 8), 1);
        // round it to the multiply of 128
        int step_x_alive = div_round_up(num_alive[gpu] * num_step, 128) * 128;

        // march rays
        kernel_march_rays<<<div_round_up(num_alive[gpu], m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
            num_alive[gpu], num_step, rays_alive[gpu].view(), rays_t[gpu].view(), rays_o[gpu].view(),
            rays_d[gpu].view(), m_bound, m_dt_gamma, m_dg_cascade, m_dg_h,
            m_density_grid[gpu].data(), m_mean_density, nears[gpu].view(), fars[gpu].view(),
            xyzs[gpu].view(), dirs[gpu].view(), deltas[gpu].view(), m_perturb, i[gpu]);

        tcnn::linear_kernel(linear_transformer<float>, 0, m_inference_stream[gpu], step_x_alive * 3,
          1.0/(2 * m_bound), 0.5, xyzs[gpu].data(), xyzs[gpu].data());
        tcnn::linear_kernel(linear_transformer<float>, 0, m_inference_stream[gpu], step_x_alive * 3,
          0.5, 0.5, dirs[gpu].data(), dirs[gpu].data());
        concat_network_in_and_out<<<div_round_up(step_x_alive, m_num_thread),
                                    m_num_thread, 0, m_inference_stream[gpu]>>>(
            xyzs[gpu].view(), dirs[gpu].view(),
            network_input[gpu].view(), step_x_alive, xyzs[gpu].rows(), dirs[gpu].rows());
        // forward through the network
        m_nerf_network[gpu]->inference_mixed_precision_impl(
            m_inference_stream[gpu], network_input[gpu], network_output[gpu]);

        // decompose network output
        decompose_network_in_and_out<<<div_round_up(step_x_alive, m_num_thread),
                                      m_num_thread, 0, m_inference_stream[gpu]>>>(
            sigmas[gpu].view(), rgbs[gpu].view(), network_output[gpu].view(), step_x_alive,
            sigmas[gpu].rows(), rgbs[gpu].cols());
        if(m_density_scale!=1) matrix_multiply_1x1n<<<div_round_up(step_x_alive, m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
            m_density_scale, step_x_alive, sigmas[gpu].view());

        // composite rays
        kernel_composite_rays<<<div_round_up(num_alive[gpu], m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
            num_alive[gpu], num_step, rays_alive[gpu].view(), rays_t[gpu].view(), sigmas[gpu].view(),
            rgbs[gpu].view(), deltas[gpu].view(), weight_sum[gpu].view(), depth[gpu].view(),
            image[gpu].view(), i[gpu]);
        step[gpu] += num_step;
        i[gpu] += 1;
      }
      // std::cout << "get image and depth" << std::endl;
      // get final image and depth
      get_image_and_depth<<<div_round_up(N, m_num_thread), m_num_thread, 0, m_inference_stream[gpu]>>>(
          image[gpu].view(), depth[gpu].view(), nears[gpu].view(), fars[gpu].view(), weight_sum[gpu].view(),
          m_bg_color, N);

      int offset = gpu*N;
      cudaMemcpyAsync(deep_h+offset, &depth[gpu].view()(0, 0), N * sizeof(float), cudaMemcpyDeviceToHost), m_inference_stream[gpu];
      cudaMemcpyAsync(image_h+3*offset, &image[gpu].view()(0, 0), N * sizeof(float) * 3, cudaMemcpyDeviceToHost, m_inference_stream[gpu]);
      cudaStreamSynchronize(m_inference_stream[gpu]);
    // }
    // for(uint64_t gpu=0;gpu<NGPU;gpu++){
      // #pragma omp parallel for num_threads(8)
      for (int i = 0; i < N ; i++) {
        int in_i  = gpu*N+i;
        int out_i = NGPU*i+gpu;
        us_depth[out_i] = (unsigned char) (255.0 * deep_h[in_i]);
        us_image[out_i*3] = (unsigned char) (255.0 * image_h[in_i*3]); 
        us_image[out_i*3+1] = (unsigned char) (255.0 * image_h[in_i*3+1]); 
        us_image[out_i*3+2] = (unsigned char) (255.0 * image_h[in_i*3+2]); 
      }
    // }
        }));}
    for(auto &thread: threads) thread.join();
    cudaDeviceSynchronize();

  Image img(resolution[0], resolution[1], us_image, us_depth);
  return img;
}

void NerfRender::generate_rays(struct Camera cam, Eigen::Matrix<float, 4, 4> pos, int threadid=-1) {

  int N = resolution[0] * resolution[1] / NGPU;  // number of pixels
  // std::cout << "N: " << N << std::endl;

  Eigen::Matrix<float, 4, 4> new_pose = nerf_matrix_to_ngp(pos, m_scale);

  int grid_size = ((N + m_num_thread) / m_num_thread);

  if(threadid==-1) for(uint64_t gpu=0;gpu<NGPU;gpu++){
    cudaSetDevice(gpu);
    set_rays_o<<<grid_size, m_num_thread, 0, m_inference_stream[gpu]>>>(rays_o[gpu].view(), new_pose.block<3, 1>(0, 3), N);
    set_rays_d<<<grid_size, m_num_thread, 0, m_inference_stream[gpu]>>>(rays_d[gpu].view(), cam, new_pose.block<3, 3>(0, 0), resolution[0], N, gpu);
  }else{
    set_rays_o<<<grid_size, m_num_thread, 0, m_inference_stream[threadid]>>>(rays_o[threadid].view(), new_pose.block<3, 1>(0, 3), N);
    set_rays_d<<<grid_size, m_num_thread, 0, m_inference_stream[threadid]>>>(rays_d[threadid].view(), cam, new_pose.block<3, 3>(0, 0), resolution[0], N, threadid);
  }
}

void NerfRender::generate_density_grid() {
  const uint32_t H = m_dg_h;
	const uint32_t H2= H*H;
	const uint32_t H3= H*H*H;
	const float decay = 0.95;
	std::vector<float> tmpV(m_dg_cascade*H3, 1.0/64);
  cudaSetDevice(0);

  m_density_grid[0].resize_and_copy_from_host(tmpV,tmpV.size());
	std::cout << "dg size: " << m_density_grid[0].size() << std::endl;
	tcnn::GPUMatrixDynamic<float> xyzs(3,H3);
	tcnn::GPUMatrixDynamic<float> cas_xyzs(3,H3);
	tcnn::GPUMatrixDynamic<precision_t> density_out(16,H3);
	tcnn::GPUMatrixDynamic<float> tmp_density(1,H3);

	int block_size = H;
  int grid_size = 3*H3/block_size;

	init_xyzs <<<grid_size, block_size, 0, m_inference_stream[0]>>> (xyzs.data(), 3*H3);

	for(int cas=0;cas<m_dg_cascade;cas++){
		float bound = 1<<cas < m_bound ? 1<<cas : m_bound;
		float half_grid_size = bound/H;

		dd_scale   <<<grid_size, block_size, 0, m_inference_stream[0]>>> (xyzs.data(), cas_xyzs.data(), 3*H3, bound-half_grid_size);
		add_random <<<grid_size, block_size, 0, m_inference_stream[0]>>> (cas_xyzs.data(), m_rng[0], 3*H3, half_grid_size);

		// m_nerf_network->density(nullptr,cas_xyzs,density_out);

		dd_scale  <<<H2, H, 0, m_inference_stream[0]>>> (density_out.slice_rows(0, 1).data(), tmp_density.data(), H3, 0.001691);
	}

	dg_update <<<H2, H, 0, m_inference_stream[0]>>> (m_density_grid[0].data(), tmp_density.data(), decay, H3);

#if NGPU > 1
  m_density_grid[0].copy_to_host(tmpV);
  for(uint64_t gpu=1;gpu<NGPU;gpu++){
    cudaSetDevice(gpu);
    m_density_grid[gpu].resize_and_copy_from_host(tmpV,tmpV.size());
  }
#endif
}

void NerfRender::load_snapshot(const std::string& filepath_string){
  tlog::info() << "Reading snapshot";
	auto config = load_network_config(filepath_string);
	if(!config.contains("snapshot")){
		throw std::runtime_error{"File " + filepath_string + " does not contain a snapshot."};
	}

	const auto& snapshot = config["snapshot"];
	// here not to check snapshot version, tmp==1

  std::vector<float> tmp_aabb(snapshot["aabb"].size(), 0);
  for(int i=0;i<snapshot["aabb"].size();i++){
    tmp_aabb[i] = snapshot["aabb"].at(i);
  }
	m_bound = snapshot.value("bound", m_bound);
	m_scale = snapshot.value("scale", m_scale);
  m_dg_cascade = snapshot.value("cascade", m_dg_cascade);
  m_dg_h = snapshot.value("density_grid_size", m_dg_h);
  m_mean_density = snapshot.value("mean_density", m_mean_density);
  std::vector<float> tmp_density_grid(snapshot["density_grid"].size(), 0);
  for(int i=0;i<snapshot["density_grid"].size();i++){
    tmp_density_grid[i] = snapshot["density_grid"].at(i);
  }
  tlog::info() << tmp_density_grid[66 * m_dg_h * m_dg_h + 66 * m_dg_h + 66] 
    << "\t" << tmp_density_grid[66 * m_dg_h * m_dg_h + 66 * m_dg_h + 67]
    << "\t" << tmp_density_grid[66 * m_dg_h * m_dg_h + 66 * m_dg_h + 68]
    << "\tDG";
  for(uint64_t gpu=0;gpu<NGPU;gpu++){
    cudaSetDevice(gpu);
    m_aabb[gpu].resize_and_copy_from_host(tmp_aabb);
    m_density_grid[gpu].resize_and_copy_from_host(tmp_density_grid);
  }
  float host_data[3] = {0};
  cudaMemcpy(host_data, &m_density_grid[0].data()[66 * m_dg_h * m_dg_h + 66 * m_dg_h + 66], 3 * sizeof(float), cudaMemcpyDeviceToHost);
  tlog::info() << "density grid : " << host_data[0] << "\t" << host_data[1] << "\t" << host_data[2];

	if (m_density_grid[0].size() != m_dg_h * m_dg_h * m_dg_h * m_dg_cascade) {
		throw std::runtime_error{"Incompatible number of grid cascades."};
	}

	m_network_config_path = filepath_string;
	m_network_config = config;
}

NGP_NAMESPACE_END
