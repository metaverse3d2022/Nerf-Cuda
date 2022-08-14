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
  m_density_grid = GPUMemory<float>(m_dg_cascade * m_dg_h * m_dg_h * m_dg_h);
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
    if(equals_case_insensitive(m_network_config_path.extension(), "msgpack")){
      load_snapshot(network_config_path);
      reset_network();
      m_nerf_network->deserialize(m_network_config["snapshot"]);
    }else if(equals_case_insensitive(m_network_config_path.extension(), "json")){
      m_network_config = load_network_config(network_config_path);
      reset_network();
      generate_density_grid();
    }
  }

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

  // functions to generate rays_o and rays_d, it takes camera parameters and
  // resolution as input
  generate_rays(cam, pos, resolution, rays_o, rays_d);

  // Calculate rays' intersection time (near and far) with aabb
  tcnn::GPUMatrixDynamic<float> nears(1, N, tcnn::RM);
  tcnn::GPUMatrixDynamic<float> fars(1, N, tcnn::RM);

  // background color [3] in range [0, 1]
  int bg_color = 1;

  // bool/int, int > 0 is used as the random seed.
  bool perturb = false;

  // scale up deltas (or sigmas), to make the density grid more sharp.
  // larger value than 1 usually improves performance.
  int density_scale = 1;

  // aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
  tcnn::GPUMatrixDynamic<float> aabb(1, 6, tcnn::RM);
  if(m_aabb_v.size()==0)  get_aabb<<<1, 6>>>(aabb.view(), m_bound);
  else if(m_aabb_v.size()==6) get_aabb0<<<1, 6>>>(aabb.view(), m_aabb_v[0], m_aabb_v[5]);

  const float min_near = 0.2;  // mini scalar

  // the thread number per block
  int N_THREAD = 256;

  // caliucate nears and fars
  kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(
      rays_o.view(), rays_d.view(), aabb.view(), N, min_near, nears.view(),
      fars.view());

  //  allocate outputs
  tcnn::GPUMatrixDynamic<float> weight_sum(
      1, N, tcnn::RM);  // the accumlate weight of each ray
  tcnn::GPUMatrixDynamic<float> depth(1, N, tcnn::RM);  // output depth img
  tcnn::GPUMatrixDynamic<float> image(N, 3, tcnn::RM);  // output rgb image
  // initial weight_sum, image and depth with 0
  weight_sum.initialize_constant(0);
  depth.initialize_constant(0);
  image.initialize_constant(0);
  std::cout << "initial weight_sum, image and depth with 0" << std::endl;

  int num_alive = N;  // initialize the initial number alive as N

  // store the alive rays number
  tcnn::GPUMatrixDynamic<int> alive_counter(1, 1, tcnn::RM);
  // the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
  // 2 is used to loop old/new
  tcnn::GPUMatrixDynamic<int> rays_alive(2, num_alive, tcnn::RM);
  // the alive rays' time, we only use the first n_alive.
  // dead rays are marked by rays_t < 0
  //  2 is used to loop old/new
  tcnn::GPUMatrixDynamic<float> rays_t(2, num_alive, tcnn::RM);
  // initial with 0
  alive_counter.initialize_constant(0);
  rays_alive.initialize_constant(0);
  rays_t.initialize_constant(0);
  std::cout << "initial alive_counter rays_alive rays_t with 0" << std::endl;

  //   // all generated points' coords
  //   tcnn::GPUMatrixDynamic<float> xyzs(num_alive, 3, tcnn::RM);
  //   // all generated points' view dirs.
  //   tcnn::GPUMatrixDynamic<float> dirs(num_alive, 3, tcnn::RM);
  //   // all generated points' deltas
  //   //(here we record two deltas, the first is for RGB, the second for
  //   depth). tcnn::GPUMatrixDynamic<float> deltas(num_alive, 2, tcnn::RM);

  int max_steps = 1024;  // the maxmize march steps
  int step = 0;          // the current march step
  int i = 0;             // the flag to index old and new rays
  while (step < max_steps) {
    std::cout << "marching step: " << step << std::endl;
    if (step == 0) {
      // init rays at first step
      init_step0<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
          rays_alive.view(), rays_t.view(), num_alive, nears.view());
    } else {
      // initialize alive_couter's value with 0
      int tmp_value = 0;
      cudaMemcpy(&alive_counter.view()(0, 0), &tmp_value, 1 * sizeof(int),
                 cudaMemcpyHostToDevice);

      // remove dead rays and reallocate alive rays, to accelerate next ray
      // marching
      int new_i = i % 2;
      int old_i = (i + 1) % 2;
      kernel_compact_rays<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
          num_alive, rays_alive.view(), rays_t.view(), alive_counter.view(),
          new_i, old_i);

      cudaMemcpy(&num_alive, &alive_counter.view()(0, 0), 1 * sizeof(int),
                 cudaMemcpyDeviceToHost);
    }
    // std::cout << "num_alives " << num_alive << std::endl;

    if (num_alive <= 0) {
      break;  // exit loop if no alive rays
    }

    // decide compact_steps
    int num_step = max(min(N / num_alive, 8), 1);
    // round it to the multiply of 128
    int step_x_alive = div_round_up(num_alive * num_step, 128) * 128;

    //   all generated points' coords
    tcnn::GPUMatrixDynamic<float> xyzs(step_x_alive, 3, tcnn::RM);
    // all generated points' view dirs.
    tcnn::GPUMatrixDynamic<float> dirs(step_x_alive, 3, tcnn::RM);
    // all generated points' deltas
    //(here we record two deltas, the first is for RGB, the second for depth).
    tcnn::GPUMatrixDynamic<float> deltas(step_x_alive, 2, tcnn::RM);

    // init it
    xyzs.initialize_constant(0);
    dirs.initialize_constant(0);
    deltas.initialize_constant(0);
    // march rays
    std::cout << "march rays" << std::endl;
    kernel_march_rays<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
        num_alive, num_step, rays_alive.view(), rays_t.view(), rays_o.view(),
        rays_d.view(), m_bound, m_dg_threshould_l, m_dg_cascade, m_dg_h,
        m_density_grid.data(), mean_density, nears.view(), fars.view(),
        xyzs.view(), dirs.view(), deltas.view(), perturb, i);
    std::cout << "march rays done!" << std::endl;

    // volume density
    tcnn::GPUMatrixDynamic<float> sigmas(1, step_x_alive, tcnn::RM);
    // emitted color
    tcnn::GPUMatrixDynamic<float> rgbs(step_x_alive, 3, tcnn::RM);

    // concated input
    tcnn::GPUMatrixDynamic<float> network_input(m_nerf_network->input_width(),
                                                step_x_alive, tcnn::RM);
    // concated output
    tcnn::GPUMatrixDynamic<precision_t> network_output(
        m_nerf_network->padded_output_width(), step_x_alive, tcnn::RM);

    concat_network_in_and_out<<<div_round_up(step_x_alive, N_THREAD),
                                N_THREAD>>>(
        xyzs.transposed().view(), dirs.transposed().view(),
        network_input.view(), step_x_alive, xyzs.cols(), dirs.cols());
    std::cout << "inference" << std::endl;
    // forward through the network
    m_nerf_network->inference_mixed_precision_impl(
        m_inference_stream, network_input, network_output);
    std::cout << "inference done" << std::endl;

    // decompose network output
    decompose_network_in_and_out<<<div_round_up(step_x_alive, N_THREAD),
                                   N_THREAD>>>(
        sigmas.view(), rgbs.view(), network_output.view(), step_x_alive,
        sigmas.rows(), rgbs.cols());

    matrix_multiply_1x1n<<<div_round_up(step_x_alive, N_THREAD), N_THREAD>>>(
        density_scale, step_x_alive, sigmas.view());

    std::cout << "composite rays" << std::endl;
    // composite rays
    kernel_composite_rays<<<div_round_up(num_alive, N_THREAD), N_THREAD>>>(
        num_alive, num_step, rays_alive.view(), rays_t.view(), sigmas.view(),
        rgbs.view(), deltas.view(), weight_sum.view(), depth.view(),
        image.view(), i);
    // std::cout << "composite rays done" << std::endl;
    step += num_step;
    i += 1;
  }
  std::cout << "get image and depth" << std::endl;
  // get final image and depth
  get_image_and_depth<<<div_round_up(N, N_THREAD), N_THREAD>>>(
      image.view(), depth.view(), nears.view(), fars.view(), weight_sum.view(),
      bg_color, N);

  float* deep_h = new float[N];
  float* image_h = new float[N * 3];

  cudaMemcpy(deep_h, &depth.view()(0, 0), N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(image_h, &image.view()(0, 0), N * sizeof(float) * 3,
             cudaMemcpyDeviceToHost);

  std::cout << "deep  " << deep_h[1] << std::endl;
  std::cout << "image " << image_h[1] << std::endl;
  // store images
  char const* deep_file_name = "./deep.png";
  char const* image_file_name = "./image.png";
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

void NerfRender::generate_density_grid() {
  // Jiang Wei
  // once the pretrained model is loaded! we can generate the density grid.
  const uint32_t H = m_dg_h;
	const uint32_t H2= H*H;
	// const uint32_t H3= H*H;//*H;
	const uint32_t H3= H*H*H;
	const float decay = 0.95;
	std::cout << "dg size: " << m_density_grid.size() << std::endl;
	std::vector<float> tmpV(m_dg_cascade*H3, 1.0/64);
	std::cout << "tmpV size: " << tmpV.size() << std::endl;
	m_density_grid.resize(m_dg_cascade*H3);
	m_density_grid.copy_from_host(tmpV);
	
	std::cout << "dg size: " << m_density_grid.size() << std::endl;
	tcnn::GPUMatrixDynamic<float> xyzs(3,H3);
	tcnn::GPUMatrixDynamic<float> cas_xyzs(3,H3);
	tcnn::GPUMatrixDynamic<precision_t> density_out(16,H3);
	tcnn::GPUMatrixDynamic<float> tmp_density(1,H3);

	int block_size = H;
  int grid_size = 3*H3/block_size;

	init_xyzs <<<grid_size, block_size>>> (xyzs.data(), 3*H3);

	for(int cas=0;cas<m_dg_cascade;cas++){
		float bound = 1<<cas < m_bound ? 1<<cas : m_bound;
		float half_grid_size = bound/H;

		dd_scale   <<<grid_size, block_size>>> (xyzs.data(), cas_xyzs.data(), 3*H3, bound-half_grid_size);
		add_random <<<grid_size, block_size>>> (cas_xyzs.data(), m_rng, 3*H3, half_grid_size);

		// m_nerf_network->density(nullptr,cas_xyzs,density_out);

		dd_scale  <<<H2, H>>> (density_out.slice_rows(0, 1).data(), tmp_density.data(), H3, 0.001691);
	}

	dg_update <<<H2, H>>> (m_density_grid.data(), tmp_density.data(), decay, H3);
}

void NerfRender::load_snapshot(const std::string& filepath_string){
  tlog::info() << "Reading snapshot";
	auto config = load_network_config(filepath_string);
	if(!config.contains("snapshot")){
		throw std::runtime_error{"File " + filepath_string + " does not contain a snapshot."};
	}

	const auto& snapshot = config["snapshot"];
	// here not to check snapshot version, tmp==1

  m_aabb_v = snapshot.value("aabb_v", m_aabb_v);
  for(int i=0;i<m_aabb_v.size();i++){
    std::cout << "aabb: " << m_aabb_v[i] << std::endl;
  }
	m_bound = snapshot["nerf"]["aabb_scale"];
  m_dg_cascade = 0;
	while ((1 << m_dg_cascade) < m_bound) {
		++m_dg_cascade;
	}
  ++m_dg_cascade;
	// m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

	if (snapshot["density_grid_size"] != m_dg_h) {
		throw std::runtime_error{"Incompatible grid size."};
	}

  nlohmann::json::binary_t density_grid_fp = snapshot["density_grid_binary_jw"];
  const uint32_t dg_size = density_grid_fp.size()/sizeof(float);
  std::cout << "Snapshot density grid size: " << dg_size << std::endl;
  tlog::info() << "Snapshot bytes: " << density_grid_fp.size();
	// GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
	// // // auto density_grid_fp16 = snapshot["density_grid_binary"];
	// std::cout << "Snapshot density grid size: " << density_grid_fp16.size() << std::endl;
	m_density_grid.resize(dg_size);
  tlog::info() << "Resize dg bytes: " << m_density_grid.bytes();
  CUDA_CHECK_THROW(cudaMemcpy(m_density_grid.data(),density_grid_fp.data(),density_grid_fp.size(),cudaMemcpyHostToDevice));
	// parallel_for_gpu(dg_size, [density_grid=m_density_grid.data(), density_grid_fp16=density_grid_fp.data()] __device__ (size_t i) {
	// 	density_grid[i] = (float)density_grid_fp16[i];
	// });

	if (m_density_grid.size() != m_dg_h * m_dg_h * m_dg_h * m_dg_cascade) {
		throw std::runtime_error{"Incompatible number of grid cascades."};
	}

	m_network_config_path = filepath_string;
	m_network_config = config;


}

NGP_NAMESPACE_END
