#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/common_device.cuh>
#include <nerf-cuda/nerf_network.h>
#include <nerf-cuda/render_utils.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <fstream>
#include <set>
#include <vector>

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

json merge_parent_network_config(const json &child, const fs::path &child_filename) {
	if (!child.contains("parent")) {
		return child;
	}
	fs::path parent_filename = child_filename.parent_path() / std::string(child["parent"]);
	tlog::info() << "Loading parent network config from: " << parent_filename.str();
	std::ifstream f{parent_filename.str()};
	json parent = json::parse(f, nullptr, true, true);
	parent = merge_parent_network_config(parent, parent_filename);
	parent.merge_patch(child);
	return parent;
}

NerfRender::NerfRender()
{
    m_network_config = {
		{"encoding", {
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"n_neurons", 64},
			{"n_layers", 2},
			{"activation", "ReLU"},
			{"output_activation", "None"},
		}},
	};
    
    CUDA_CHECK_THROW(cudaStreamCreate(&m_inference_stream));
    m_rng = tcnn::pcg32(m_seed);
    m_density_grid = tcnn::GPUMemory<float> (m_dg_cascade * m_dg_h * m_dg_h * m_dg_h);
}

NerfRender::~NerfRender()
{
}

json NerfRender::load_network_config(const fs::path& network_config_path) {
	if (!network_config_path.empty()) {
		m_network_config_path = network_config_path;
	}

	tlog::info() << "Loading network config from: " << network_config_path;

	if (network_config_path.empty() || !network_config_path.exists()) {
		throw std::runtime_error{std::string{"Network config \""} + network_config_path.str() + "\" does not exist."};
	}

	json result;
	if (equals_case_insensitive(network_config_path.extension(), "json")) {
		std::ifstream f{network_config_path.str()};
		result = json::parse(f, nullptr, true, true);
		result = merge_parent_network_config(result, network_config_path);
	} else if (equals_case_insensitive(network_config_path.extension(), "msgpack")) {
		std::ifstream f{network_config_path.str(), std::ios::in | std::ios::binary};
		result = json::from_msgpack(f);
		// we assume parent pointers are already resolved in snapshots.
	}

	return result;
}

void NerfRender::reload_network_from_file(const std::string& network_config_path) {
    if (!network_config_path.empty()) {
		m_network_config_path = network_config_path;
	}
    m_network_config = load_network_config(m_network_config_path);
    reset_network();
    
    // Haoran 
    // Need to do
    // 1. Load pretrained model
    // 2. Generate Density Grid, which will be used by ray sampler strategy!!!
    // now we just initialize the weight randomly.
    m_nerf_network -> initialize_xavier_uniform(m_rng);
}

void NerfRender::reset_network()
{
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
	uint32_t n_extra_dims = 0;   // Now, it's set to zero but it needs furture check! By Hangkun, 2022/06/30

    // Automatically determine certain parameters if we're dealing with the (hash)grid encoding
	if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
		encoding_config["n_pos_dims"] = n_pos_dims;   // 3 dimenison input

		const uint32_t n_features_per_level = encoding_config.value("n_features_per_level", 2u);
        uint32_t m_num_levels = 16u;

		if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
			m_num_levels = (uint32_t)encoding_config["n_features"] / n_features_per_level;
		} else {
			m_num_levels = encoding_config.value("n_levels", 16u);
		}

		// m_level_stats.resize(m_num_levels);
		// m_first_layer_column_stats.resize(m_num_levels);

		const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

		uint32_t m_base_grid_resolution = encoding_config.value("base_resolution", 0);
		if (!m_base_grid_resolution) {
			m_base_grid_resolution = 1u << ((log2_hashmap_size) / n_pos_dims);
			encoding_config["base_resolution"] = m_base_grid_resolution;
		}

		float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube

		// Automatically determine suitable per_level_scale
		float m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
		if (m_per_level_scale <= 0.0f && m_num_levels > 1) {
			m_per_level_scale = std::exp(std::log(desired_resolution * (float) m_bound / (float) m_base_grid_resolution) / (m_num_levels-1));
			encoding_config["per_level_scale"] = m_per_level_scale;
		}

		tlog::info()
			<< "GridEncoding: "
			<< " Nmin=" << m_base_grid_resolution
			<< " b=" << m_per_level_scale
			<< " F=" << n_features_per_level
			<< " T=2^" << log2_hashmap_size
			<< " L=" << m_num_levels
			;
	}
    
    // Generate the network
    m_nerf_network = std::make_shared<NerfNetwork<precision_t>>(
			n_pos_dims,
			n_dir_dims,
			n_extra_dims,
			n_pos_dims + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
			encoding_config,
			dir_encoding_config,
			network_config,
			rgb_network_config
	);
	
}

void NerfRender::render_frame(struct Camera cam, Eigen::Matrix<float, 4, 4> pos, Eigen::Vector2i resolution)
{
    // cam : parameters of cam
    // pos : camera external parameters
    // resolution : [Width, Height]


    int N = resolution[0] * resolution[1];   // number of pixels
    tcnn::GPUMatrixDynamic<float> rays_o(N,3,tcnn::RM);      // initial points corresponding to pixels, in world coordination
    tcnn::GPUMatrixDynamic<float> rays_d(N,3,tcnn::RM);      // direction corresponding to pixels, in world coordination
    generate_rays(cam, pos, resolution, rays_o, rays_d);

    // Zilong
    // Need to do
    // 1. ray sample. refer the instant-ngp paper for details of the sample strategy. You can do it from the easiest equally spaced sampling strategy to the ray marching strategy.
    // 2. infer and volume rendering.
    // please refer to
    //     1. https://github.com/ashawkey/torch-ngp/blob/main/nerf/renderer.py line 318 ~ 380. espically the functions raymarching.compact_rays, raymarching.march_rays and raymarching.composite_rays. These function are places in https://github.com/ashawkey/torch-ngp/tree/main/raymarching/src.
    // 这里没有指定返回，后续可以再讨论返回图片是以什么格式。先渲染出来再说
	// below is an example of inference!
    int batch_size = 4096;
    tcnn::GPUMatrixDynamic<float> network_input(m_nerf_network -> input_width(), batch_size);
    tcnn::GPUMatrixDynamic<precision_t> network_output(m_nerf_network -> padded_output_width(), batch_size);
    network_input.initialize_constant(0);
    network_output.initialize_constant(0.1);
	tcnn::MatrixView<precision_t> view = network_output.view();
    precision_t host_data[1] = {0};
    cudaMemcpy(host_data, &view(1,1), 1 * sizeof(precision_t), cudaMemcpyDeviceToHost);    // copy one data to host, you can also copy a list of data to host!
    std::cout << "Network Output [1,1] before inference : " << (float) host_data[0] << std::endl;
    m_nerf_network -> inference_mixed_precision(network_input, network_output);
	cudaMemcpy(host_data, &view(1,1), 1 * sizeof(precision_t), cudaMemcpyDeviceToHost);    // copy one data to host, you can also copy a list of data to host!
    std::cout << "Network Output [1,1] after inference : " << (float) host_data[0] << std::endl;
}

void NerfRender::generate_rays(struct Camera cam, Eigen::Matrix<float, 4, 4> pos, Eigen::Vector2i resolution, tcnn::GPUMatrixDynamic<float>& rays_o, tcnn::GPUMatrixDynamic<float>& rays_d) {
    // Weixuan
    // Generate rays according to the input
    // please refer to 
    //     1. https://github.com/ashawkey/torch-ngp/blob/main/nerf/provider.py function nerf_matrix_to_ngp
    //     2. https://github.com/ashawkey/torch-ngp/blob/main/nerf/utils.py function get_rays
    // use cuda to speed up
    
    int N = resolution[0] * resolution[1];   // number of pixels
    std::cout << "N: " << N << std::endl;
    
    Eigen::Matrix<float, 4, 4> new_pose = nerf_matrix_to_ngp(pos);
    
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);

    tcnn::MatrixView<float> rays_o_view = rays_o.view();
    set_rays_o<<<grid_size, block_size>>>(rays_o_view, new_pose.block<1, 3>(0, 3), N);
    
    tcnn::MatrixView<float> rays_d_view = rays_d.view();
    set_rays_d<<<grid_size, block_size>>>(rays_d_view, cam, new_pose.block<3, 3>(0, 0), resolution[0], N);
    
    float host_data[3] = {0, 0, 0};
    
    cudaMemcpy(host_data, &rays_o_view(0,0), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "rays_o[0, :]: " << host_data[0] << ", " << host_data[1] << ", " << host_data[2] << std::endl;
    
    cudaMemcpy(host_data, &rays_d_view(0,0), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "rays_d[0, :]: " << host_data[0] << ", " << host_data[1] << ", " << host_data[2] << std::endl;
}

void NerfRender::generate_density_grid()
{   
    // Jiang Wei
    // once the pretrained model is loaded! we can generate the density grid.
    std::vector<float> tmp_density(m_dg_cascade * m_dg_h * m_dg_h * m_dg_h, 0.1);
    m_density_grid.copy_from_host(tmp_density);
}

NGP_NAMESPACE_END
