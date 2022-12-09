/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Hangkun
 */

#ifdef _WIN32
  #include <GL/gl3w.h>
#else
  #include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>
#include <nerf-cuda/dlss.h>
#include <nerf-cuda/render_buffer.h>
#include <nerf-cuda/npy.hpp>

#include <cuda.h>
#include <filesystem/path.h>
#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/common.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>
#include <args/args.hxx>
#include <iostream>
#include <string>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

void simple_glfw_error_callback(int error, const char* description) 
{
    std::cout << "GLFW error #" << error << ": " << description << std::endl;
}

__global__ void dlss_prep_kernel(
	Eigen::Vector2i resolution,
	float* depth_buffer,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	const float depth = depth_buffer[idx];
	Eigen::Vector2f mvec = {0., 0.}; // motion vector

	surf2Dwrite(make_float2(mvec.x(), mvec.y()), mvec_surface, x_orig * sizeof(float2), y_orig);

	// Scale depth buffer to be guaranteed in [0,1].
	surf2Dwrite(std::min(std::max(depth / 128.0f, 0.0f), 1.0f), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

void render_frame(ngp::CudaRenderBuffer& render_buffer, unsigned char *depth, unsigned char *rgb, int H, int W) 
{
    std::cout << "render frame begin" << std::endl;
    
    // CUDA stuff
	  tcnn::StreamAndEvent m_stream;
    render_buffer.clear_frame(m_stream.get());
    render_buffer.set_color_space(ngp::EColorSpace::Linear);
	  render_buffer.set_tonemap_curve(ngp::ETonemapCurve::Identity);

    std::cout << "load depth buffer..." << std::endl;
	  std::vector<float> data(H*W);
	  for (int i=0; i<H*W; i++) {
      data[i] = float(depth[i]);
    }
	  render_buffer.host_to_depth_buffer(data);

    // Prepare DLSS data: motion vectors, scaled depth, exposure
    std::cout << "prepare the dlss data..." << std::endl;
    auto res = render_buffer.in_resolution();
    //bool distortion = false;
    const dim3 threads = { 16, 8, 1 };
	  const dim3 blocks = { tcnn::div_round_up((uint32_t)res.x(), threads.x), tcnn::div_round_up((uint32_t)res.y(), threads.y), 1 };
    float m_dlss_sharpening = 0.0;
    dlss_prep_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			res,
			render_buffer.depth_buffer(),
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure()
	);
    render_buffer.set_dlss_sharpening(m_dlss_sharpening);

    std::cout << "run dlss..." << std::endl;
    float m_exposure = 0.0;
    Eigen::Array4f m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};
    render_buffer.accumulate(m_exposure, m_stream.get());

    render_buffer.host_to_accumulate_buffer(rgb, H*W);

    render_buffer.tonemap(m_exposure, m_background_color, ngp::EColorSpace::Linear, m_stream.get());
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

int main(int argc, char** argv) {

  cout << "Hello, Metavese!" << endl;

  std::cout << "custom glfw init" << std::endl;
    glfwSetErrorCallback(simple_glfw_error_callback);
    if (!glfwInit()) {
		throw std::runtime_error{"GLFW could not be initialized."};
	}
    std::cout << "custom enable dlss" << std::endl;
    try {
		ngp::vulkan_and_ngx_init();
	} catch (const std::runtime_error& e) {
		tlog::warning() << "Could not initialize Vulkan and NGX. DLSS not supported. (" << e.what() << ")";
	}

  NerfRender* render = new NerfRender();
  string config_path = "freality.msgpack";
  render->reload_network_from_file(config_path);  // Init Model
  Camera cam = {3550.115/8, 3554.515/8, 3010.45/8, 1996.027/8};
  Eigen::Matrix<float, 4, 4> pos;
  pos << -0.5575427361517304, -0.11682263918046752, 0.8218871992959822, 3.9673954052389253,
      0.8300327085486383, -0.094966079921629, 0.5495699649760266, 2.667431152445114,
      0.013849191732089516, 0.9886020001326434, 0.14991425965987268, 0.45955395816033995,
      0.0, 0.0, 0.0, 1.0;
  Eigen::Vector2i resolution(4000/8, 4000/8);
  assert(resolution[0]*resolution[1]%NGPU==0);
  render -> set_resolution(resolution);
  clock_t start_t, end_t;
  start_t = clock();
  Image img = render->render_frame(cam, pos);
  end_t = clock();
  double total_time = static_cast<double> (end_t - start_t) / 1 / CLOCKS_PER_SEC;
  printf("Process time : %f s / frame", total_time);
  // store images
  char const* deep_file_name = "./deep.png";
  char const* image_file_name = "./image.png";
  stbi_write_png(deep_file_name, img.W, img.H, 1, img.depth, img.W * 1);
  stbi_write_png(image_file_name, img.W, img.H, 3, img.rgb, img.W * 3);

  cudaSetDevice(0); // tmp set 0 for dlss
  int in_height = img.H;
	int in_width = img.W;
  ngp::CudaRenderBuffer m_windowless_render_surface{std::make_shared<ngp::CudaSurface2D>()};
  m_windowless_render_surface.resize({in_width, in_height});
	m_windowless_render_surface.reset_accumulation();

  unsigned long out_height = img.H*2;
	unsigned long out_width = img.W*2;
  // enable dlss
	tlog::info() << "custom enable dlss for render buffer";
	m_windowless_render_surface.enable_dlss({out_width, out_height});
	auto render_res = m_windowless_render_surface.in_resolution();
	if (m_windowless_render_surface.dlss()) {
		render_res = m_windowless_render_surface.dlss()->clamp_resolution(render_res);
	}
	m_windowless_render_surface.resize(render_res);

  render_frame(m_windowless_render_surface, img.depth, img.rgb, in_height, in_width);

  std::cout << "begin to transfer data..." << std::endl;

  //float *result = (float*)malloc(sizeof(float)*out_height*out_width*4);
  std::vector<float> result(out_height*out_width*4, 0.0);

  cudaError_t x = cudaMemcpy2DFromArray(&result[0], out_width * sizeof(float) * 4, m_windowless_render_surface.surface_provider().array(), 0, 0, out_width * sizeof(float) * 4, out_height, cudaMemcpyDeviceToHost);
  CUDA_CHECK_THROW(x);

  unsigned char* rgb_dlss = new unsigned char[out_height*out_width*3];
  for (int i=0; i<out_height*out_width; i++)
      for (int j=0; j<3; j++) {
        rgb_dlss[i*3+j] = (unsigned char)(result[i*4+j]*255);
      }

  char const* dlss_file_name = "./dlss.png";
  stbi_write_png(dlss_file_name, out_width, out_height, 3, rgb_dlss, out_width * 3);

  
/*
  const std::vector<long unsigned> shape{out_height, out_width, 4};
	const bool fortran_order{false};
  const std::string path{"out.npy"};
	
	// try to save frame_buffer here?
	std::cout << "save frame buffer..." << std::endl;
	npy::SaveArrayAsNumpy(path, fortran_order, shape.size(), shape.data(), result);


  // save fig as numpy
  std::vector<unsigned char> image_data(img.rgb, img.rgb + img.W*img.H*3);

  const std::vector<long unsigned> shape_image{(unsigned long)(img.H), (unsigned long)(img.W), 3};
  const std::string path_image{"image.npy"};
	
	std::cout << "save image.npy..." << std::endl;
	npy::SaveArrayAsNumpy(path_image, fortran_order, shape_image.size(), shape_image.data(), image_data);

  std::vector<unsigned char> deep_data(img.depth, img.depth + img.W*img.H);

  const std::vector<long unsigned> shape_deep{(unsigned long)(img.H), (unsigned long)(img.W)};
  const std::string path_deep{"deep.npy"};
	
	std::cout << "save deep.npy..." << std::endl;
	npy::SaveArrayAsNumpy(path_deep, fortran_order, shape_deep.size(), shape_deep.data(), deep_data);

*/
}
