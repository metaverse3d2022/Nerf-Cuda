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

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <args/args.hxx>
#include <filesystem/path.h>
#include <iostream>
#include <string>
#include <nerf-cuda/nerf_render.h>
#include <cuda.h>
#include <Eigen/Dense>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

__global__ void add_one(float* data, const int N = 5) {
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index > N) {
		return ;
	}
	data[index] += 1;
}

int main(int argc, char** argv) {
    cout << "Hello, Metavese!" << endl;
    NerfRender* render = new NerfRender();
    string config_path = "./configs/nerf/base.json";
    render -> reload_network_from_file(config_path);   // Init Model
    render -> generate_density_grid(); // generate densiy grid if we do not load it.
    Camera cam={1920, 1920, 0, 0};
    Eigen::Matrix<float, 4, 4> pos;
    pos << 1, 0 , 0, 0, 
	   0, 1 , 0, 0,
	   0, 0 , 1, 0,
	   0, 0 , 0, 1;
    Eigen::Vector2i resolution (1920, 1080);
    render -> render_frame(cam, pos, resolution);
     
    // GPU memory reading and write
    tcnn::GPUMemory<float> mom(5); // length is equal to 5.
    std::vector<float> mom_h; // length is equal to 5.
    mom_h.resize(5,0);
    mom.copy_from_host(mom_h);
    cout << "Second Value of GPUMemory<float> : " << mom_h[1] << endl;
    add_one<<<1,5>>> (mom.data(), 5); // error operation : mom.data()[1] += 1; device variable should be operated in __device__ function !!!
    mom.copy_to_host(mom_h);
    cout << "Second Value of GPUMemory<float> (changed!) : " << mom_h[1] << endl;
}
