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
#include <args/args.hxx>
#include <filesystem/path.h>
#include <iostream>
#include <string>

#include <nerf-cuda/nerf_render.h>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

int main(int argc, char** argv) {
	cout << "Hello, Metavese!" << endl;
    NerfRender* render = new NerfRender();
    string config_path = "./configs/nerf/base.json";
    render -> reload_network_from_file(config_path);
}