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
#include <thread>
#include <sockpp/tcp_acceptor.h>
#include <sockpp/version.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

int main(int argc, char** argv) {

  cout << "Hello, Metavese!" << endl;
  NerfRender* render = new NerfRender();
  string config_path = "./freality.msgpack";
  render->reload_network_from_file(config_path);  // Init Model
  double scale_x, scale_y;
  scale_x = 4;
  scale_y = 4;
  // Camera cam = {3550.115 / 4, 3554.515 / 4, 3010.45 / 4, 1996.027 / 4};
  Camera cam = {840, 840, 339, 590};
  Eigen::Matrix<float, 4, 4> pos;
  pos << -0.5575427361517304, -0.11682263918046752, 0.8218871992959822, 3.9673954052389253,
      0.8300327085486383, -0.094966079921629, 0.5495699649760266, 2.667431152445114,
      0.013849191732089516, 0.9886020001326434, 0.14991425965987268, 0.45955395816033995,
      0.0, 0.0, 0.0, 1.0;
  Eigen::Vector2i resolution(1080, 1080);
  assert(resolution[0]*resolution[1]%NGPU==0);
  render -> set_resolution(resolution);

  cout << "Sample TCP echo server for 'sockpp' "
		<< sockpp::SOCKPP_VERSION << '\n' << endl;

	in_port_t port = (argc > 1) ? atoi(argv[1]) : 12345;

	sockpp::socket_initializer sockInit;

	sockpp::tcp_acceptor acc(port);

	if (!acc) {
		cerr << "Error creating the acceptor: " << acc.last_error_str() << endl;
		return 1;
	}
    //cout << "Acceptor bound to address: " << acc.address() << endl;
	cout << "Awaiting connections on port " << port << "..." << endl;

	while (true) {
		sockpp::inet_address peer;

		// Accept a new client connection
		sockpp::tcp_socket sock = acc.accept(&peer);
		cout << "Received a connection request from " << peer << endl;

		if (!sock) {
			cerr << "Error accepting incoming connection: " 
				<< acc.last_error_str() << endl;
		}
		else {
			// Create a thread and transfer the new stream to it.
			//thread thr(run_echo, std::move(sock));
			//thr.detach();
            int n;
            float nerf_pos[16] = {0};
            while ((n = sock.read(nerf_pos, sizeof(nerf_pos))) > 0) {
                Eigen::Matrix<float, 4, 4> nerf_pos_eigen;
                nerf_pos_eigen << nerf_pos[0], nerf_pos[1], nerf_pos[2], nerf_pos[3],
                    nerf_pos[4], nerf_pos[5], nerf_pos[6], nerf_pos[7],
                    nerf_pos[8], nerf_pos[9], nerf_pos[10], nerf_pos[11],
                    nerf_pos[12], nerf_pos[13], nerf_pos[14], nerf_pos[15];
                Image img = render->render_frame(cam, nerf_pos_eigen);
		        sock.write_n(img.rgb, 3 * 1080 * 1080);
            }
	        cout << "Connection closed from " << sock.peer_address() << endl;  
		}
	}

	return 0;
}
