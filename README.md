## Introduction
This project achieves NeRF rendering based on C++, Cuda and some open source packages like tiny-cuda-nn and so on. Generally, it's faster than NeRF implementation based on python.

## plan
- Static scene rendering based on NeRF model
- Dynamic scene rendering based on some dynamic NeRF model 

## Building the Library

- Dependencies
    - sockpp (https://github.com/fpagliughi/sockpp)

- Download
```shell
$ git clone git@github.com:metaverse3d2022/Nerf-Cuda.git 
$ git submodule update --init --recursive  
```
- Compile
```shell
$ mkdir build
$ cd build
$ cmake ..
$ make -j16
$ # In another way, you can just run "source build.sh". After compiling, you can run ./build/testbed for testing.
```

## Usage
```shell
$ ./build/render_server # open the render server
```

## Troubleshooting compile errors

| Problem | Resolution |
|---------|------------|
| __CMake error:__ No CUDA toolset found / CUDA_ARCHITECTURES is empty for target "cmTC_0c70f" | __Windows:__ the Visual Studio CUDA integration was not installed correctly. Follow [these instructions](https://github.com/mitsuba-renderer/mitsuba2/issues/103#issuecomment-618378963) to fix the problem without re-installing CUDA. ([#18](https://github.com/NVlabs/instant-ngp/issues/18)) |
| | __Linux:__ Environment variables for your CUDA installation are probably incorrectly set. You may work around the issue using ```cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<your cuda version>/bin/nvcc``` ([#28](https://github.com/NVlabs/instant-ngp/issues/28)) |

## Reference
- [instant-ngp](https://github.com/NVlabs/instant-ngp)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [torch-ngp](https://github.com/ashawkey/torch-ngp)
