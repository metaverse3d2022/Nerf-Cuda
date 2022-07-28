if [ -d "build" ]; then
    rm -r build
fi
mkdir build 
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.3/bin/nvcc
make -j16
cd ..
./build/testbed