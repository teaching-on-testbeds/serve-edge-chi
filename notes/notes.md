## Building ONNX Runtime with ACL:

* Start from `ghcr.io/chameleoncloud/edge_ssh_image:latest` container image, add key to allow SSH.
* Get latest `cmake`:

```
wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-aarch64.sh
bash /root/cmake-3.31.6-linux-aarch64.sh
export PATH="/root/cmake-3.31.6-linux-aarch64/bin:$PATH"
```

* Install prerequisites:

```
apt install -y cmake ninja-build git wget unzip \
    build-essential libprotobuf-dev protobuf-compiler \
    autoconf automake libtool
```

```
python3.12 -m pip install numpy
python3.12 -m pip install scons
python3.12 -m pip install packaging
python3.12 -m pip install --upgrade setuptools
```

* Build ACM version 24.07:

```
wget https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.07.tar.gz
tar -zxf v24.07.tar.gz
cd ComputeLibrary-24.07
scons Werror=1 -j$(nproc) debug=0 neon=1 opencl=0 os=linux arch=arm64-v8a
```

* Build ONNX runtime with ACL:

```
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime

python3.12 tools/ci_build/build.py \
    --config Release \
    --build_dir build_arm \
    --parallel \
    --use_acl \
    --acl_home /root/ComputeLibrary-24.07 \
    --acl_libs /root/ComputeLibrary-24.07/build \
    --arm \
    --build_wheel \
    --skip_tests \
    --allow_running_as_root \
    --cmake_extra_defines \
        ACL_INCLUDE_DIR=/root/ComputeLibrary-v24.07/include
```