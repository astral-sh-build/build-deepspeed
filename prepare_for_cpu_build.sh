pip install -U "setuptools<=77.0.1" wheel auditwheel-symbols
pip install lit

python --version
pip --version
which python
which pip

# # Install libaio
# echo "Install libaio 0.3.113..."
# curl https://pagure.io/libaio/archive/libaio-0.3.113/libaio-libaio-0.3.113.tar.gz -o libaio-libaio-0.3.113.tar.gz
# tar -zxvf libaio-libaio-0.3.113.tar.gz
# cd /project/libaio-libaio-0.3.113
# make prefix=/usr install
# cd /project

# # install oneCCL: /project/oneCCL/build/_install
# echo "Install oneCCL"
# cd /project/oneCCL
# mkdir build
# cd build
# cmake ..
# make -j 1 install
# cd /project

# # patch "setup.py" and "deepspeed/env_report.py" to support ops of different accelerators
# echo "Patch setup.py and env_report.py"
# sed -i "s/'{accelerator_name}'/{{'{accelerator_name}', 'cpu'}}/g" setup.py
# sed -i "s/accelerator_name == get_accelerator()._name/get_accelerator()._name in accelerator_name/g" deepspeed/env_report.py
# sed -i "s/accelerator_name == get_accelerator()._name/get_accelerator()._name in accelerator_name/g" op_builder/builder.py
# sed -i "s/accelerator_name == get_accelerator()._name/get_accelerator()._name in accelerator_name/g" op_builder/xpu/builder.py

# # patch libaio
# echo "Patch libaio"
# sed -i "s/'-laio'/'-Wl,-Bstatic', '-laio', '-Wl,-Bdynamic'/g" op_builder/async_io.py
# sed -i "s/'-laio'/'-Wl,-Bstatic', '-laio', '-Wl,-Bdynamic'/g" op_builder/cpu/async_io.py

# # not support xpu and npu now
# # sed -i "s/'-laio'/'-Wl,-Bstatic', '-laio', '-Wl,-Bdynamic'/g" op_builder/npu/async_io.py
# # sed -i "s/'-laio'/'-Wl,-Bstatic', '-laio', '-Wl,-Bdynamic'/g" op_builder/xpu/async_io.py

# # patch cufile use static link ?
# # sed -i "s/'-lcufile'/'-Wl,-Bstatic', '-lcufile_static', '-Wl,-Bdynamic'/g" op_builder/gds.py

# # patch oneCCL use static link
# sed -i "s/'-lccl'/'-Wl,-Bstatic', '-lccl', '-Wl,-Bdynamic'/g" op_builder/cpu/comm.py

# # patch cuda triton
# cp build_scripts/cuda_accelerator.py accelerator/cuda_accelerator.py

# # force compile comm ops
# cp op_builder/cpu/comm.py op_builder/comm.py
# sed -i 's/CPUOpBuilder/TorchCPUOpBuilder/g' op_builder/comm.py

# triton==3.0.0 to support fp_quantizer
# pip install hjson ninja numpy packaging psutil py-cpuinfo pydantic pynvml tqdm libaio deepspeed-kernels "triton>=2.3.0,<=3.0.0"

# echo "install torch==${CI_TORCH_VERSION}+cpu"
# pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
