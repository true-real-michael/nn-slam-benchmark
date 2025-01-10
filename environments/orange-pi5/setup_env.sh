#!/bin/sh
if [ "$(id -u)" -ne 0 ]; then
  echo "Please run with sudo"
  echo "This is necessary to install the required packages"
  exit 1
fi

wget https://github.com/rockchip-linux/rknpu2/raw/refs/heads/master/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so
mv librknnt.so /usr/lib

conda env create -f environments/orange-pi5/environment.yml
conda activate nnsb

git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1 --branch v1.6.0 thirdparty/rknn-toolkit2
pip install -r thirdparty/rknn-toolkit2/rknn_toolkit_lite2/packages/rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
pip uninstall numpy
pip install 'numpy<2.0'

if [ ! -f /usr/lib/librknnrt.so ]; then
  cp thirdparty/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib
fi

if [ ! -f /usr/bin/rknn_server ]; then
  cp thirdparty/rknn-toolkit2/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin
  sudo chmod +x /usr/bin/rknn_server
  sudo chmod +x /usr/bin/start_rknn.sh
  sudo chmod +x /usr/bin/restart_rknn.sh
fi
