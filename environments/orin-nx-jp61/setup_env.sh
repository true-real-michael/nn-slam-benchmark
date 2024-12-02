#!/bin/sh
if [ "$(id -u)" -ne 0 ]; then
  echo "Please run with sudo"
  echo "This is necessary to install the required packages"
  exit 1
fi

# Install required packages
apt-get update
apt-get install -y \
  nvidia-jetpack \
  python3-pip \
  python3-venv \
apt-get install python3-pip -y
pip3 install jetson-stats
systemctl restart jtop.service

# Install cuSPARSELt for torchvision
wget https://developer.download.nvidia.com/compute/cusparselt/0.6.3/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.6.3/cusparselt-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install libcusparselt0 libcusparselt-dev
rm cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb

python3 -m venv .venv
.venv/bin/pip install -r environments/orin-nx-jp61/requirements.txt

# Now you can download the weights and run the benchmark