# XHandControl Python SDK Usage (For Python 3.10 and 3.12)

## Official Document for XHand
https://di6kz6gamrw.feishu.cn/drive/folder/WGyhflqb1lRu9ddtc0scjDhwngg

## Install Dependencies
The XHandControl Python SDK is a software development kit for communication with the XHand robotic hand, allowing control of the hand's joints and reading sensor data.

## Installation 
```bash
sudo apt update && sudo apt install -y \
    cmake \
    g++ \
    libcurl4-openssl-dev \
    libssl-dev \
    nlohmann-json3-dev
```

## Run
### Hardware Serial Port Permission Settings
Before using serial communication, you need to set the serial port permissions, otherwise, you may encounter an error when trying to open it:
```bash
# The serial port ID should be modified based on the actual situation
sudo chmod 666 /dev/ttyUSB0
```

### [Recommended] Execute within a conda environment
```bash
# ❗❗Note: Due to EtherCAT communication requirements, the Python interpreter permissions in the conda environment need to be set.
# ❗❗Note: This setting grants the Python process in the current conda environment access to raw network sockets.

# Create a conda environment (optional)
conda create -n xhand python=3.10.12 -y

# Activate the conda environment
conda activate xhand

# Install xhand_controller
# Enter the xhand_control_sdk_py/ directory.
pip install -r requirements.txt
pip install wheels/xhand_controller-*-cp310-cp310-*.whl

# Set permissions for the conda interpreter
sudo setcap cap_net_raw+ep $(readlink -f $(which python3))
```

### Run in the native environment (requires Python 3.10 system installation; Ubuntu 24 LTS ships with Python 3.12 by default)
```bash
# Enter the xhand_control_sdk_py/ directory.
pip install --user -r requirements.txt

# ❗❗Note: Please make sure to check your Python 3 version before installation
python3 --version

# ❗❗Note: Choose one of the following options based on your local Python environment: 
# ❗❗Note: Option 1: If your local Python environment is Python 3.10, use the following command
pip install --user wheels/xhand_controller-*-cp310-cp310-*.whl

# ❗❗Note: Option 2: If your local Python environment is Python 3.12, use the following command
pip install --user wheels/xhand_controller-*-cp312-cp312-*.whl

# Add library path
echo "$(python3 -c 'import os, sys; from importlib import util; spec = util.find_spec("xhand_controller"); print(os.path.dirname(spec.origin) if spec else "")')" | sudo tee /etc/ld.so.conf.d/xhand.conf
sudo ldconfig
```

### Deploy RL policy on xhand hardware
```
# Please remember to change `JIT_POLICY_PATH` and `SCREWDRIVER`
python3 xhand_deploy.py
```

## Fingure Joint Mode
- In the **FingerCommand_t** class, the **mode** parameter is used to control the mode of finger joints.
- The modes of finger joints and the meanings of the parameters are shown in the table below:

| Name   | Value   | Notes        |
|--------|:--------:|-------------|
| Passive Mode  | 0    |    |
| POSIPosition Control ModeTION  | 3    |    |
| Force Control Mode  | 5    |    |

