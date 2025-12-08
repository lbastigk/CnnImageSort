#!/bin/bash

# Make sure the script is not run as root
if [ "$EUID" -eq 0 ]; then
    echo "Sudo user detected. Please do not run this script as root."
    exit 1
fi

# CPU or GPU
type="gpu"

# Check if cpu or gpu is passed as an argument
if [ "$1" == "cpu" ] || [ "$1" == "gpu" ]; then
    type="$1"
fi

if [ "$type" == "cpu" ]; then
    #######################################################
    # Create and activate the virtual environment for cpu
    #######################################################
    echo "Installing for CPU only."

    deactivate
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate

    echo ""
    echo "Installation for CPU is running..."
    echo ""
    pip install --upgrade pip           > /dev/null 2>&1
    pip install -r requirements.txt     > /dev/null 2>&1
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1

    echo "-----------------------------------"
    echo "Installation for CPU completed."
    echo ""
    echo "Packages:"
    echo ""
    pip list

else
    #######################################################
    # Create and activate the virtual environment for gpu
    #######################################################
    echo "Installing for CPU and GPU."

    deactivate
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate

    echo ""
    echo "Installation for GPU is running..."
    echo ""
    pip install --upgrade pip           > /dev/null 2>&1
    pip install -r requirements.txt     > /dev/null 2>&1

    # Getting installed gpu type
    if [[ $(lspci | grep -i nvidia) ]]; then
        echo "NVIDIA GPU detected."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 > /dev/null 2>&1
    elif [[ $(lspci | grep -i amd) ]]; then
        echo "AMD GPU detected."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4 > /dev/null 2>&1
    else
        echo "No supported GPU detected. Defaulting to CPU."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
    fi

    echo "-----------------------------------"
    echo "Installation for GPU completed."
    echo ""
    echo "Packages:"
    echo ""
    pip list
fi