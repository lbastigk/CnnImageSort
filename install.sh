#!/bin/bash

# CPU or GPU
type="gpu"

# Check if cpu or gpu is passed as an argument
if [ "$1" == "cpu" ] || [ "$1" == "gpu" ]; then
    type="$1"
fi

if [ "$type" == "cpu" ]; then
    #######################################################
    # Erstelle und aktiviere die virtuelle Umgebung für cpu
    #######################################################
    echo "Installation wird nur für CPU durchgeführt."

    deactivate
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate

    echo ""
    echo "Installation für CPU läuft..."
    echo ""
    pip install --upgrade pip           > /dev/null 2>&1
    pip install -r requirements.txt     > /dev/null 2>&1
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1

    echo "-----------------------------------"
    echo "Installation für CPU abgeschlossen."
    echo ""
    echo "Pakete:"
    echo ""
    pip list

else
    #######################################################
    # Erstelle und aktiviere die virtuelle Umgebung für gpu
    #######################################################
    echo "Installation wird für CPU und GPU durchgeführt."

    deactivate
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate

    echo ""
    echo "Installation für GPU läuft..."
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
fi
echo "-----------------------------------"
echo "Installation für GPU abgeschlossen."
echo ""
echo "Pakete:"
echo ""
pip list