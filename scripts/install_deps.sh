#!/bin/bash
# Install dependencies for Lunar Lander with PyO3

echo "=== Installing system dependencies for box2d ==="
sudo apt-get update
sudo apt-get install -y swig build-essential python3-dev

echo ""
echo "=== Installing Python packages ==="
pip install gymnasium[box2d]

echo ""
echo "=== Installation complete! ==="
echo "You can now run: ./scripts/test_parallel.sh"
