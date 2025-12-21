#!/bin/bash
# Test script for Lunar Lander with multiprocessing
# Runs the example and monitors CPU usage to verify true parallelism

echo "=== Building release version ==="
cargo build --release --example openai_lunar_lander --features openai

echo ""
echo "=== Starting Lunar Lander with multiprocessing ==="
echo "Press Ctrl+C to stop"
echo ""

# Run in background and capture PID
timeout 60 ./target/release/examples/openai_lunar_lander 2>&1 &
MAIN_PID=$!
echo "Main process PID: $MAIN_PID"

# Wait a bit for Python processes to spawn
sleep 5

echo ""
echo "=== CPU Usage Monitor ==="
echo "Expected: Multiple Python processes at ~100% CPU each (true parallelism)"
echo ""
ps aux | head -1
ps aux | grep -E "(python|lunar)" | grep -v grep | head -20

echo ""
echo "=== Process Count ==="
NUM_PYTHON=$(ps aux | grep python | grep -v grep | wc -l)
NUM_CPUS=$(nproc)
echo "Python worker processes: $NUM_PYTHON"
echo "Available CPUs: $NUM_CPUS"

if [ "$NUM_PYTHON" -gt 1 ]; then
    echo "✓ TRUE PARALLELISM DETECTED: Multiple Python processes running"
else
    echo "✗ Warning: Only 1 Python process detected"
fi

# Wait for process to finish or timeout
wait $MAIN_PID 2>/dev/null
echo ""
echo "=== Test completed ==="
