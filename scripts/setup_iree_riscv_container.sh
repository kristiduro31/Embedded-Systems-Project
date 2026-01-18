#!/bin/bash
# setup_iree_riscv_container.sh
# Main orchestrator to build and run the IREE RISC-V environment

set -e # Exit on error

WORKSPACE_DIR=~/polimi/hw5
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

echo "--- Launching Automated IREE RISC-V Container ---"
# Launches container and immediately runs the internal reproduction script
docker run --platform linux/amd64 -it \
  -v "$(pwd)":/work -w /work \
  ubuntu:22.04 \
  /bin/bash -c "chmod +x run_simulations_and_reproduce_benchmark_suite.sh && ./run_simulations_and_reproduce_benchmark_suite.sh"

echo "--- Simulation Run Complete. Check $WORKSPACE_DIR for results ---"