#!/bin/bash
# setup_iree_riscv_container.sh

set -e 

# Directory where scripts actually live
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR=~/polimi/hw_5

mkdir -p "$WORKSPACE_DIR"

echo "--- Launching Automated IREE RISC-V Container ---"

# We mount the SCRIPT_DIR so the files are actually visible inside /work
docker run --platform linux/amd64 -it \
  -v "$SCRIPT_DIR":/work \
  -v "$WORKSPACE_DIR":/output \
  -w /work \
  ubuntu:22.04 \
  /bin/bash -c "chmod +x run_simulations_and_reproduce_benchmark_suite.sh && ./run_simulations_and_reproduce_benchmark_suite.sh"

echo "--- Simulation Run Complete ---"
