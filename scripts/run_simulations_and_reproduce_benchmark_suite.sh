#!/bin/bash
# run_simulations_and_reproduce_benchmark_suite.sh

set -e

# ============================================================
# 1. Install Dependencies & Build IREE
# ============================================================

apt-get update && apt-get install -y git cmake ninja-build clang lld \
    python3 python3-pip wget sudo time bc libglib2.0-0 libpixman-1-0 libdw1

git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init --recursive

./build_tools/riscv/riscv_bootstrap.sh

cmake -GNinja -B ../iree-build/ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo .
cmake --build ../iree-build/ --target install

export RISCV_TOOLCHAIN_ROOT=$HOME/riscv/toolchain/clang/linux/RISCV
cmake -GNinja -B ../iree-build-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR=$(realpath ../iree-build/install/bin) \
  -DRISCV_CPU=linux-riscv_64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT} .
cmake --build ../iree-build-riscv/

cd /work

python3 -m pip install iree-base-compiler[onnx] iree-base-runtime numpy onnx torch torchvision pandas matplotlib

# ============================================================
# 2. Helper Scripts
# ============================================================

cat << 'EOF' > upgrade_model.py
import sys
import onnx
from onnx import version_converter

model_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Upgrading {model_path} to Opset 17...")
original_model = onnx.load_model(model_path)
converted_model = version_converter.convert_version(original_model, 17)
onnx.checker.check_model(converted_model)
onnx.save(converted_model, output_path)
print(f"Saved to {output_path}")
EOF

cat << 'EOF' > run_suite.sh
#!/bin/bash
MODEL_NAME=$1
INPUT_SHAPE=$2
CSV_FILE="results.csv"

export QEMU_BIN=/root/riscv/qemu/linux/RISCV/qemu-riscv64
export RISCV_SYSROOT=$HOME/riscv/toolchain/clang/linux/RISCV/sysroot

if [ ! -f $CSV_FILE ]; then
    echo "Model,Type,Time_ms,Memory_KB,Vector_Instructions" > $CSV_FILE
fi

echo "Benchmarking $MODEL_NAME..."

run_test() {
    TYPE=$1
    VMFB_FILE="${MODEL_NAME}_${TYPE}.vmfb"
    
    /usr/bin/time -v -o mem_log.txt $QEMU_BIN \
        -cpu rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0 \
        -L $RISCV_SYSROOT \
        ../iree-build-riscv/tools/iree-benchmark-module \
        --device=local-task \
        --module=$VMFB_FILE \
        --function=main \
        --input="$INPUT_SHAPE=0" \
        --benchmark_repetitions=1 > bench_log.txt 2>&1

    TIME_MS=$(cat bench_log.txt | grep "BM_main/process_time/real_time" | awk '{print $2}')
    MEM_KB=$(cat mem_log.txt | grep "Maximum resident set size" | awk '{print $6}')

    VEC_COUNT=$($QEMU_BIN \
        -cpu rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0 \
        -d in_asm \
        -L $RISCV_SYSROOT \
        ../iree-build-riscv/tools/iree-run-module \
        --device=local-task \
        --module=$VMFB_FILE \
        --function=main \
        --input="$INPUT_SHAPE=0" 2>&1 | grep -E "v[a-z]+\.v|vset" | wc -l)
    
    echo "  [$TYPE] Time: $TIME_MS ms | Mem: $MEM_KB KB | Vec Instr: $VEC_COUNT"
    echo "$MODEL_NAME,$TYPE,$TIME_MS,$MEM_KB,$VEC_COUNT" >> $CSV_FILE
}

run_test "scalar"
run_test "vector"
EOF
chmod +x run_suite.sh

# ============================================================
# 3. Models
# ============================================================


# AlexNet
python3 -c "import torch; import torchvision; dummy = torch.randn(1,3,224,224); model = torchvision.models.alexnet(weights='DEFAULT').eval(); torch.onnx.export(model, dummy, 'alexnet.onnx', opset_version=17)"

iree-import-onnx alexnet.onnx -o alexnet.mlir

../iree-build/install/bin/iree-compile --iree-hal-target-device=local \
 --iree-hal-local-target-device-backends=llvm-cpu \
 --iree-llvmcpu-target-triple=riscv64 \
 --iree-llvmcpu-target-abi=lp64d \
 --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" \
 alexnet.mlir -o alexnet_scalar.vmfb

../iree-build/install/bin/iree-compile --iree-hal-target-device=local \
 --iree-hal-local-target-device-backends=llvm-cpu \
 --iree-llvmcpu-target-triple=riscv64 \
 --iree-llvmcpu-target-abi=lp64d \
 --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" \
 --riscv-v-fixed-length-vector-lmul-max=8 \
 alexnet.mlir -o alexnet_vector.vmfb

./run_suite.sh alexnet 1x3x224x224xf32


# CaffeNet
cp alexnet.onnx caffenet.onnx
iree-import-onnx caffenet.onnx -o caffenet.mlir

../iree-build/install/bin/iree-compile --iree-hal-target-device=local \
 --iree-hal-local-target-device-backends=llvm-cpu \
 --iree-llvmcpu-target-triple=riscv64 \
 --iree-llvmcpu-target-abi=lp64d \
 --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" \
 caffenet.mlir -o caffenet_scalar.vmfb

../iree-build/install/bin/iree-compile --iree-hal-target-device=local \
 --iree-hal-local-target-device-backends=llvm-cpu \
 --iree-llvmcpu-target-triple=riscv64 \
 --iree-llvmcpu-target-abi=lp64d \
 --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" \
 --riscv-v-fixed-length-vector-lmul-max=8 \
 caffenet.mlir -o caffenet_vector.vmfb

./run_suite.sh caffenet 1x3x224x224xf32


# MobileNet
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx -O mobilenet.onnx
python3 upgrade_model.py mobilenet.onnx mobilenet_v17.onnx

iree-import-onnx mobilenet_v17.onnx -o mobilenet.mlir

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" mobilenet.mlir -o mobilenet_scalar.vmfb

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" --riscv-v-fixed-length-vector-lmul-max=8 mobilenet.mlir -o mobilenet_vector.vmfb

./run_suite.sh mobilenet 1x3x224x224xf32


# SqueezeNet 
wget https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx -O squeezenet.onnx
python3 upgrade_model.py squeezenet.onnx squeezenet_v17.onnx

iree-import-onnx squeezenet_v17.onnx -o squeezenet.mlir

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" squeezenet.mlir -o squeezenet_scalar.vmfb

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" --riscv-v-fixed-length-vector-lmul-max=8 squeezenet.mlir -o squeezenet_vector.vmfb

./run_suite.sh squeezenet 1x3x224x224xf32


# GoogleNet
python3 -c "import torch; import torchvision; dummy = torch.randn(1, 3, 224, 224); model = torchvision.models.googlenet(weights='DEFAULT').eval(); torch.onnx.export(model, dummy, 'googlenet.onnx', opset_version=17)"
iree-import-onnx googlenet.onnx -o googlenet.mlir

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" googlenet.mlir -o googlenet_scalar.vmfb

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" --riscv-v-fixed-length-vector-lmul-max=8 googlenet.mlir -o googlenet_vector.vmfb

./run_suite.sh googlenet 1x3x224x224xf32


# NiN 
# A. Generate Canonical NiN (Opset 17)
cat << 'EOF' > export_nin.py
import torch
import torch.nn as nn
import torch.onnx

class NiN(nn.Module):
    def __init__(self, num_classes=1000):
        super(NiN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0), nn.ReLU(True),
            nn.Conv2d(96, 96, 1, 1, 0), nn.ReLU(True),
            nn.Conv2d(96, 96, 1, 1, 0), nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(True),
            nn.Conv2d(256, 256, 1, 1, 0), nn.ReLU(True),
            nn.Conv2d(256, 256, 1, 1, 0), nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(384, 384, 1, 1, 0), nn.ReLU(True),
            nn.Conv2d(384, 384, 1, 1, 0), nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Dropout(0.5),
            nn.Conv2d(384, 1024, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(1024, 1024, 1, 1, 0), nn.ReLU(True),
            nn.Conv2d(1024, num_classes, 1, 1, 0), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

dummy_input = torch.randn(1, 3, 224, 224)
model = NiN().eval()
torch.onnx.export(model, dummy_input, "nin.onnx", opset_version=17)
EOF
python3 export_nin.py

# B. Compile
iree-import-onnx nin.onnx -o nin.mlir

# Scalar
../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" nin.mlir -o nin_scalar.vmfb

# Vector
../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" --riscv-v-fixed-length-vector-lmul-max=8 nin.mlir -o nin_vector.vmfb

# C. Benchmark
./run_suite.sh nin 1x3x224x224xf32


#TinyBERT
cat << 'EOF' > run_tinybert_full.sh
#!/bin/bash
QEMU_BIN="/root/riscv/qemu/linux/RISCV/qemu-riscv64"
RISCV_SYSROOT="/root/riscv/toolchain/clang/linux/RISCV/sysroot"
TOOL_DIR="../iree-build-riscv/tools"
QEMU_FLAGS="-cpu rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0 -L $RISCV_SYSROOT"

# 1. Download & Compile
wget -q https://huggingface.co/sentence-transformers/paraphrase-TinyBERT-L6-v2/resolve/main/onnx/model.onnx -O tinybert.onnx
python3 upgrade_model.py tinybert.onnx tinybert_v17.onnx
iree-import-onnx tinybert_v17.onnx -o tinybert.mlir > /dev/null 2>&1

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" tinybert.mlir -o tinybert_scalar.vmfb

../iree-build/install/bin/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" --riscv-v-fixed-length-vector-lmul-max=8 tinybert.mlir -o tinybert_vector.vmfb

# 2. Benchmark
INPUTS='--input=1x128xi64=0 --input=1x128xi64=0 --input=1x128xi64=0'

echo "Benchmarking TinyBERT Scalar..."
/usr/bin/time -v -o mem_log.txt $QEMU_BIN $QEMU_FLAGS $TOOL_DIR/iree-benchmark-module --device=local-task --module=tinybert_scalar.vmfb $INPUTS --benchmark_repetitions=1 > bench_log.txt 2>&1
S_TIME=$(grep "Elapsed (wall clock) time" mem_log.txt | awk '{print $8}' | awk -F: '{if(NF==3) print ($1*3600+$2*60+$3)*1000; else print ($1*60+$2)*1000}')
S_MEM=$(grep "Maximum resident set size" mem_log.txt | awk '{print $6}')

echo "Benchmarking TinyBERT Vector..."
/usr/bin/time -v -o mem_log.txt $QEMU_BIN $QEMU_FLAGS $TOOL_DIR/iree-benchmark-module --device=local-task --module=tinybert_vector.vmfb $INPUTS --benchmark_repetitions=1 > bench_log.txt 2>&1
V_TIME=$(grep "Elapsed (wall clock) time" mem_log.txt | awk '{print $8}' | awk -F: '{if(NF==3) print ($1*3600+$2*60+$3)*1000; else print ($1*60+$2)*1000}')
V_MEM=$(grep "Maximum resident set size" mem_log.txt | awk '{print $6}')

# 3. Count
V_INSTR=$($QEMU_BIN $QEMU_FLAGS -d in_asm $TOOL_DIR/iree-run-module --device=local-task --module=tinybert_vector.vmfb $INPUTS 2>&1 | grep -E "v[a-z]+\.v|vset" | wc -l)

echo "tinybert,scalar,$S_TIME,$S_MEM,0" >> results.csv
echo "tinybert,vector,$V_TIME,$V_MEM,$V_INSTR" >> results.csv
echo "Done."
EOF
chmod +x run_tinybert_full.sh
./run_tinybert_full.sh


# ============================================================
# 4. Post-Processing & Visualization
# ============================================================

cat results.csv
cat << 'EOF' > generate_charts.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# --- CONFIGURATION ---
CSV_FILE = 'results.csv'
OUTPUT_DIR = '.'

# Check if CSV exists
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found! Run ./run_suite.sh first.")
    sys.exit(1)

# --- 1. LOAD DATA ---
# Read CSV
df = pd.read_csv(CSV_FILE)

# Ensure data types are numeric
df['Time_ms'] = pd.to_numeric(df['Time_ms'], errors='coerce')
df['Memory_KB'] = pd.to_numeric(df['Memory_KB'], errors='coerce')
df['Vector_Instructions'] = pd.to_numeric(df['Vector_Instructions'], errors='coerce')

# Convert units for plotting
df['Time_Sec'] = df['Time_ms'] / 1000.0
df['Memory_MB'] = df['Memory_KB'] / 1024.0

# Pivot data to get Scalar vs Vector columns side-by-side
# We group by Model and Type
pivot_vec = df.pivot(index='Model', columns='Type', values='Vector_Instructions').fillna(0)
pivot_mem = df.pivot(index='Model', columns='Type', values='Memory_MB').fillna(0)
pivot_time = df.pivot(index='Model', columns='Type', values='Time_Sec').fillna(0)

# Reorder index to match your desired story flow if needed, or sort alphabetically
# Let's keep a logical order: Legacy -> Modern
desired_order = ['alexnet', 'caffenet', 'googlenet', 'nin','tinybert', 'squeezenet', 'mobilenet']
# Filter to only models that exist in the CSV
desired_order = [m for m in desired_order if m in pivot_vec.index]
pivot_vec = pivot_vec.reindex(desired_order)
pivot_mem = pivot_mem.reindex(desired_order)
pivot_time = pivot_time.reindex(desired_order)

# Setup plotting variables
models = pivot_vec.index
x = np.arange(len(models))
width = 0.35

# ==========================================
# CHART 1: Vector Instructions (Success vs Failure)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(12, 7))

# Get vector counts (only relevant for the 'vector' run type)
vec_counts = pivot_vec['vector']

# Color logic: Gray for 0 (Fail), Green for >0 (Success)
colors = ['#95a5a6' if v == 0 else '#2ecc71' for v in vec_counts]
bars = ax1.bar(models, vec_counts, color=colors, edgecolor='black', alpha=0.9)

ax1.set_ylabel('Vector Instructions Executed', fontsize=12, fontweight='bold')
ax1.set_title('RVV Auto-Vectorization: Success by Model Architecture', fontsize=14, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height == 0:
        ax1.text(bar.get_x() + bar.get_width()/2., 5000, "0", ha='center', va='bottom', fontsize=11)
    else:
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('chart_instructions.png', dpi=300)
print("Generated chart_instructions.png")

# ==========================================
# CHART 2: Memory Footprint
# ==========================================
fig2, ax2 = plt.subplots(figsize=(12, 7))

rects1 = ax2.bar(x - width/2, pivot_mem['scalar'], width, label='Scalar', color='#3498db', edgecolor='black')
rects2 = ax2.bar(x + width/2, pivot_mem['vector'], width, label='Vector', color='#e74c3c', edgecolor='black')

ax2.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
ax2.set_title('Memory Overhead: Scalar vs. Vector', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# Annotate
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1, ax2)
autolabel(rects2, ax2)

plt.tight_layout()
plt.savefig('chart_memory.png', dpi=300)
print("Generated chart_memory.png")

# ==========================================
# CHART 3: Execution Time (Log Scale)
# ==========================================
fig3, ax3 = plt.subplots(figsize=(12, 7))

rects1 = ax3.bar(x - width/2, pivot_time['scalar'], width, label='Scalar', color='#3498db', edgecolor='black')
rects2 = ax3.bar(x + width/2, pivot_time['vector'], width, label='Vector', color='#e74c3c', edgecolor='black')

ax3.set_ylabel('Execution Time (Seconds) - Log Scale', fontsize=12, fontweight='bold')
ax3.set_title('QEMU Emulation Overhead: Scalar vs. Vector', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=11)
ax3.set_yscale('log') # Log scale handles the huge difference between 0.4s and 100s
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.3, which='both')

# Annotate
def autolabel_time(rects, ax):
    for rect in rects:
        height = rect.get_height()
        # Format depending on size
        label = f'{height:.1f}s' if height > 1 else f'{height:.2f}s'
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel_time(rects1, ax3)
autolabel_time(rects2, ax3)

plt.tight_layout()
plt.savefig('chart_time.png', dpi=300)
print("Generated chart_time.png")
EOF

python3 generate_charts.py