ROOT_DIR := $(abspath $(CURDIR)/../../../..)
BUILD_DIR := $(CURDIR)/build

OPT_BIN := $(ROOT_DIR)/build/tools/analog-mlir-opt/analog-mlir-opt
RUNTIME_LIB := $(ROOT_DIR)/build/runtime/libMLIRAnalogRuntime.a
TORCH_MLIR ?= $(HOME)/Develop/analog/build/torch-mlir
TORCH_MLIR_OPT := $(TORCH_MLIR)/bin/torch-mlir-opt
LLVM_BIN_DIR ?= /root/Develop/analog/build/llvm-project/bin
MLIR_TRANSLATE ?= $(LLVM_BIN_DIR)/mlir-translate
CXX := clang++
PYTHON_LINK_LIB := $(shell python3 -c "import os, sysconfig; libname = sysconfig.get_config_var('LDLIBRARY') or ''; candidates = [os.path.join(sysconfig.get_config_var('LIBPL') or '', libname), os.path.join(sysconfig.get_config_var('LIBDIR') or '', libname)]; print(next((path for path in candidates if path and os.path.exists(path)), ''))")
PYTHON_LIBS := $(shell python3 -c "import sysconfig; print(' '.join(filter(None, [(sysconfig.get_config_var('LIBS') or '').strip(), (sysconfig.get_config_var('SYSLIBS') or '').strip()])))")
PYTHON_LDFLAGS := $(PYTHON_LINK_LIB) $(PYTHON_LIBS)

MODEL_PY := $(CURDIR)/conv3d_bias.py
GENERATED_SRC_LL := $(BUILD_DIR)/conv3d_bias.ll
MAIN_SRC := $(CURDIR)/main.cpp
GENERATED_MLIR := $(BUILD_DIR)/conv3d_bias.runtime.mlir
GENERATED_LL := $(BUILD_DIR)/conv3d_bias.runtime.ll
GENERATED_OBJ := $(BUILD_DIR)/conv3d_bias.runtime.o
RUNNER := $(BUILD_DIR)/conv3d_bias

ARRAY_ROWS ?= 8
ARRAY_COLS ?= 8
ARRAYS_PER_CORE ?= 1
NUM_CORES ?= 1
ANALOG_DEBUG_SIM_DEBUG ?= 1

GOLEM_DEBUG_PASSES := \
	--analog-extract-layers \
	--analog-convert-layers="array-rows=${ARRAY_ROWS} array-cols=${ARRAY_COLS}" \
	--analog-isolate-weights \
	--analog-assemble-task-graph \
	--analog-lower-to-golem

OPT_PASSES := \
	$(GOLEM_DEBUG_PASSES) \
	--empty-tensor-to-alloc-tensor \
	--one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
	--convert-bufferization-to-memref \
	--convert-linalg-to-loops \
	--lower-affine \
	--convert-scf-to-cf \
	--convert-math-to-llvm \
	--expand-strided-metadata \
	--lower-affine \
	--convert-arith-to-llvm \
	--convert-index-to-llvm \
	--convert-cf-to-llvm \
	--finalize-memref-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts \
	--analog-emit-runtime-graph

.PHONY: all build run debug clean

all: build

build: $(RUNNER)

$(OPT_BIN) $(RUNTIME_LIB) $(TORCH_MLIR_OPT):
	@cmake --build $(ROOT_DIR)/build --target analog-mlir-opt MLIRAnalogRuntime -j2

$(BUILD_DIR):
	@mkdir -p $@

$(GENERATED_SRC_LL): $(MODEL_PY) $(TORCH_MLIR_OPT) | $(BUILD_DIR)
	@PYTHONWARNINGS="ignore::FutureWarning" python3 $< --mode mlir | \
		$(TORCH_MLIR_OPT) --torch-backend-to-linalg-on-tensors-backend-pipeline > $@

$(GENERATED_MLIR): $(GENERATED_SRC_LL) $(OPT_BIN) | $(BUILD_DIR)
	@$(OPT_BIN) $(OPT_PASSES) $< > $@

$(GENERATED_LL): $(GENERATED_MLIR)
	@$(MLIR_TRANSLATE) --mlir-to-llvmir $< > $@

$(GENERATED_OBJ): $(GENERATED_LL)
	@$(CXX) -Wno-override-module -c -x ir $< -o $@

$(RUNNER): $(MAIN_SRC) $(GENERATED_OBJ) $(RUNTIME_LIB)
	@$(CXX) -std=c++20 \
		-I$(ROOT_DIR)/tests/runtime/golem-simulator/lib \
		$< $(GENERATED_OBJ) $(RUNTIME_LIB) $(PYTHON_LDFLAGS) \
		-o $@

run: $(RUNNER)
	@ANALOG_DEBUG_SIM_DEBUG=$(ANALOG_DEBUG_SIM_DEBUG) NUM_CORES=$(NUM_CORES) ARRAY_ROWS=$(ARRAY_ROWS) ARRAY_COLS=$(ARRAY_COLS) ARRAYS_PER_CORE=$(ARRAYS_PER_CORE) $(RUNNER)

debug: $(GENERATED_SRC_LL) $(OPT_BIN)
	@$(OPT_BIN) $(GOLEM_DEBUG_PASSES) $(GENERATED_SRC_LL)

clean:
	@rm -rf $(BUILD_DIR)
