SHELL := /bin/bash
MAKEFLAGS += --silent

ROOT := $(abspath ../../../..)
ANALOG_BUILD := $(ROOT)/build
LLVM_BIN := $(ROOT)/../../../analog/build/llvm-project/bin
TORCH_MLIR_BIN := $(ROOT)/../../../analog/build/torch-mlir/bin

OPT_BIN := $(ANALOG_BUILD)/tools/analog-mlir-opt/analog-mlir-opt
MLIR_TRANSLATE := $(LLVM_BIN)/mlir-translate
TORCH_MLIR_OPT := $(TORCH_MLIR_BIN)/torch-mlir-opt
CLANG := clang
CLANGXX := clang++
RELEASE_OPTFLAGS ?= -O2
DEBUG_OPTFLAGS ?= -O0 -g
CXXFLAGS := -std=c++17 -I$(ROOT) $(RELEASE_OPTFLAGS)
DEBUG_CXXFLAGS := -std=c++17 -I$(ROOT) $(DEBUG_OPTFLAGS) -DDEBUG_MODE
PYTHON_INCLUDE := $(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_LIBDIR := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_CXXFLAGS := $(CXXFLAGS) -I$(PYTHON_INCLUDE)
DEBUG_PYTHON_CXXFLAGS := -std=c++17 -I$(ROOT) $(DEBUG_OPTFLAGS) -I$(PYTHON_INCLUDE) -DDEBUG_MODE
PYTHON_LDFLAGS := $(shell python3 -c "import sysconfig, os, glob; libdir=sysconfig.get_config_var('LIBDIR') or ''; lib=sysconfig.get_config_var('LDLIBRARY') or ''; libpath=os.path.join(libdir, lib); matches=glob.glob(libpath + '*') if lib and not os.path.exists(libpath) else []; lib=os.path.basename(matches[0]) if matches else lib; print(f'-L{libdir} -l:{lib} -ldl -lm')")


NUM_CORES ?= 3
ARRAYS_PER_CORE ?= 1
ARRAY_ROWS ?= 8
ARRAY_COLS ?= 8
TEST_NAME := one_layer_linear

PYTHON_MODEL_SRC := $(ROOT)/tools/debug-simulator/tests/$(TEST_NAME)/$(TEST_NAME).py
OUT_DIR := $(ROOT)/tools/debug-simulator/tests/$(TEST_NAME)/out
INPUT_MLIR := $(OUT_DIR)/$(TEST_NAME).ll

ANALOG_MLIR := $(OUT_DIR)/$(TEST_NAME)_analog.mlir
LOWERED_MLIR := $(OUT_DIR)/$(TEST_NAME)_lowered.mlir
LOWERED_LL := $(OUT_DIR)/$(TEST_NAME)_llvm.ll
OBJECT_FILE := $(OUT_DIR)/$(TEST_NAME).o
PYTHON_SO := $(OUT_DIR)/libpython_$(TEST_NAME).so
BINARY_FILE := $(OUT_DIR)/$(TEST_NAME).exe
MAIN_SRC := $(ROOT)/tools/debug-simulator/tests/$(TEST_NAME)/main.cpp
ARM_ADAPTER_SRC := $(ROOT)/tools/debug-simulator/shims/arm_adapter.cpp
MAIN_OBJECT := $(OUT_DIR)/main.o
ARM_ADAPTER_OBJECT := $(OUT_DIR)/arm_adapter.o
LAYER_DISPATCH_OBJECT := $(OUT_DIR)/layer_dispatch.o
LAYER_DISPATCH_MULTI_OBJECT := $(OUT_DIR)/layer_dispatch_multi_input.o
WEIGHT_DISPATCH_OBJECT := $(OUT_DIR)/weight_dispatch.o
ANALOG_OPERATIONS_OBJECT := $(OUT_DIR)/analog_operations.o
PYTHON_BRIDGE_OBJECT := $(OUT_DIR)/python_bridge.o


.PHONY: run build build-debug analog-transformations mlir-transformations object python-so binary clean

run: build
	NUM_CORES=$(NUM_CORES) ARRAY_ROWS=$(ARRAY_ROWS) ARRAY_COLS=$(ARRAY_COLS) \
	  ARRAYS_PER_CORE=$(ARRAYS_PER_CORE) $(BINARY_FILE)

build:
	@$(MAKE) analog-transformations
	@$(MAKE) mlir-transformations
	@$(MAKE) object
	@$(MAKE) python-so
	@$(MAKE) binary

build-debug:
	$(MAKE) DEBUG_MODE=1 analog-transformations
	$(MAKE) DEBUG_MODE=1 mlir-transformations
	$(MAKE) DEBUG_MODE=1 object
	$(MAKE) DEBUG_MODE=1 python-so
	$(MAKE) DEBUG_MODE=1 binary

analog-transformations: $(ANALOG_MLIR)
	@:

mlir-transformations: $(LOWERED_MLIR)
	@:

object: $(OBJECT_FILE)
	@:

python-so: $(PYTHON_SO)
	@:

binary: $(BINARY_FILE)
	@:

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

$(INPUT_MLIR): $(PYTHON_MODEL_SRC) | $(OUT_DIR)
	cd $(dir $(PYTHON_MODEL_SRC)) && \
	  PYTHONWARNINGS="ignore::FutureWarning" python3 $(notdir $(PYTHON_MODEL_SRC)) | \
	  $(TORCH_MLIR_OPT) --torch-backend-to-linalg-on-tensors-backend-pipeline \
	  > $(INPUT_MLIR)

$(ANALOG_MLIR): $(INPUT_MLIR) | $(OUT_DIR)
	$(OPT_BIN) $(INPUT_MLIR) \
	  --analog-materialize-and-place="array-rows=$(ARRAY_ROWS) array-cols=$(ARRAY_COLS)" \
	  --analog-execute-and-replace \
	  --canonicalize \
	  --cse \
	  --symbol-dce \
	  --analog-dispatch-runtime \
	  --convert-analog-to-golem-backend \
	  --canonicalize \
	  --cse \
	  --symbol-dce \
	  > $(ANALOG_MLIR)

$(LOWERED_MLIR): $(ANALOG_MLIR) | $(OUT_DIR)
	$(OPT_BIN) $(ANALOG_MLIR) \
	  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
	  --convert-bufferization-to-memref \
	  --expand-strided-metadata \
	  --convert-linalg-to-loops \
	  --lower-affine \
	  --convert-scf-to-cf \
	  --convert-math-to-llvm \
	  --convert-arith-to-llvm \
	  --convert-index-to-llvm \
	  --convert-cf-to-llvm \
	  --finalize-memref-to-llvm \
	  --convert-func-to-llvm \
	  --analog-convert-to-debug-shims \
	  --reconcile-unrealized-casts \
	  > $(LOWERED_MLIR)

$(LOWERED_LL): $(LOWERED_MLIR) | $(OUT_DIR)
	$(MLIR_TRANSLATE) --mlir-to-llvmir $(LOWERED_MLIR) -o $(LOWERED_LL)

$(OBJECT_FILE): $(LOWERED_LL) | $(OUT_DIR)
	$(CLANG) $(if $(DEBUG_MODE),$(DEBUG_OPTFLAGS),$(RELEASE_OPTFLAGS)) -Wno-override-module -fPIC -c $(LOWERED_LL) -o $(OBJECT_FILE)

$(MAIN_OBJECT): $(MAIN_SRC) | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $(MAIN_SRC) -o $(MAIN_OBJECT)

$(ARM_ADAPTER_OBJECT): $(ARM_ADAPTER_SRC) | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $(ARM_ADAPTER_SRC) -o $(ARM_ADAPTER_OBJECT)

$(LAYER_DISPATCH_OBJECT): $(ROOT)/tools/debug-simulator/shims/layer_dispatch.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@

$(LAYER_DISPATCH_MULTI_OBJECT): $(ROOT)/tools/debug-simulator/shims/layer_dispatch_multi_input.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@

$(WEIGHT_DISPATCH_OBJECT): $(ROOT)/tools/debug-simulator/shims/weight_dispatch.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@

$(ANALOG_OPERATIONS_OBJECT): $(ROOT)/tools/debug-simulator/shims/analog_operations.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@


$(PYTHON_BRIDGE_OBJECT): $(ROOT)/tools/debug-simulator/shims/python_bridge.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_PYTHON_CXXFLAGS),$(PYTHON_CXXFLAGS)) \
	  -DNUM_CORES=$(NUM_CORES) \
	  -c $< -o $@

$(PYTHON_SO): $(OBJECT_FILE) | $(OUT_DIR)
	$(CLANGXX) -shared -Wl,--allow-shlib-undefined $(OBJECT_FILE) -o $(PYTHON_SO)

$(BINARY_FILE): $(OBJECT_FILE) $(MAIN_OBJECT) $(ARM_ADAPTER_OBJECT) $(LAYER_DISPATCH_OBJECT) $(LAYER_DISPATCH_MULTI_OBJECT) $(WEIGHT_DISPATCH_OBJECT) $(ANALOG_OPERATIONS_OBJECT) $(PYTHON_BRIDGE_OBJECT) $(PYTHON_SO) | $(OUT_DIR)
	$(CLANGXX) -rdynamic \
	  $(OBJECT_FILE) \
	  $(MAIN_OBJECT) \
	  $(ARM_ADAPTER_OBJECT) \
	  $(LAYER_DISPATCH_OBJECT) \
	  $(LAYER_DISPATCH_MULTI_OBJECT) \
	  $(WEIGHT_DISPATCH_OBJECT) \
	  $(ANALOG_OPERATIONS_OBJECT) \
	  $(PYTHON_BRIDGE_OBJECT) \
	  $(PYTHON_LDFLAGS) \
	  -o $(BINARY_FILE)


clean:
	rm -rf $(OUT_DIR)
	rm -f $(INPUT_MLIR)
