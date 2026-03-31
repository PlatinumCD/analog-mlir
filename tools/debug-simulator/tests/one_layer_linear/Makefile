SHELL := /bin/bash
MAKEFLAGS += --silent

ROOT := $(abspath ../../../..)
ANALOG_BUILD := $(ROOT)/build
LLVM_BIN := $(ROOT)/../../../analog/build/llvm-project/bin

OPT_BIN := $(ANALOG_BUILD)/tools/analog-mlir-opt/analog-mlir-opt
MLIR_TRANSLATE := $(LLVM_BIN)/mlir-translate
CLANG := clang
CLANGXX := clang++
CXXFLAGS := -std=c++17 -I$(ROOT)
DEBUG_CXXFLAGS := $(CXXFLAGS) -DDEBUG_MODE
PYTHON_INCLUDE := $(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_LIBDIR := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_LIBNAME := $(shell python3 -c "print('python{}.{}'.format(*__import__('sys').version_info[:2]))")
PYTHON_CXXFLAGS := $(CXXFLAGS) -I$(PYTHON_INCLUDE)
DEBUG_PYTHON_CXXFLAGS := $(PYTHON_CXXFLAGS) -DDEBUG_MODE
PYTHON_LDFLAGS := -L$(PYTHON_LIBDIR) -l$(PYTHON_LIBNAME) -ldl -lm


ARRAY_ROWS ?= 8
ARRAY_COLS ?= 8
TEST_NAME := one_layer_linear

INPUT_MLIR := $(ROOT)/tools/debug-simulator/tests/$(TEST_NAME)/$(TEST_NAME).ll
OUT_DIR := $(ROOT)/tools/debug-simulator/tests/$(TEST_NAME)/out

ANALOG_MLIR := $(OUT_DIR)/$(TEST_NAME)_analog.mlir
LOWERED_MLIR := $(OUT_DIR)/$(TEST_NAME)_lowered.mlir
LOWERED_LL := $(OUT_DIR)/$(TEST_NAME).ll
OBJECT_FILE := $(OUT_DIR)/$(TEST_NAME).o
PYTHON_SO := $(OUT_DIR)/libpython_$(TEST_NAME).so
BINARY_FILE := $(OUT_DIR)/$(TEST_NAME).exe
MAIN_SRC := $(ROOT)/tools/debug-simulator/tests/$(TEST_NAME)/main.cpp
MAIN_OBJECT := $(OUT_DIR)/main.o
LAYER_DISPATCH_OBJECT := $(OUT_DIR)/layer_dispatch.o
LAYER_DISPATCH_MULTI_OBJECT := $(OUT_DIR)/layer_dispatch_multi_input.o
WEIGHT_DISPATCH_OBJECT := $(OUT_DIR)/weight_dispatch.o
ANALOG_OPERATIONS_OBJECT := $(OUT_DIR)/analog_operations.o
+PYTHON_BRIDGE_OBJECT := $(OUT_DIR)/python_bridge.o


.PHONY: build build-debug analog-transformations mlir-transformations object python-so binary clean

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
	@echo "Analog transformations written to $(ANALOG_MLIR)"

mlir-transformations: $(LOWERED_MLIR)
	@echo "MLIR transformations written to $(LOWERED_MLIR)"

object: $(OBJECT_FILE)
	@echo "Object artifact written to $(OBJECT_FILE)"

python-so: $(PYTHON_SO)
	@echo "Python artifact written to $(PYTHON_SO)"

binary: $(BINARY_FILE)
	@echo "Binary artifact is not implemented yet."

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

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
	$(CLANG) -fPIC -c $(LOWERED_LL) -o $(OBJECT_FILE)

$(MAIN_OBJECT): $(MAIN_SRC) | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $(MAIN_SRC) -o $(MAIN_OBJECT)

$(LAYER_DISPATCH_OBJECT): $(ROOT)/tools/debug-simulator/shims/layer_dispatch.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@

$(LAYER_DISPATCH_MULTI_OBJECT): $(ROOT)/tools/debug-simulator/shims/layer_dispatch_multi_input.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@

$(WEIGHT_DISPATCH_OBJECT): $(ROOT)/tools/debug-simulator/shims/weight_dispatch.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@

$(ANALOG_OPERATIONS_OBJECT): $(ROOT)/tools/debug-simulator/shims/analog_operations.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_CXXFLAGS),$(CXXFLAGS)) -c $< -o $@


$(PYTHON_BRIDGE_OBJECT): $(ROOT)/tools/debug-simulator/shims/python_bridge.cpp | $(OUT_DIR)
	$(CLANGXX) $(if $(DEBUG_MODE),$(DEBUG_PYTHON_CXXFLAGS),$(PYTHON_CXXFLAGS)) -c $< -o $@

$(PYTHON_SO): $(OBJECT_FILE) | $(OUT_DIR)
	$(CLANGXX) -shared -Wl,--allow-shlib-undefined $(OBJECT_FILE) -o $(PYTHON_SO)

$(BINARY_FILE): $(OBJECT_FILE) $(MAIN_OBJECT) $(LAYER_DISPATCH_OBJECT) $(LAYER_DISPATCH_MULTI_OBJECT) $(WEIGHT_DISPATCH_OBJECT) $(ANALOG_OPERATIONS_OBJECT) $(PYTHON_BRIDGE_OBJECT) $(PYTHON_SO) | $(OUT_DIR)
	$(CLANGXX) -rdynamic \
	  $(OBJECT_FILE) \
	  $(MAIN_OBJECT) \
	  $(LAYER_DISPATCH_OBJECT) \
	  $(LAYER_DISPATCH_MULTI_OBJECT) \
	  $(WEIGHT_DISPATCH_OBJECT) \
	  $(ANALOG_OPERATIONS_OBJECT) \
	  $(PYTHON_BRIDGE_OBJECT) \
	  $(PYTHON_LDFLAGS) \
	  -o $(BINARY_FILE)


clean:
	rm -rf $(OUT_DIR)
