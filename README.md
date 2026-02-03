# analog-mlir

**analog-mlir** is an experimental MLIR-based compiler infrastructure for targeting **analog compute-in-memory (CIM)** architectures.
It extends MLIR with an `analog` dialect and a sequence of transformation passes that progressively lower high-level tensor constants and linear algebra into representations suitable for analog tile arrays.

The project focuses on **explicitly modeling analog weights, tiles, and execution structure in IR**, enabling systematic hardware–software co-design rather than ad-hoc backend lowering.

---

## Project Goals

* Make analog compute **explicit in the IR**, not implicit in backend codegen
* Separate **logical model structure** from **physical analog layout**
* Serve as a research platform for analog accelerator compilation

---

## High-Level Architecture

The compiler pipeline is organized as a sequence of **semantic refinements**:

```
Dense tensor constants
        ↓
!analog.weights        (logical, layer-aware weights)
        ↓
!analog.tile           (physical tile layout)
        ↓
Analog execution ops   (future work)
```

Each step introduces additional hardware-relevant structure while preserving analyzability and transformation legality.

---

## The `analog` Dialect

The `analog` dialect introduces first-class IR constructs for analog compute:

### Types

* `!analog.weights<shape x element-type>`

  * Represents a logical weight matrix intended for analog execution
  * Carries layer identity and separates weights from generic tensors

* `!analog.tile<shape x element-type, stride, base>`

  * Represents a physically realizable analog tile
  * Encodes layout information needed for array mapping

### Operations

* `analog.weights_from_const`

  * Materializes analog weights from dense constant tensors

* `analog.tiles_from_weights`

  * Converts logical weights into tile-level representations

These ops deliberately **do not perform computation**; they model *data placement and structure*.

---

## Transformation Passes

### `MaterializeWeightsFromConstPass`

**Argument:** `-analog-materialize-weights`

* Identifies `arith.constant` operations backed by `DenseResourceElementsAttr`
* Replaces implicit constants with explicit `!analog.weights`
* Assigns a monotonically increasing layer index
* Annotates constants to prevent duplicate materialization

This pass establishes **analog-aware ownership of model parameters**.

---

### `MaterializeTilesFromWeightsPass`

**Argument:** `-analog-materialize-tiles`

* Consumes `analog.weights_from_const` operations
* Produces `!analog.tile` values via `analog.tiles_from_weights`
* Uses configurable tile geometry (`tile_rows`, `tile_cols`)
* Currently handles the single-tile case explicitly
* Prepares weights for physical array mapping

This pass bridges **logical weights** and **physical analog layout**.

---

## Build Instructions

### Prerequisites

* LLVM + MLIR (built from source)
* CMake ≥ 3.22
* Ninja (recommended)

### Build

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
ninja
```

The primary tool is:

```bash
build/bin/analog-mlir-opt
```

---

## Example Usage

```bash
analog-mlir-opt input.mlir \
  -analog-materialize-weights \
  -analog-materialize-tiles="tile-rows=32 tile-cols=32"
```

This will progressively rewrite dense tensor constants into analog-aware representations.
