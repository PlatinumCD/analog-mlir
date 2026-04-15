# analog-mlir

**analog-mlir** is an experimental MLIR-based compiler for lowering a small set
of tensor and `linalg` kernels into an **analog compute-in-memory (CIM)**
execution model, then lowering that model to the repo's **Golem simulator**
runtime.

The project is organized around one concrete flow:

1. extract supported layer-shaped regions from `forward`
2. convert those outlined layer functions into the `analog` dialect
3. isolate weight initialization into helper functions
4. assemble a symbolic task graph
5. lower analog execution to Golem-oriented MLIR
6. emit runtime graph builders 

## Current Scope

The registered extraction and conversion path currently supports these layer
families:

| Layer Family | `layer_type` Values |
|---|---|
| Linear | `linear`, `linear_w_bias` |
| Conv1D | `conv1d`, `conv1d_w_bias` |
| Conv2D | `conv2d`, `conv2d_w_bias` |
| Grouped Conv2D | `conv2d_grouped`, `conv2d_grouped_w_bias` |
| Conv3D | `conv3d`, `conv3d_w_bias` |
| RNN Cell | `rnn_cell`, `rnn_cell_w_bias` |

## Analog IR Overview

### 1. Types

| Type | Mnemonic | Represents | Notes |
|---|---|---|---|
| `Analog_MatrixType` | `matrix` | Logical analog matrix storage | Owns the full matrix shape |
| `Analog_VectorType` | `vector` | Logical analog vector storage | Owns the full vector shape |
| `Analog_MatrixGridType` | `matrix.grid` | 2D tiled view over matrix arrays | View derived from `matrix` |
| `Analog_VectorSliceType` | `vector.slice` | Partitioned view over vector slices | View derived from `vector` |
| `Analog_TaskGraphType` | `task_graph` | Symbolic task-graph handle | Root handle for graph construction |
| `Analog_TaskType` | `task` | Symbolic task node handle | Produced by `analog.task.create` |
| `Analog_RuntimeHandleType` | `runtime_handle` | Opaque persistent runtime state | Used for isolated weight resources |
| `Analog_TaskResourceType` | `task_resource` | Symbolic graph resource handle | Wraps a concrete payload type |

### 2. Operations

#### Analog Execution IR

| Operation | Inputs | Outputs | Effect |
|---|---|---|---|
| `analog.matrix.from_tensor` | `tensor` | `Analog_MatrixType` | Materialize analog matrix storage |
| `analog.matrix.partition` | `Analog_MatrixType` | `Analog_MatrixGridType` | Declare the matrix as a tiled array grid |
| `analog.array.matrix.place` | `MatrixGrid`, `rowIndex`, `colIndex`, `indices...` | — | Place one matrix tile into the accelerator |
| `analog.vector.from_tensor` | `tensor` | `Analog_VectorType` | Materialize analog vector storage |
| `analog.vector.partition` | `Analog_VectorType` | `Analog_VectorSliceType` | Declare the vector as partitioned slices |
| `analog.array.vector.place` | `VectorSlice`, `sliceIndex`, `indices...` | — | Place one vector slice into the accelerator |
| `analog.array.execute` | `indices...` | `Analog_MatrixGridType` | Execute placed analog arrays |
| `analog.array.store` | `MatrixGrid`, `memref`, `indices...` | — | Store accelerator results into memory |

#### Task Graph IR

| Operation | Inputs | Outputs | Effect |
|---|---|---|---|
| `analog.task_graph.create` | — | `Analog_TaskGraphType` | Create a task-graph handle |
| `analog.task_graph.input` | `TaskGraph` | `TaskResource` | Create a graph input resource |
| `analog.task_graph.output` | `TaskGraph` | `TaskResource` | Create a graph output resource |
| `analog.task_graph.temporary` | `TaskGraph` | `TaskResource` | Create an internal temporary resource |
| `analog.task_graph.persistent` | `TaskGraph` | `TaskResource` | Create a persistent runtime-owned resource |
| `analog.task.create` | `TaskGraph`, callee, resources, deps | `TaskType` | Create one symbolic analog or digital task node |

## Passes

| Pass | Purpose | Operates On |
|---|---|---|
| `analog-extract-layers` | Outline supported layer-shaped regions from `forward` | Digital `forward` IR |
| `analog-convert-layers` | Rewrite extracted digital layer functions into analog execution IR | Outlined layer functions |
| `analog-isolate-weights` | Split persistent matrix setup into private weight-init helpers | Analog layer functions |
| `analog-assemble-task-graph` | Build the symbolic task graph and runtime execution plan | `forward` plus outlined helpers |
| `analog-lower-to-golem` | Lower analog execution/storage IR into Golem-oriented MLIR | Analog execution IR + task graph shell |
| `analog-emit-runtime-graph` | Emit runtime graph builders and public runtime entry points | Lowered LLVM-ready module with task graph |

## Recommended Pipeline Order

### Front Half

1. `analog-extract-layers`
2. `analog-convert-layers`
3. `analog-isolate-weights`

### Full Analog-to-Runtime Flow

1. `analog-extract-layers`
2. `analog-convert-layers`
3. `analog-isolate-weights`
4. `analog-assemble-task-graph`
5. `analog-lower-to-golem`
6. bufferization and LLVM lowering
7. `analog-emit-runtime-graph`

## Typical Commands

### Build

By default the helper scripts expect an LLVM/MLIR build tree at
`$HOME/Develop/analog/build/llvm-project`.

```bash
./configure.sh
./build.sh
```

Common overrides:

- `BUILD_DIR`
- `LLVM_BUILD`
- `LLVM_DIR`
- `MLIR_DIR`
- `CMAKE_BUILD_TYPE`
