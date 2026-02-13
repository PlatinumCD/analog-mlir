# analog-mlir

**analog-mlir** is an experimental MLIR-based compiler infrastructure for targeting **analog compute-in-memory (CIM)** architectures.
It extends MLIR with an `analog` dialect and a sequence of transformation passes that progressively lower high-level tensor constants and linear algebra into representations suitable for analog arrays.

# Analog IR Overview

## 1. Types

| Type | Mnemonic | Represents | Notes |
|---|---|---|---|
| `Analog_MatrixType` | `matrix` | Logical analog matrix storage | Owns full matrix shape |
| `Analog_VectorType` | `vector` | Logical analog vector storage | 1D container |
| `Analog_MatrixGridType` | `matrix.grid` | 2D grid view over matrix arrays | View derived from matrix |
| `Analog_VectorSliceType` | `vector.slice` | 1D slice view over vector partitions | View derived from vector |

---

## 2. Operations

| Operation | Inputs | Outputs | Effect |
|---|---|---|---|
| `analog.matrix.from_tensor` | `tensor` | `Analog_MatrixType` | Materialize analog matrix storage |
| `analog.matrix.partition` | `Analog_MatrixType` | `Analog_MatrixGridType` | Declare matrix as a grid |
| `analog.array.matrix.place` | `MatrixGrid`, `rowIndex`, `colIndex`, `indices…` | — | Place a single matrix partition into accelerator |
| `analog.vector.from_tensor` | `tensor` | `Analog_VectorType` | Materialize analog vector storage |
| `analog.vector.partition` | `Analog_VectorType` | `Analog_VectorSliceType` | Declare vector as partitioned slices |
| `analog.array.vector.place` | `VectorSlice`, `sliceIndex`, `indices…` | — | Place a single vector slice |
| `analog.array.execute` | `indices…` | `Analog_MatrixGridType` | Execute placed arrays |
| `analog.array.store` | `MatrixGrid`, `memref`, `indices…` | — | Store accelerator results |

---

## 3. Conceptual Dataflow

| Stage | Matrix Path | Vector Path |
|---|---|---|
| Materialize | `matrix.from_tensor` | `vector.from_tensor` |
| Partition | `matrix.partition → MatrixGridType` | `vector.partition → VectorSliceType` |
| Placement | `array.matrix.place (row, col)` | `array.vector.place (slice)` |
| Execute | `array.execute` | *(implicit via array.execute)* |
| Store | `array.store` | *(folded into combine)* |

---

## 4. Passes

| Pass | Purpose | Operates On |
|---|---|---|
| `MaterializeMatrixFromTensorPass` | Tensor → analog matrix | Tensor constants |
| `MaterializeVectorFromTensorPass` | Tensor → analog vector | Tensor constants |
| `PartitionMatrixPass` | Matrix → array grid | `Analog_MatrixType` |
| `PartitionVectorPass` | Vector → vector slice | `Analog_VectorType` |
| `PlaceMatricesPass` | MatrixGrid → placed partitions | `Analog_MatrixGridType` |
| `PlaceVectorsPass` | VectorSlice → placed partitions | `Analog_VectorSliceType` |
| `ExecuteArrayPass` | Issue accelerator execution | Placed arrays |
| `CombineArrayResultsPass` | Reduce / writeback results | Array outputs |

---

## 5. Pipelines

| Pipeline | Contains |
|---|---|
| `MaterializePipeline` | Matrix + vector materialization |
| `PartitionPipeline` | Matrix and vector partitioning |
| `PlacePipeline` | Placement and execution prep |
