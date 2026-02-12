# analog-mlir

**analog-mlir** is an experimental MLIR-based compiler infrastructure for targeting **analog compute-in-memory (CIM)** architectures.
It extends MLIR with an `analog` dialect and a sequence of transformation passes that progressively lower high-level tensor constants and linear algebra into representations suitable for analog tile arrays.

# Analog IR Overview

## 1. Types

| Type | Mnemonic | Represents | Notes |
|---|---|---|---|
| `Analog_MatrixType` | `matrix` | Logical analog matrix storage | Owns full matrix shape |
| `Analog_VectorType` | `vector` | Logical analog vector storage | 1D container |
| `Analog_TileGridType` | `tile.grid` | 2D grid view over matrix tiles | View derived from matrix |
| `Analog_VTileSliceType` | `vtile.slice` | 1D slice view over vector tiles | View derived from vector |

---

## 2. Operations

| Operation | Inputs | Outputs | Effect |
|---|---|---|---|
| `analog.matrix.from_tensor` | `tensor` | `Analog_MatrixType` | Materialize analog matrix storage |
| `analog.tile.partition` | `Analog_MatrixType` | `Analog_TileGridType` | Declare matrix as tiled grid |
| `analog.tile.place` | `TileGrid`, `rowIndex`, `colIndex`, `indices…` | — | Place a single tile into accelerator |
| `analog.vector.from_tensor` | `tensor` | `Analog_VectorType` | Materialize analog vector storage |
| `analog.vtile.partition` | `Analog_VectorType` | `Analog_VTileSliceType` | Declare vector as vtile slices |
| `analog.vtile.place` | `VTileSlice`, `sliceIndex`, `indices…` | — | Place a single vtile |
| `analog.tile.execute` | `indices…` | `Analog_TileGridType` | Execute placed tiles |
| `analog.tile.store` | `TileGrid`, `memref`, `indices…` | — | Store accelerator results |

---

## 3. Conceptual Dataflow

| Stage | Matrix Path | Vector Path |
|---|---|---|
| Materialize | `matrix.from_tensor` | `vector.from_tensor` |
| Partition | `tile.partition → TileGridType` | `vtile.partition → VTileSliceType` |
| Placement | `tile.place (row, col)` | `vtile.place (slice)` |
| Execute | `tile.execute` | *(implicit via tile.execute)* |
| Store | `tile.store` | *(folded into combine)* |

---

## 4. Passes

| Pass | Purpose | Operates On |
|---|---|---|
| `MaterializeMatrixFromTensorPass` | Tensor → analog matrix | Tensor constants |
| `MaterializeVectorFromTensorPass` | Tensor → analog vector | Tensor constants |
| `PartitionMatrixPass` | Matrix → tile grid | `Analog_MatrixType` |
| `PartitionVectorPass` | Vector → vtile slice | `Analog_VectorType` |
| `PlaceTilesPass` | TileGrid → placed tiles | `Analog_TileGridType` |
| `PlaceVTilesPass` | VTileSlice → placed vtiles | `Analog_VTileSliceType` |
| `ExecuteTilesPass` | Issue accelerator execution | Placed tiles |
| `CombineTileResultsPass` | Reduce / writeback results | Tile outputs |

---

## 5. Pipelines

| Pipeline | Contains |
|---|---|
| `MaterializePipeline` | Matrix + vector materialization |
| `PartitionPipeline` | Matrix and vector partitioning |
| `PlacePipeline` | Placement and execution prep |

