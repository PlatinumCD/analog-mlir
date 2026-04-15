#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/LoopUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"

#include <optional>

namespace mlir {
namespace analog {
namespace converter_utils {

// Assign stable ids that let matrix and vector lowering stages rendezvous.
int64_t nextMatrixId = 0;
int64_t nextVectorId = 0;

// Keep materialized analog values available for later id-based lookup.
llvm::SmallVector<Value, 16> processedMatrices;
llvm::SmallVector<Value, 16> processedVectors;

namespace {

// Reuses an existing id attribute or assigns the next id to the value producer.
FailureOr<int64_t> getOrSetId(Value value, OpBuilder &builder,
                              StringRef attrName, int64_t &nextId) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  auto existingId = definingOp->getAttrOfType<IntegerAttr>(attrName);
  if (existingId)
    return existingId.getInt();

  int64_t id = nextId++;
  definingOp->setAttr(attrName, builder.getI64IntegerAttr(id));
  return id;
}

// Assigns a specific id to the value producer and advances the global counter
// so later auto-assigned ids do not collide.
FailureOr<int64_t> setId(Value value, OpBuilder &builder,
                         StringRef attrName, int64_t id, int64_t &nextId) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  auto existingId = definingOp->getAttrOfType<IntegerAttr>(attrName);
  if (existingId) {
    if (existingId.getInt() != id)
      return mlir::failure();
    return existingId.getInt();
  }

  definingOp->setAttr(attrName, builder.getI64IntegerAttr(id));
  if (nextId <= id)
    nextId = id + 1;
  return id;
}

// Accepts only the 2D f32 tensors that the analog array lowering understands.
FailureOr<RankedTensorType> getRanked2DF32Tensor(Type type) {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return mlir::failure();

  if (tensorType.getRank() != 2)
    return mlir::failure();

  if (!tensorType.getElementType().isF32())
    return mlir::failure();

  return tensorType;
}

// Computes the number of array tiles needed to cover a matrix-like shape.
llvm::SmallVector<int64_t, 2> computeTilingGrid(ArrayRef<int64_t> shape,
                                                int64_t arrayRows,
                                                int64_t arrayCols) {
  return {
      (shape[0] + arrayRows - 1) / arrayRows,
      (shape[1] + arrayCols - 1) / arrayCols,
  };
}

} // namespace

// Assigns or retrieves the stable matrix id stored on the producing operation.
FailureOr<int64_t> getOrSetMatrixId(Value materializedMatrix,
                                    OpBuilder &builder) {
  return getOrSetId(materializedMatrix, builder, "matrix_id", nextMatrixId);
}

// Assigns or retrieves the stable vector id stored on the producing operation.
FailureOr<int64_t> getOrSetVectorId(Value materializedVector,
                                    OpBuilder &builder) {
  return getOrSetId(materializedVector, builder, "vector_id", nextVectorId);
}

// Finds a recorded matrix whose producer carries the requested matrix id.
FailureOr<Value> getMatrixWithId(int64_t matrixId) {
  for (Value matrix : processedMatrices) {
    Operation *definingOp = matrix.getDefiningOp();
    if (!definingOp)
      continue;

    auto existingId = definingOp->getAttrOfType<IntegerAttr>("matrix_id");
    if (!existingId)
      continue;

    if (existingId.getInt() == matrixId)
      return matrix;
  }

  return mlir::failure();
}

// Converts a weight constant into analog matrix IR and records it for reuse.
FailureOr<Value> materializeAnalogMatrix(arith::ConstantOp weightConstant,
                                         OpBuilder &builder) {
  auto tensorType = getRanked2DF32Tensor(weightConstant.getType());
  if (failed(tensorType))
    return mlir::failure();

  builder.setInsertionPointAfter(weightConstant);

  auto matrixType = analog::MatrixType::get(builder.getContext(),
                                            (*tensorType).getShape(),
                                            (*tensorType).getElementType());
  auto materialized = builder.create<analog::MatrixFromTensorOp>(
      weightConstant.getLoc(), matrixType, weightConstant.getResult());
  processedMatrices.push_back(materialized.getResult());

  auto matrixId = getOrSetMatrixId(materialized.getResult(), builder);
  if (failed(matrixId))
    return mlir::failure();

  return materialized.getResult();
}

// Tiles a materialized matrix into a grid sized for the target array.
FailureOr<Value> partitionAnalogMatrix(Value materializedMatrix,
                                       OpBuilder &builder,
                                       int64_t arrayRows,
                                       int64_t arrayCols) {
  if (arrayRows <= 0 || arrayCols <= 0)
    return mlir::failure();

  auto matrixType =
      llvm::dyn_cast<analog::MatrixType>(materializedMatrix.getType());
  if (!matrixType)
    return mlir::failure();

  auto matrixId = getOrSetMatrixId(materializedMatrix, builder);
  if (failed(matrixId))
    return mlir::failure();

  Operation *definingOp = materializedMatrix.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  builder.setInsertionPointAfter(definingOp);
  llvm::SmallVector<int64_t, 2> gridShape =
      computeTilingGrid(matrixType.getShape(), arrayRows, arrayCols);
  llvm::SmallVector<int64_t, 2> arrayShape{arrayRows, arrayCols};
  auto arrayGridType = analog::MatrixGridType::get(builder.getContext(),
                                                   gridShape, arrayShape,
                                                   matrixType);
  auto partition = builder.create<analog::MatrixPartitionOp>(
      definingOp->getLoc(), arrayGridType, materializedMatrix);
  partition->setAttr("matrix_id", builder.getI64IntegerAttr(*matrixId));

  return partition.getResult();
}

// Emits a placement loop nest that maps every matrix tile to an array address.
FailureOr<scf::ForOp> placeAnalogMatrix(Value partitionedMatrix,
                                        OpBuilder &builder) {
  auto matrixGridType =
      llvm::dyn_cast<analog::MatrixGridType>(partitionedMatrix.getType());
  if (!matrixGridType)
    return mlir::failure();

  auto matrixId = getOrSetMatrixId(partitionedMatrix, builder);
  if (failed(matrixId))
    return mlir::failure();

  auto gridShape = matrixGridType.getGridShape();
  int64_t numGridRows = gridShape[0];
  int64_t numGridCols = gridShape[1];

  Operation *definingOp = partitionedMatrix.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  Location loc = definingOp->getLoc();
  builder.setInsertionPointAfter(definingOp);
  scf::ForOp outerLoop = loop_utils::build2DIndexLoopNest(
      builder, loc, numGridRows, numGridCols,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value tr, Value tc) {
        loopBuilder.create<analog::ArrayMatrixPlaceOp>(
            loopLoc, partitionedMatrix, tr, tc, ValueRange{tr, tc});
      });
  outerLoop->setAttr("matrix_id", builder.getI64IntegerAttr(*matrixId));

  return outerLoop;
}

// Materializes a tensor value as an analog vector near its defining source and
// optionally pins it to a specific matrix id for later partition lookup.
static FailureOr<Value>
materializeAnalogVectorImpl(Value value,
                            std::optional<int64_t> explicitVectorId,
                            OpBuilder &builder) {
  auto tensorType = getRanked2DF32Tensor(value.getType());
  if (failed(tensorType))
    return mlir::failure();

  // Values produced by operations can be wrapped directly after their producer.
  Operation *definingOp = value.getDefiningOp();
  Location loc = builder.getUnknownLoc();
  if (definingOp) {
    loc = definingOp->getLoc();
    builder.setInsertionPointAfter(definingOp);
  } else {
    // Block arguments are placed after setup ops so later rewrites see inputs
    // before the first computation in the layer body.
    auto blockArg = llvm::dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return mlir::failure();

    loc = blockArg.getLoc();
    Block *ownerBlock = blockArg.getOwner();
    Operation *insertAfter = nullptr;
    for (Operation &op : *ownerBlock) {
      if (llvm::isa<arith::ConstantOp, analog::MatrixFromTensorOp,
                    analog::MatrixPartitionOp, scf::ForOp>(op)) {
        insertAfter = &op;
        continue;
      }
      break;
    }

    if (insertAfter)
      builder.setInsertionPointAfter(insertAfter);
    else
      builder.setInsertionPointToStart(ownerBlock);
  }

  auto vectorType = analog::VectorType::get(builder.getContext(),
                                            (*tensorType).getShape(),
                                            (*tensorType).getElementType());
  auto materialized = builder.create<analog::VectorFromTensorOp>(
      loc, vectorType, value);
  processedVectors.push_back(materialized.getResult());

  FailureOr<int64_t> vectorId =
      explicitVectorId ? setId(materialized.getResult(), builder, "vector_id",
                               *explicitVectorId, nextVectorId)
                       : getOrSetVectorId(materialized.getResult(), builder);
  if (failed(vectorId))
    return mlir::failure();

  return materialized.getResult();
}

// Materializes a tensor value as an analog vector near its defining source.
FailureOr<Value> materializeAnalogVector(Value value, OpBuilder &builder) {
  return materializeAnalogVectorImpl(value, std::nullopt, builder);
}

// Materializes a tensor value as an analog vector tied to a specific matrix id.
FailureOr<Value> materializeAnalogVector(Value value, int64_t vectorId,
                                         OpBuilder &builder) {
  return materializeAnalogVectorImpl(value, std::optional<int64_t>(vectorId),
                                     builder);
}

// Slices a vector according to the matrix grid that will consume it.
FailureOr<Value> partitionAnalogVector(Value materializedVector,
                                       OpBuilder &builder,
                                       int64_t arrayRows,
                                       int64_t arrayCols) {
  if (arrayRows <= 0 || arrayCols <= 0)
    return mlir::failure();

  auto vectorType =
      llvm::dyn_cast<analog::VectorType>(materializedVector.getType());
  if (!vectorType)
    return mlir::failure();

  auto vectorId = getOrSetVectorId(materializedVector, builder);
  if (failed(vectorId))
    return mlir::failure();

  auto matrix = getMatrixWithId(*vectorId);
  if (failed(matrix))
    return mlir::failure();

  auto matrixType = llvm::dyn_cast<analog::MatrixType>((*matrix).getType());
  if (!matrixType)
    return mlir::failure();

  Operation *definingOp = materializedVector.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  builder.setInsertionPointAfter(definingOp);

  llvm::SmallVector<int64_t, 2> gridShape =
      computeTilingGrid(matrixType.getShape(), arrayRows, arrayCols);
  llvm::SmallVector<int64_t, 2> arrayShape{arrayRows, arrayCols};
  auto vectorSliceType = analog::VectorSliceType::get(builder.getContext(),
                                                      gridShape, arrayShape,
                                                      vectorType);
  auto partition = builder.create<analog::VectorPartitionOp>(
      definingOp->getLoc(), vectorSliceType, materializedVector);
  partition->setAttr("vector_id", builder.getI64IntegerAttr(*vectorId));

  return partition.getResult();
}

// Emits a placement loop nest that maps every vector slice to an array input.
FailureOr<scf::ForOp> placeAnalogVector(Value partitionedVector,
                                        OpBuilder &builder) {
  auto vectorSliceType =
      llvm::dyn_cast<analog::VectorSliceType>(partitionedVector.getType());
  if (!vectorSliceType)
    return mlir::failure();

  auto gridShape = vectorSliceType.getGridShape();
  int64_t numGridRows = gridShape[0];
  int64_t numGridCols = gridShape[1];

  Operation *definingOp = partitionedVector.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  Location loc = definingOp->getLoc();
  builder.setInsertionPointAfter(definingOp);
  scf::ForOp outerLoop = loop_utils::build2DIndexLoopNest(
      builder, loc, numGridRows, numGridCols,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value tr, Value tc) {
        loopBuilder.create<analog::ArrayVectorPlaceOp>(
            loopLoc, partitionedVector, tc, ValueRange{tr, tc});
      });

  return outerLoop;
}

// Schedules array execution after both placement loops and stores tile outputs.
FailureOr<Value> insertArrayExecution(Value placedMatrix, Value placedVector,
                                      scf::ForOp matrixPlacementLoop,
                                      scf::ForOp vectorPlacementLoop,
                                      OpBuilder &builder) {
  // Validate that matrix and vector partitions describe the same grid.
  auto matrixGridType = llvm::dyn_cast<analog::MatrixGridType>(
      placedMatrix.getType());
  if (!matrixGridType)
    return mlir::failure();

  auto vectorSliceType = llvm::dyn_cast<analog::VectorSliceType>(
      placedVector.getType());
  if (!vectorSliceType)
    return mlir::failure();

  auto matrixGridShape = matrixGridType.getGridShape();
  if (matrixGridShape.size() != 2)
    return mlir::failure();

  auto vectorGridShape = vectorSliceType.getGridShape();
  if (vectorGridShape.size() != 2)
    return mlir::failure();

  if (matrixGridShape != vectorGridShape)
    return mlir::failure();

  auto arrayShape = matrixGridType.getArrayShape();
  if (arrayShape.size() != 2)
    return mlir::failure();

  int64_t numGridRows = matrixGridShape[0];
  int64_t numGridCols = matrixGridShape[1];
  int64_t arrayRows = arrayShape[0];

  // Insert after the later placement loop so execution sees both operands.
  Operation *matrixLoopOp = matrixPlacementLoop.getOperation();
  Operation *vectorLoopOp = vectorPlacementLoop.getOperation();
  if (!matrixLoopOp || !vectorLoopOp)
    return mlir::failure();

  // Conv2D reuses one top-level matrix placement loop across nested output
  // loops, so the vector placement may live in a deeper block. In that case,
  // insert execution after the vector loop because the matrix setup already
  // dominates the current location.
  Operation *insertAfter = vectorLoopOp;
  if (matrixLoopOp->getBlock() == vectorLoopOp->getBlock()) {
    insertAfter =
        matrixLoopOp->isBeforeInBlock(vectorLoopOp) ? vectorLoopOp : matrixLoopOp;
  }
  builder.setInsertionPointAfter(insertAfter);

  Location loc = insertAfter->getLoc();
  auto arrayOutputBufferType =
      MemRefType::get({numGridRows, numGridCols, arrayRows},
                      builder.getF32Type());
  Value arrayOutputBuffer =
      builder.create<memref::AllocOp>(loc, arrayOutputBufferType);

  // Execute each placed array tile and capture its lane output in a buffer.
  loop_utils::build2DIndexLoopNest(
      builder, loc, numGridRows, numGridCols,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value tr, Value tc) {
        Value array = loopBuilder
                          .create<analog::ArrayExecuteOp>(
                              loopLoc, matrixGridType, ValueRange{tr, tc})
                          .getResult();
        loopBuilder.create<analog::ArrayStoreOp>(
            loopLoc, array, arrayOutputBuffer, ValueRange{tr, tc});
      });

  return arrayOutputBuffer;
}

namespace {

// Groups constants shared by the reduction loops to avoid recreating them.
struct ReductionConstants {
  Value c0;
  Value c0f;
};

// Builds the index and floating-point zero constants used during reduction.
static ReductionConstants buildReductionConstants(OpBuilder &builder,
                                                  Location loc) {
  return ReductionConstants{
      builder.create<arith::ConstantIndexOp>(loc, 0),
      builder.create<arith::ConstantFloatOp>(
          loc, builder.getF32Type(), llvm::APFloat(0.0f)),
  };
}

// Allocates per-row accumulators and initializes every lane to zero.
static Value allocateZeroedRowBuffers(OpBuilder &builder, Location loc,
                                      int64_t gridRows, int64_t laneWidth,
                                      const ReductionConstants &constants) {
  auto rowBufferType =
      MemRefType::get({gridRows, laneWidth}, builder.getF32Type());
  Value rowBuffers = builder.create<memref::AllocOp>(loc, rowBufferType);
  loop_utils::build2DIndexLoopNest(
      builder, loc, gridRows, laneWidth,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value r, Value j) {
        loopBuilder.create<memref::StoreOp>(loopLoc, constants.c0f, rowBuffers,
                                            ValueRange{r, j});
      });

  return rowBuffers;
}

// Accumulates all grid-column outputs into one lane buffer per matrix row tile.
static void emitRowReduction(OpBuilder &builder, Location loc, Value rowBuffers,
                             Value executionBuffer, int64_t gridRows,
                             int64_t gridCols, int64_t laneWidth,
                             const ReductionConstants &constants) {
  loop_utils::build3DIndexLoopNest(
      builder, loc, gridRows, gridCols, laneWidth,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value r, Value c,
          Value j) {
        Value acc = loopBuilder.create<memref::LoadOp>(
            loopLoc, rowBuffers, ValueRange{r, j});
        Value val = loopBuilder.create<memref::LoadOp>(
            loopLoc, executionBuffer, ValueRange{r, c, j});
        Value sum = loopBuilder.create<arith::AddFOp>(loopLoc, acc, val);
        loopBuilder.create<memref::StoreOp>(
            loopLoc, sum, rowBuffers, ValueRange{r, j});
      });
}

// Converts reduced lane buffers back into the matmul result tensor shape.
static FailureOr<Value> materializeReducedTensor(
    OpBuilder &builder, Location loc, Value rowBuffers, int64_t gridRows,
    int64_t laneWidth, int64_t matrixRows,
    const ReductionConstants &constants, RankedTensorType resultType) {
  if (resultType.getRank() != 2)
    return mlir::failure();

  auto resultShape = resultType.getShape();
  if (resultShape.size() != 2 || resultShape[0] != 1)
    return mlir::failure();

  auto outBufferType = MemRefType::get(resultShape, resultType.getElementType());
  Value out = builder.create<memref::AllocOp>(loc, outBufferType);
  Value cMatrixRows = builder.create<arith::ConstantIndexOp>(loc, matrixRows);

  // Scatter only lanes that correspond to real matrix rows; edge tiles may pad.
  loop_utils::build2DIndexLoopNest(
      builder, loc, gridRows, laneWidth,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value r, Value j) {
        Value value = loopBuilder.create<memref::LoadOp>(
            loopLoc, rowBuffers, ValueRange{r, j});
        Value cLaneWidth =
            loopBuilder.create<arith::ConstantIndexOp>(loopLoc, laneWidth);
        Value colOffset =
            loopBuilder.create<arith::MulIOp>(loopLoc, r, cLaneWidth);
        Value col =
            loopBuilder.create<arith::AddIOp>(loopLoc, colOffset, j);
        Value inBounds = loopBuilder.create<arith::CmpIOp>(
            loopLoc, arith::CmpIPredicate::slt, col, cMatrixRows);
        loopBuilder.create<scf::IfOp>(
            loopLoc, inBounds,
            [&](OpBuilder &ifBuilder, Location ifLoc) {
              ifBuilder.create<memref::StoreOp>(
                  ifLoc, value, out, ValueRange{constants.c0, col});
              ifBuilder.create<scf::YieldOp>(ifLoc);
            });
      });

  auto toTensor =
      builder.create<bufferization::ToTensorOp>(loc, resultType, out);
  toTensor->setAttr("restrict", builder.getUnitAttr());
  return toTensor.getResult();
}

} // namespace

// Reduces array execution output and rewires the original matmul result.
FailureOr<Value> insertArrayReduction(Value executionBuffer,
                                      Value partitionedMatrix,
                                      linalg::MatmulOp matmulOp,
                                      OpBuilder &builder) {
  // Confirm the output buffer and matrix grid agree on the reduction shape.
  auto resultType =
      llvm::dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
  if (!resultType)
    return mlir::failure();

  auto gridType =
      llvm::dyn_cast<analog::MatrixGridType>(partitionedMatrix.getType());
  if (!gridType)
    return mlir::failure();

  auto gridShape = gridType.getGridShape();
  if (gridShape.size() != 2)
    return mlir::failure();

  auto matrixType = gridType.getMatrix();
  auto matrixShape = matrixType.getShape();
  if (matrixShape.size() != 2)
    return mlir::failure();

  auto bufferType = llvm::dyn_cast<MemRefType>(executionBuffer.getType());
  if (!bufferType)
    return mlir::failure();

  auto bufferShape = bufferType.getShape();
  if (bufferShape.size() != 3)
    return mlir::failure();

  int64_t gridRows = gridShape[0];
  int64_t gridCols = gridShape[1];
  int64_t matrixRows = matrixShape[0];
  int64_t laneWidth = bufferShape[2];

  Location loc = matmulOp.getLoc();
  builder.setInsertionPoint(matmulOp);

  // The accelerator returns one lane buffer per grid column. Reduce columns
  // into per-row buffers, then scatter valid lanes back to the tensor shape.
  ReductionConstants constants = buildReductionConstants(builder, loc);
  Value rowBuffers =
      allocateZeroedRowBuffers(builder, loc, gridRows, laneWidth, constants);
  emitRowReduction(builder, loc, rowBuffers, executionBuffer, gridRows,
                   gridCols, laneWidth, constants);

  auto reducedTensor =
      materializeReducedTensor(builder, loc, rowBuffers, gridRows, laneWidth,
                               matrixRows, constants, resultType);
  if (failed(reducedTensor))
    return mlir::failure();

  matmulOp.getResult(0).replaceAllUsesWith(*reducedTensor);
  return *reducedTensor;
}

} // namespace converter_utils
} // namespace analog
} // namespace mlir
