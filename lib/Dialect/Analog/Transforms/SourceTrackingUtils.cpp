#include "analog-mlir/Dialect/Analog/Transforms/SourceTrackingUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformAttrs.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace detail {

using detail::kMatrixSourceIdAttr;

llvm::DenseMap<int64_t, analog::MatrixGridType>
collectMatrixGridsBySourceId(func::FuncOp func) {
  llvm::DenseMap<int64_t, analog::MatrixGridType> gridByMatrixSourceId;
  func.walk([&](::mlir::analog::MatrixPartitionOp op) {
    auto gridTy =
        llvm::dyn_cast<::mlir::analog::MatrixGridType>(op.getResult().getType());
    if (!gridTy)
      return;
    auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId)
      return;
    gridByMatrixSourceId.try_emplace(matrixSourceId.getInt(), gridTy);
  });
  return gridByMatrixSourceId;
}

IntegerAttr getMatrixSourceIdAttr(Operation *op) {
  return op ? op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr) : IntegerAttr();
}

IntegerAttr findMatrixSourceIdOnValue(Value value) {
  if (!value)
    return {};

  if (auto definingOp = value.getDefiningOp()) {
    if (auto matrixSourceId = getMatrixSourceIdAttr(definingOp))
      return matrixSourceId;
  }

  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>())
    return constantOp->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);

  if (auto transposeOp = value.getDefiningOp<linalg::TransposeOp>())
    return findMatrixSourceIdOnValue(transposeOp.getInput());

  return {};
}

IntegerAttr getOrInferMatmulSourceId(linalg::MatmulOp op) {
  if (auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr))
    return matrixSourceId;

  if (op.getInputs().size() < 2)
    return {};

  IntegerAttr matrixSourceId = findMatrixSourceIdOnValue(op.getInputs()[1]);
  if (!matrixSourceId)
    return {};

  op->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  return matrixSourceId;
}

int64_t getNextMatrixSourceId(func::FuncOp func) {
  int64_t nextMatrixSourceId = 0;
  func.walk([&](arith::ConstantOp op) {
    auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId)
      return;
    nextMatrixSourceId =
        std::max(nextMatrixSourceId, matrixSourceId.getInt() + 1);
  });
  return nextMatrixSourceId;
}

IntegerAttr getOrCreateMatrixSourceId(arith::ConstantOp op,
                                      int64_t &nextMatrixSourceId) {
  if (auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr))
    return matrixSourceId;

  auto matrixSourceId = IntegerAttr::get(
      IntegerType::get(op.getContext(), 64), nextMatrixSourceId++);
  op->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  return matrixSourceId;
}

void propagateMatrixSourceId(arith::ConstantOp op, Operation *materializedOp) {
  if (!materializedOp)
    return;
  if (auto matrixSourceId = op->getAttr(kMatrixSourceIdAttr))
    materializedOp->setAttr(kMatrixSourceIdAttr, matrixSourceId);
}

} // namespace detail
} // namespace analog
} // namespace mlir
