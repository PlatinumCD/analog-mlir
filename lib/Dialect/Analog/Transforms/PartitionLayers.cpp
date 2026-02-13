#include "analog-mlir/Dialect/Analog/Transforms/PartitionLayers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

static bool isBefore(Operation *lhs, Operation *rhs) {
  return lhs != rhs && lhs->isBeforeInBlock(rhs);
}

static void addUniqueOp(SmallVectorImpl<Operation *> &ops, Operation *op) {
  if (!op)
    return;
  if (llvm::is_contained(ops, op))
    return;
  ops.push_back(op);
}

static void collectLayerOps(linalg::MatmulOp matmul,
                            SmallVectorImpl<Operation *> &layerOps) {
  Operation *matmulOp = matmul.getOperation();
  Block *block = matmulOp->getBlock();

  addUniqueOp(layerOps, matmulOp);

  // RHS path: linalg.transpose -> matmul(rhs operand)
  if (Value rhs = matmulOp->getOperand(1)) {
    if (auto transpose = rhs.getDefiningOp<linalg::TransposeOp>()) {
      if (transpose->getBlock() == block)
        addUniqueOp(layerOps, transpose.getOperation());
    }
  }

  // Output-init path: linalg.fill -> matmul(outs operand)
  if (Value outInit = matmulOp->getOperand(2)) {
    if (auto fill = outInit.getDefiningOp<linalg::FillOp>()) {
      if (fill->getBlock() == block)
        addUniqueOp(layerOps, fill.getOperation());
    }
  }

  llvm::sort(layerOps, [](Operation *a, Operation *b) { return isBefore(a, b); });
}

} // namespace

llvm::StringRef PartitionLayersPass::getArgument() const {
  return "analog-partition-layers";
}

llvm::StringRef PartitionLayersPass::getDescription() const {
  return "Wrap each NN layer block in a numbered execute_region";
}

void PartitionLayersPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
}

void PartitionLayersPass::runOnOperation() {
  func::FuncOp func = getOperation();

  SmallVector<arith::ConstantOp> denseConstants;
  func.walk([&](arith::ConstantOp op) {
    if (!op->getBlock() || !isa<func::FuncOp>(op->getBlock()->getParentOp()))
      return;
    if (!llvm::isa<DenseResourceElementsAttr>(op.getValue()))
      return;
    denseConstants.push_back(op);
  });

  int64_t denseLayerGroup = 0;
  for (arith::ConstantOp cst : denseConstants) {
    if (!cst->getBlock())
      continue;

    OpBuilder builder(cst);
    auto exec = builder.create<scf::ExecuteRegionOp>(cst.getLoc(), cst.getType());
    exec->setAttr("layer_group", builder.getI64IntegerAttr(denseLayerGroup++));
    exec->setAttr("group_kind", builder.getStringAttr("dense_resource"));

    Block *regionBlock = new Block();
    exec.getRegion().push_back(regionBlock);

    cst->moveBefore(regionBlock, regionBlock->end());

    OpBuilder regionBuilder = OpBuilder::atBlockEnd(regionBlock);
    auto yield =
        regionBuilder.create<scf::YieldOp>(cst.getLoc(), cst.getResult());
    cst.getResult().replaceAllUsesExcept(exec.getResult(0), yield.getOperation());
  }

  SmallVector<linalg::MatmulOp> matmuls;
  func.walk([&](linalg::MatmulOp op) {
    if (op->getBlock() && isa<func::FuncOp>(op->getBlock()->getParentOp()))
      matmuls.push_back(op);
  });

  int64_t layerId = 0;
  for (linalg::MatmulOp matmul : matmuls) {
    if (!matmul->getBlock())
      continue;

    SmallVector<Operation *> layerOps;
    collectLayerOps(matmul, layerOps);
    if (layerOps.empty())
      continue;

    OpBuilder builder(matmul);
    auto exec = builder.create<scf::ExecuteRegionOp>(
        matmul.getLoc(), matmul->getResultTypes());
    exec->setAttr("layer_group", builder.getI64IntegerAttr(layerId++));
    exec->setAttr("group_kind", builder.getStringAttr("layer"));

    Block *regionBlock = new Block();
    exec.getRegion().push_back(regionBlock);

    for (Operation *op : layerOps)
      op->moveBefore(regionBlock, regionBlock->end());

    OpBuilder regionBuilder = OpBuilder::atBlockEnd(regionBlock);
    auto yield = regionBuilder.create<scf::YieldOp>(matmul.getLoc(),
                                                    matmul->getResults());

    for (auto it : llvm::zip(matmul->getResults(), exec->getResults()))
      std::get<0>(it).replaceAllUsesExcept(std::get<1>(it), yield.getOperation());
  }
}

std::unique_ptr<mlir::Pass> createPartitionLayersPass() {
  return std::make_unique<PartitionLayersPass>();
}

} // namespace analog
} // namespace mlir
