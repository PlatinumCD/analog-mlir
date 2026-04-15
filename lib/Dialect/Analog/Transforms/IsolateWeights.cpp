#include "analog-mlir/Dialect/Analog/Transforms/IsolateWeights.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

#include <string>

namespace mlir {
namespace analog {

namespace {

// Adds an operation and its entry-block producers to the set cloned into the
// weight helper.
void collectTopLevelDependencies(
    mlir::Operation *op, mlir::Block *entryBlock,
    llvm::SmallPtrSetImpl<mlir::Operation *> &weightOpSet) {
  if (!weightOpSet.insert(op).second)
    return;

  // Keep the whole top-level producer chain with the placement loop so the
  // extracted weight helper remains self-contained.
  op->walk([&](mlir::Operation *nestedOp) {
    for (mlir::Value operand : nestedOp->getOperands()) {
      mlir::Operation *definingOp = operand.getDefiningOp();
      if (!definingOp || definingOp->getBlock() != entryBlock)
        continue;

      collectTopLevelDependencies(definingOp, entryBlock, weightOpSet);
    }
  });
}

// Recognizes loops that place a matrix and therefore own weight setup.
bool isMatrixPlacementLoop(mlir::scf::ForOp loop) {
  if (!loop->hasAttr("matrix_id"))
    return false;

  bool hasMatrixPlaceOp = false;
  loop.walk([&](mlir::analog::ArrayMatrixPlaceOp) { hasMatrixPlaceOp = true; });
  return hasMatrixPlaceOp;
}

// Finds the single matrix placement loop and its top-level dependencies.
mlir::LogicalResult collectWeightOps(
    mlir::func::FuncOp func,
    llvm::SmallVectorImpl<mlir::Operation *> &weightOps, int64_t &matrixId) {
  mlir::Block &entryBlock = func.getBody().front();
  llvm::SmallPtrSet<mlir::Operation *, 16> weightOpSet;
  bool foundWeightLoop = false;

  // Validate the expected placement shape before moving any weight setup.
  for (mlir::Operation &op : entryBlock) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
      if (!isMatrixPlacementLoop(loop))
        continue;

      if (foundWeightLoop) {
        func.emitError("expected a single top-level matrix placement loop");
        return mlir::failure();
      }

      mlir::analog::ArrayMatrixPlaceOp matrixPlaceOp;
      loop.walk([&](mlir::analog::ArrayMatrixPlaceOp placeOp) {
        if (!matrixPlaceOp)
          matrixPlaceOp = placeOp;
      });
      if (!matrixPlaceOp) {
        func.emitError("expected matrix placement loop to contain "
                       "analog.array.matrix.place");
        return mlir::failure();
      }

      auto matrixIdAttr = loop->getAttrOfType<mlir::IntegerAttr>("matrix_id");
      if (!matrixIdAttr) {
        func.emitError("expected matrix placement loop to carry matrix_id");
        return mlir::failure();
      }

      auto partitionOp =
          matrixPlaceOp->getOperand(0).getDefiningOp<mlir::analog::MatrixPartitionOp>();
      if (!partitionOp) {
        func.emitError("expected matrix placement operand to come from "
                       "analog.matrix.partition");
        return mlir::failure();
      }

      auto materializeOp =
          partitionOp->getOperand(0).getDefiningOp<mlir::analog::MatrixFromTensorOp>();
      if (!materializeOp) {
        func.emitError("expected matrix partition operand to come from "
                       "analog.matrix.from_tensor");
        return mlir::failure();
      }

      auto constantOp =
          materializeOp->getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
      if (!constantOp) {
        func.emitError("expected matrix materialization operand to come from "
                       "arith.constant");
        return mlir::failure();
      }

      collectTopLevelDependencies(&op, &entryBlock, weightOpSet);
      matrixId = matrixIdAttr.getInt();
      foundWeightLoop = true;
    }
  }

  if (!foundWeightLoop)
    return mlir::success();

  // Preserve entry-block order so cloned dependencies keep dominance intact.
  for (mlir::Operation &op : entryBlock) {
    if (weightOpSet.contains(&op))
      weightOps.push_back(&op);
  }

  return mlir::success();
}

// Clones isolated weight setup into a private helper and records the dependency.
mlir::LogicalResult createWeightFunction(mlir::func::FuncOp func,
                                         mlir::RewriterBase &rewriter,
                                         mlir::func::FuncOp &weightsFunc) {
  auto module = func->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return mlir::failure();

  // Gather the movable operations before creating or mutating helper symbols.
  llvm::SmallVector<mlir::Operation *> weightOps;
  int64_t matrixId = 0;
  if (failed(collectWeightOps(func, weightOps, matrixId)))
    return mlir::failure();
  if (weightOps.empty())
    return mlir::success();

  std::string functionName = (func.getName() + "_weights").str();
  weightsFunc = module.lookupSymbol<mlir::func::FuncOp>(functionName);
  // Reuse an existing helper so rerunning the pass does not duplicate symbols.
  if (!weightsFunc) {
    auto functionType = rewriter.getFunctionType({}, {});
    rewriter.setInsertionPointToEnd(module.getBody());
    weightsFunc = rewriter.create<mlir::func::FuncOp>(
        func.getLoc(), functionName, functionType);
    weightsFunc.setPrivate();

    mlir::Block *entryBlock = weightsFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    mlir::IRMapping mapping;
    for (mlir::Operation *op : weightOps)
      rewriter.clone(*op, mapping);

    rewriter.create<mlir::func::ReturnOp>(func.getLoc());
  }

  // Record the helper relationship before erasing now-dead setup in the layer.
  weightsFunc->setAttr("layer_domain", rewriter.getStringAttr("analog"));
  weightsFunc->setAttr("layer_type", rewriter.getStringAttr("weight_init"));
  func->setAttr("weight_dependencies",
                rewriter.getArrayAttr({rewriter.getStringAttr(functionName)}));

  for (auto it = weightOps.rbegin(); it != weightOps.rend(); ++it) {
    if ((*it)->use_empty())
      rewriter.eraseOp(*it);
  }

  return mlir::success();
}

// Inserts each missing weight helper before the first forward call that needs it.
mlir::LogicalResult insertWeightDependencyCalls(mlir::func::FuncOp forward,
                                                mlir::RewriterBase &rewriter) {
  if (!forward.getBody().hasOneBlock()) {
    forward.emitError("expected forward to have a single block");
    return mlir::failure();
  }

  auto module = forward->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return mlir::failure();

  // Snapshot layer calls before inserting helpers so the traversal stays stable.
  llvm::SmallVector<mlir::func::CallOp> orderedCalls;
  for (mlir::Operation &op : forward.getBody().front()) {
    if (auto call = llvm::dyn_cast<mlir::func::CallOp>(&op))
      orderedCalls.push_back(call);
  }

  llvm::StringSet<> initializedHelpers;
  for (mlir::func::CallOp call : orderedCalls) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      continue;

    // Treat existing forward callees as already initialized, then insert each
    // missing helper once before the first dependent layer call.
    initializedHelpers.insert(calleeAttr.getValue());

    auto calleeFunc =
        module.lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue());
    if (!calleeFunc)
      continue;

    auto weightDependencies =
        calleeFunc->getAttrOfType<mlir::ArrayAttr>("weight_dependencies");
    if (!weightDependencies)
      continue;

    for (mlir::Attribute attr : weightDependencies) {
      auto dependencyName = llvm::dyn_cast<mlir::StringAttr>(attr);
      if (!dependencyName) {
        calleeFunc.emitError("expected weight_dependencies to be an array of "
                             "string attrs");
        return mlir::failure();
      }

      auto helperFunc =
          module.lookupSymbol<mlir::func::FuncOp>(dependencyName.getValue());
      if (!helperFunc) {
        calleeFunc.emitError("could not find weight dependency function '")
            << dependencyName.getValue() << "'";
        return mlir::failure();
      }

      if (!initializedHelpers.insert(dependencyName.getValue()).second)
        continue;

      rewriter.setInsertionPoint(call);
      rewriter.create<mlir::func::CallOp>(
          call.getLoc(), helperFunc.getSymName(),
          llvm::ArrayRef<mlir::Type>{}, llvm::ArrayRef<mlir::Value>{});
    }
  }

  return mlir::success();
}

} // namespace

// Splits analog layer weight setup into helpers and wires them into forward.
void IsolateWeightsPass::runOnOperation() {
  llvm::SmallVector<mlir::func::FuncOp> analogLayerFuncs;
  mlir::func::FuncOp forward;
  // First find analog layer functions and the optional forward dispatcher.
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getName() == "forward")
      forward = func;

    auto layerDomain = func->getAttrOfType<mlir::StringAttr>("layer_domain");
    if (!layerDomain || layerDomain.getValue() != "analog")
      continue;

    analogLayerFuncs.push_back(func);
  }

  mlir::IRRewriter rewriter(&getContext());
  // Then isolate per-layer weight setup into private helper functions.
  for (mlir::func::FuncOp func : analogLayerFuncs) {
    mlir::func::FuncOp weightsFunc;
    if (failed(createWeightFunction(func, rewriter, weightsFunc))) {
      signalPassFailure();
      return;
    }
  }

  // Finally ensure forward initializes helpers before dependent layer calls.
  if (forward && failed(insertWeightDependencyCalls(forward, rewriter))) {
    signalPassFailure();
    return;
  }
}

// Makes the isolate-weights pass available to the analog transform pipeline.
void registerIsolateWeightsPass() {
  PassRegistration<IsolateWeightsPass>();
}

} // namespace analog
} // namespace mlir
