#include "analog-mlir/Dialect/Analog/Transforms/DispatchWeights.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

struct WeightCallInfo {
  func::CallOp call;
  int64_t weightId;
  FlatSymbolRefAttr callee;
};

static func::FuncOp getOrCreateExternDecl(ModuleOp module, StringRef name,
                                          FunctionType type) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name)) {
    return fn;
  }

  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static void eraseSymbolIfPresent(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name)) {
    fn.erase();
  }
}

static func::FuncOp createWeightDispatcher(ModuleOp module,
                                           ArrayRef<WeightCallInfo> weights) {
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());

  auto i32Ty = b.getI32Type();
  auto fnTy = b.getFunctionType(TypeRange{i32Ty}, TypeRange{});
  auto fn = b.create<func::FuncOp>(module.getLoc(), "analog_run_weight", fnTy);
  fn.setPublic();

  Block *entry = fn.addEntryBlock();
  OpBuilder body = OpBuilder::atBlockEnd(entry);
  Value weightIdArg = entry->getArgument(0);

  for (const WeightCallInfo &info : weights) {
    Value idConst = body.create<arith::ConstantIntOp>(fn.getLoc(), info.weightId, 32);
    Value isMatch = body.create<arith::CmpIOp>(fn.getLoc(), arith::CmpIPredicate::eq,
                                               weightIdArg, idConst);

    body.create<scf::IfOp>(fn.getLoc(), isMatch, [&](OpBuilder &ifb, Location loc) {
      auto call = ifb.create<func::CallOp>(loc, info.callee, TypeRange{}, ValueRange{});
      call->setAttr("weight-id", ifb.getI64IntegerAttr(info.weightId));
      ifb.create<scf::YieldOp>(loc);
    });
  }

  body.create<func::ReturnOp>(fn.getLoc());
  return fn;
}

static func::FuncOp createWeightInitFunc(ModuleOp module,
                                         ArrayRef<WeightCallInfo> weights,
                                         func::FuncOp spawnDecl,
                                         func::FuncOp joinDecl) {
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());

  auto fnTy = b.getFunctionType(TypeRange{}, TypeRange{});
  auto fn = b.create<func::FuncOp>(module.getLoc(), "analog_init_weights", fnTy);

  Block *entry = fn.addEntryBlock();
  OpBuilder body = OpBuilder::atBlockEnd(entry);

  for (const WeightCallInfo &info : weights) {
    Value idConst = body.create<arith::ConstantIntOp>(fn.getLoc(), info.weightId, 32);
    auto spawn = body.create<func::CallOp>(fn.getLoc(), spawnDecl.getSymName(),
                                           TypeRange{}, ValueRange{idConst});
    spawn->setAttr("weight-id", body.getI64IntegerAttr(info.weightId));
  }

  body.create<func::CallOp>(fn.getLoc(), joinDecl.getSymName(), TypeRange{}, ValueRange{});
  body.create<func::ReturnOp>(fn.getLoc());
  return fn;
}

} // namespace

llvm::StringRef DispatchWeightsPass::getArgument() const {
  return "analog-dispatch-weights";
}

llvm::StringRef DispatchWeightsPass::getDescription() const {
  return "Create analog_init_weights and dispatch weight initialization calls via runtime dispatch hooks";
}

void DispatchWeightsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
}

void DispatchWeightsPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp forward = module.lookupSymbol<func::FuncOp>("forward");
  if (!forward || forward.isExternal()) {
    return;
  }

  SmallVector<WeightCallInfo> weights;
  llvm::DenseSet<int64_t> seenWeightIds;
  bool hadError = false;
  forward.walk([&](func::CallOp call) {
    if (hadError) {
      return;
    }
    auto idAttr = call->getAttrOfType<IntegerAttr>("weight-id");
    if (!idAttr) {
      return;
    }
    if (call.getNumOperands() != 0 || call.getNumResults() != 0) {
      call.emitError("expected weight-id call to have no operands/results");
      hadError = true;
      return;
    }
    FlatSymbolRefAttr callee = call.getCalleeAttr();
    if (!callee) {
      call.emitError("expected direct func.call for weight-id call");
      hadError = true;
      return;
    }
    int64_t weightId = idAttr.getValue().getSExtValue();
    if (!seenWeightIds.insert(weightId).second) {
      call.emitError("duplicate weight-id on call");
      hadError = true;
      return;
    }
    weights.push_back(WeightCallInfo{call, weightId, callee});
  });

  if (hadError) {
    signalPassFailure();
    return;
  }

  if (weights.empty()) {
    return;
  }

  OpBuilder moduleBuilder(module.getBodyRegion());
  auto i32Ty = moduleBuilder.getI32Type();
  auto spawnTy = moduleBuilder.getFunctionType(TypeRange{i32Ty}, TypeRange{});
  auto joinTy = moduleBuilder.getFunctionType(TypeRange{}, TypeRange{});
  func::FuncOp spawnDecl =
      getOrCreateExternDecl(module, "analog_dispatch_weight", spawnTy);
  func::FuncOp joinDecl =
      getOrCreateExternDecl(module, "analog_wait_weights", joinTy);
  spawnDecl->setAttr("analog-shim-required", moduleBuilder.getUnitAttr());
  joinDecl->setAttr("analog-shim-required", moduleBuilder.getUnitAttr());

  eraseSymbolIfPresent(module, "analog_run_weight");
  eraseSymbolIfPresent(module, "analog_init_weights");

  createWeightDispatcher(module, weights);
  createWeightInitFunc(module, weights, spawnDecl, joinDecl);

  for (WeightCallInfo &info : weights) {
    if (info.call && info.call->getBlock()) {
      info.call.erase();
    }
  }
}

std::unique_ptr<mlir::Pass> createDispatchWeightsPass() {
  return std::make_unique<DispatchWeightsPass>();
}

} // namespace analog
} // namespace mlir
