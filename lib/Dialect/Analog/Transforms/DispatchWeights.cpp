#include "analog-mlir/Dialect/Analog/Transforms/DispatchWeights.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

constexpr StringLiteral kForwardFnName = "forward";
constexpr StringLiteral kDispatchWeightFnName = "analog_dispatch_weight";
constexpr StringLiteral kWaitWeightsFnName = "analog_wait_weights";
constexpr StringLiteral kRunWeightFnName = "analog_run_weight";
constexpr StringLiteral kInitWeightsFnName = "analog_init_weights";
constexpr StringLiteral kWeightIdAttr = "weight-id";
constexpr StringLiteral kShimRequiredAttr = "analog-shim-required";

struct WeightCallInfo {
  func::CallOp call;
  int64_t weightId;
  FlatSymbolRefAttr callee;
};


// Returns an existing external declaration for the runtime hook or
// creates a new private declaration if it is missing.

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


// Creates a new function at module scope with an entry block ready for
// body construction.

static func::FuncOp createModuleFunction(ModuleOp module, StringRef name,
                                         FunctionType type, bool isPublic) {
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
  if (isPublic) {
    fn.setPublic();
  } else {
    fn.setPrivate();
  }
  fn.addEntryBlock();
  return fn;
}


// Validates one candidate weight call and extracts the information
// needed for runtime weight dispatching.

static FailureOr<std::optional<WeightCallInfo>>
analyzeWeightCall(func::CallOp call, llvm::DenseSet<int64_t> &seenWeightIds) {
  auto idAttr = call->getAttrOfType<IntegerAttr>(kWeightIdAttr);
  if (!idAttr) {
    return std::optional<WeightCallInfo>{};
  }
  if (call.getNumOperands() != 0 || call.getNumResults() != 0) {
    call.emitError("expected weight-id call to have no operands/results");
    return failure();
  }

  FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee) {
    call.emitError("expected direct func.call for weight-id call");
    return failure();
  }

  int64_t weightId = idAttr.getValue().getSExtValue();
  if (!seenWeightIds.insert(weightId).second) {
    call.emitError("duplicate weight-id on call");
    return failure();
  }

  return std::optional<WeightCallInfo>{WeightCallInfo{call, weightId, callee}};
}


// Collects the unique outlined weight routines referenced from the
// forward function.

static LogicalResult collectWeightCalls(func::FuncOp forward,
                                        SmallVectorImpl<WeightCallInfo> &weights) {
  llvm::DenseSet<int64_t> seenWeightIds;
  bool hadError = false;
  forward.walk([&](func::CallOp call) {
    if (hadError) {
      return;
    }

    FailureOr<std::optional<WeightCallInfo>> maybeWeight =
        analyzeWeightCall(call, seenWeightIds);
    if (failed(maybeWeight)) {
      hadError = true;
      return;
    }
    if (*maybeWeight) {
      weights.push_back(**maybeWeight);
    }
  });
  return failure(hadError);
}


// Removes a named function symbol from the module when it exists.

static void eraseSymbolIfPresent(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name)) {
    fn.erase();
  }
}


// Builds the public weight dispatcher that selects one outlined weight
// routine based on the runtime weight id.

static func::FuncOp createWeightDispatcher(ModuleOp module,
                                           ArrayRef<WeightCallInfo> weights) {
  OpBuilder b(module.getBodyRegion());
  auto i32Ty = b.getI32Type();
  auto fnTy = b.getFunctionType(TypeRange{i32Ty}, TypeRange{});
  auto fn = createModuleFunction(module, kRunWeightFnName, fnTy, true);
  Block *entry = &fn.getBody().front();
  OpBuilder body = OpBuilder::atBlockEnd(entry);
  Value weightIdArg = entry->getArgument(0);

  for (const WeightCallInfo &info : weights) {
    Value idConst = body.create<arith::ConstantIntOp>(fn.getLoc(), info.weightId, 32);
    Value isMatch = body.create<arith::CmpIOp>(fn.getLoc(), arith::CmpIPredicate::eq,
                                               weightIdArg, idConst);

    body.create<scf::IfOp>(fn.getLoc(), isMatch, [&](OpBuilder &ifb, Location loc) {
      auto call = ifb.create<func::CallOp>(loc, info.callee, TypeRange{}, ValueRange{});
      call->setAttr(kWeightIdAttr, ifb.getI64IntegerAttr(info.weightId));
      ifb.create<scf::YieldOp>(loc);
    });
  }

  body.create<func::ReturnOp>(fn.getLoc());
  return fn;
}


// Builds the public initialization entrypoint that dispatches all
// weight routines and waits for their completion.

static func::FuncOp createWeightInitFunc(ModuleOp module,
                                         ArrayRef<WeightCallInfo> weights,
                                         func::FuncOp spawnDecl,
                                         func::FuncOp joinDecl) {
  OpBuilder b(module.getBodyRegion());
  auto fnTy = b.getFunctionType(TypeRange{}, TypeRange{});
  auto fn = createModuleFunction(module, kInitWeightsFnName, fnTy, true);
  Block *entry = &fn.getBody().front();
  OpBuilder body = OpBuilder::atBlockEnd(entry);

  for (const WeightCallInfo &info : weights) {
    Value idConst = body.create<arith::ConstantIntOp>(fn.getLoc(), info.weightId, 32);
    auto spawn = body.create<func::CallOp>(fn.getLoc(), spawnDecl.getSymName(),
                                           TypeRange{}, ValueRange{idConst});
    spawn->setAttr(kWeightIdAttr, body.getI64IntegerAttr(info.weightId));
  }

  body.create<func::CallOp>(fn.getLoc(), joinDecl.getSymName(), TypeRange{}, ValueRange{});
  body.create<func::ReturnOp>(fn.getLoc());
  return fn;
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef DispatchWeightsPass::getArgument() const {
  return "analog-dispatch-weights";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef DispatchWeightsPass::getDescription() const {
  return "Create analog_init_weights and dispatch weight initialization calls via runtime dispatch hooks";
}


// Declares the dialects this pass may create while building runtime
// weight dispatch wrappers.

void DispatchWeightsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
}


// Collects outlined weight routines, creates runtime dispatch helpers,
// and removes the original direct call sites.

void DispatchWeightsPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp forward = module.lookupSymbol<func::FuncOp>(kForwardFnName);
  if (!forward || forward.isExternal()) {
    return;
  }

  SmallVector<WeightCallInfo> weights;
  if (failed(collectWeightCalls(forward, weights))) {
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
      getOrCreateExternDecl(module, kDispatchWeightFnName, spawnTy);
  func::FuncOp joinDecl =
      getOrCreateExternDecl(module, kWaitWeightsFnName, joinTy);
  spawnDecl->setAttr(kShimRequiredAttr, moduleBuilder.getUnitAttr());
  joinDecl->setAttr(kShimRequiredAttr, moduleBuilder.getUnitAttr());

  eraseSymbolIfPresent(module, kRunWeightFnName);
  eraseSymbolIfPresent(module, kInitWeightsFnName);

  createWeightDispatcher(module, weights);
  createWeightInitFunc(module, weights, spawnDecl, joinDecl);

  for (WeightCallInfo &info : weights) {
    if (info.call && info.call->getBlock()) {
      info.call.erase();
    }
  }
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createDispatchWeightsPass() {
  return std::make_unique<DispatchWeightsPass>();
}

} // namespace analog
} // namespace mlir
