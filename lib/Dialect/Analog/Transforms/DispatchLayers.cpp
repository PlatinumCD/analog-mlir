#include "analog-mlir/Dialect/Analog/Transforms/DispatchLayers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

constexpr StringLiteral kForwardFnName = "forward";
constexpr StringLiteral kDispatchLayer2DFnName = "analog_dispatch_layer_2d";
constexpr StringLiteral kDispatchLayer3DFnName = "analog_dispatch_layer_3d";
constexpr StringLiteral kDispatchLayer4DFnName = "analog_dispatch_layer_4d";
constexpr StringLiteral kDispatchLayer5DFnName = "analog_dispatch_layer_5d";
constexpr StringLiteral kRunLayer2DFnName = "analog_run_layer_2d";
constexpr StringLiteral kRunLayer3DFnName = "analog_run_layer_3d";
constexpr StringLiteral kRunLayer4DFnName = "analog_run_layer_4d";
constexpr StringLiteral kRunLayer5DFnName = "analog_run_layer_5d";
constexpr StringLiteral kWaitLayers2DFnName = "analog_wait_layers_2d";
constexpr StringLiteral kWaitLayers3DFnName = "analog_wait_layers_3d";
constexpr StringLiteral kWaitLayers4DFnName = "analog_wait_layers_4d";
constexpr StringLiteral kWaitLayers5DFnName = "analog_wait_layers_5d";
constexpr StringLiteral kInvokeLayerPrefix = "analog_invoke_layer_";
constexpr StringLiteral kLayerIdAttr = "layer-id";
constexpr StringLiteral kShimRequiredAttr = "analog-shim-required";

struct LayerCallInfo {
  func::CallOp call;
  int64_t layerId;
  FlatSymbolRefAttr callee;
  RankedTensorType inputTy;
  RankedTensorType resultTy;
};

struct DispatcherAbi {
  RankedTensorType dynInputTy;
  RankedTensorType dynResultTy;
};

struct LayerRuntimeHooks {
  func::FuncOp dispatchDecl;
  func::FuncOp waitDecl;
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


// Removes a named function symbol from the module when it exists.

static void eraseSymbolIfPresent(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name)) {
    fn.erase();
  }
}


// Removes all function symbols whose names share the given prefix so
// stale generated entrypoints do not accumulate.

static void eraseSymbolsWithPrefix(ModuleOp module, StringRef prefix) {
  SmallVector<func::FuncOp> toErase;
  for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
    if (fn.getSymName().starts_with(prefix)) {
      toErase.push_back(fn);
    }
  }
  for (func::FuncOp fn : toErase) {
    fn.erase();
  }
}


// Converts a ranked tensor type into an equivalent fully dynamic type
// for the shared dispatcher ABI.

static RankedTensorType makeDynamicLike(RankedTensorType ty) {
  SmallVector<int64_t> shape(ty.getRank(), ShapedType::kDynamic);
  return RankedTensorType::get(shape, ty.getElementType());
}


// Identifies calls that already target the runtime dispatch hooks so
// they are ignored during collection.

static bool isRuntimeDispatchCallee(StringRef calleeName) {
  return calleeName == kDispatchLayer2DFnName ||
         calleeName == kDispatchLayer3DFnName ||
         calleeName == kDispatchLayer4DFnName ||
         calleeName == kDispatchLayer5DFnName ||
         calleeName == kRunLayer2DFnName ||
         calleeName == kRunLayer3DFnName ||
         calleeName == kRunLayer4DFnName ||
         calleeName == kRunLayer5DFnName ||
         calleeName == kWaitLayers2DFnName ||
         calleeName == kWaitLayers3DFnName ||
         calleeName == kWaitLayers4DFnName ||
         calleeName == kWaitLayers5DFnName;
}


// Validates one candidate layer call and extracts the information
// needed for dispatcher generation.

static FailureOr<std::optional<LayerCallInfo>>
analyzeLayerCall(func::CallOp call) {
  auto idAttr = call->getAttrOfType<IntegerAttr>(kLayerIdAttr);
  if (!idAttr) {
    return std::optional<LayerCallInfo>{};
  }

  FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee) {
    call.emitError("expected direct func.call for layer-id call");
    return failure();
  }
  if (isRuntimeDispatchCallee(callee.getValue())) {
    return std::optional<LayerCallInfo>{};
  }

  if (call.getNumOperands() != 1 || call.getNumResults() != 1) {
    call.emitError("expected layer-id call to have 1 operand and 1 result");
    return failure();
  }

  auto inputTy = dyn_cast<RankedTensorType>(call.getOperand(0).getType());
  auto resultTy = dyn_cast<RankedTensorType>(call.getResult(0).getType());
  if (!inputTy || !resultTy) {
    call.emitError("expected tensor operand/result types on layer-id call");
    return failure();
  }
  if (inputTy.getElementType() != resultTy.getElementType()) {
    call.emitError("expected matching element type for layer dispatcher ABI");
    return failure();
  }

  return std::optional<LayerCallInfo>{LayerCallInfo{
      call,
      idAttr.getValue().getSExtValue(),
      callee,
      inputTy,
      resultTy,
  }};
}


// Collects the layer helper calls inside `forward` and validates that
// they match the expected dispatcher ABI shape.

static LogicalResult collectLayerCalls(func::FuncOp forward,
                                       SmallVectorImpl<LayerCallInfo> &layers) {
  bool hadError = false;
  forward.walk([&](func::CallOp call) {
    if (hadError) {
      return;
    }

    FailureOr<std::optional<LayerCallInfo>> maybeLayer = analyzeLayerCall(call);
    if (failed(maybeLayer)) {
      hadError = true;
      return;
    }
    if (*maybeLayer) {
      layers.push_back(**maybeLayer);
    }
  });
  return failure(hadError);
}


// Computes the fully dynamic tensor types shared by the unified layer
// dispatcher across all collected layer calls.

static FailureOr<DispatcherAbi>
computeUnifiedDispatcherTypes(MutableArrayRef<LayerCallInfo> layers) {
  RankedTensorType dynInputTy = makeDynamicLike(layers.front().inputTy);
  RankedTensorType dynResultTy = makeDynamicLike(layers.front().resultTy);
  for (LayerCallInfo &info : layers) {
    if (makeDynamicLike(info.inputTy) != dynInputTy ||
        makeDynamicLike(info.resultTy) != dynResultTy) {
      info.call.emitError("incompatible layer signatures for unified "
                          "analog_dispatch_layer ABI");
      return failure();
    }
  }
  return DispatcherAbi{dynInputTy, dynResultTy};
}


// Returns the rank-specific dispatch and wait hook names for the shared
// runtime ABI and rejects unsupported tensor ranks.
static FailureOr<std::pair<StringRef, StringRef>>
getRuntimeHookNamesForTypes(RankedTensorType dynInputTy,
                            RankedTensorType dynResultTy) {
  unsigned inputRank = dynInputTy.getRank();
  if (inputRank != dynResultTy.getRank())
    return failure();
  switch (inputRank) {
  case 2:
    return std::make_pair(StringRef(kDispatchLayer2DFnName),
                          StringRef(kWaitLayers2DFnName));
  case 3:
    return std::make_pair(StringRef(kDispatchLayer3DFnName),
                          StringRef(kWaitLayers3DFnName));
  case 4:
    return std::make_pair(StringRef(kDispatchLayer4DFnName),
                          StringRef(kWaitLayers4DFnName));
  case 5:
    return std::make_pair(StringRef(kDispatchLayer5DFnName),
                          StringRef(kWaitLayers5DFnName));
  default:
    return failure();
  }
}


// Returns the generated dispatcher function name for the given tensor rank.
static FailureOr<StringRef> getDispatcherNameForTypes(RankedTensorType dynInputTy,
                                                      RankedTensorType dynResultTy) {
  unsigned inputRank = dynInputTy.getRank();
  if (inputRank != dynResultTy.getRank())
    return failure();
  switch (inputRank) {
  case 2:
    return StringRef(kRunLayer2DFnName);
  case 3:
    return StringRef(kRunLayer3DFnName);
  case 4:
    return StringRef(kRunLayer4DFnName);
  case 5:
    return StringRef(kRunLayer5DFnName);
  default:
    return failure();
  }
}


// Ensures the rank-specific external runtime hook declarations exist and tags
// them as required shims for downstream lowering.
static FailureOr<LayerRuntimeHooks>
getOrCreateRuntimeHooks(ModuleOp module, RankedTensorType dynInputTy,
                        RankedTensorType dynResultTy) {
  OpBuilder moduleBuilder(module.getBodyRegion());
  auto i32Ty = moduleBuilder.getI32Type();
  FailureOr<std::pair<StringRef, StringRef>> maybeNames =
      getRuntimeHookNamesForTypes(dynInputTy, dynResultTy);
  if (failed(maybeNames))
    return failure();

  auto [dispatchName, waitName] = *maybeNames;
  auto dispatchTy =
      moduleBuilder.getFunctionType(TypeRange{dynInputTy, i32Ty}, TypeRange{});
  auto waitTy = moduleBuilder.getFunctionType(TypeRange{}, TypeRange{dynResultTy});

  func::FuncOp dispatchDecl =
      getOrCreateExternDecl(module, dispatchName, dispatchTy);
  func::FuncOp waitDecl =
      getOrCreateExternDecl(module, waitName, waitTy);
  dispatchDecl->setAttr(kShimRequiredAttr, moduleBuilder.getUnitAttr());
  waitDecl->setAttr(kShimRequiredAttr, moduleBuilder.getUnitAttr());
  return LayerRuntimeHooks{dispatchDecl, waitDecl};
}


// Replaces direct layer helper calls with dispatch and wait hook calls
// using the unified dynamic tensor ABI.

static void rewriteLayerCallSites(MutableArrayRef<LayerCallInfo> layers,
                                  LayerRuntimeHooks hooks,
                                  RankedTensorType dynInputTy,
                                  RankedTensorType dynResultTy) {
  for (LayerCallInfo &info : layers) {
    if (!info.call || !info.call->getBlock()) {
      continue;
    }

    OpBuilder b(info.call);
    Location loc = info.call.getLoc();
    Value idConst = b.create<arith::ConstantIntOp>(loc, info.layerId, 32);
    Value dynInput =
        b.create<tensor::CastOp>(loc, dynInputTy, info.call.getOperand(0));
    auto dispatch = b.create<func::CallOp>(loc, hooks.dispatchDecl.getSymName(),
                                           TypeRange{},
                                           ValueRange{dynInput, idConst});
    dispatch->setAttr(kLayerIdAttr, b.getI64IntegerAttr(info.layerId));
    auto wait = b.create<func::CallOp>(loc, hooks.waitDecl.getSymName(),
                                       TypeRange{dynResultTy}, ValueRange{});
    wait->setAttr(kLayerIdAttr, b.getI64IntegerAttr(info.layerId));
    Value typedResult =
        b.create<tensor::CastOp>(loc, info.resultTy, wait.getResult(0));

    info.call.getResult(0).replaceAllUsesWith(typedResult);
    info.call.erase();
  }
}


// Creates the public dispatcher function with its entry block and
// shared exit block for dynamic tensor results.

static std::pair<func::FuncOp, Block *> createLayerDispatcherSkeleton(
    ModuleOp module, StringRef fnName, RankedTensorType dynInputTy,
    RankedTensorType dynResultTy) {
  eraseSymbolIfPresent(module, fnName);

  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());

  auto i32Ty = b.getI32Type();
  auto fnTy = b.getFunctionType(TypeRange{dynInputTy, i32Ty},
                                TypeRange{dynResultTy});
  auto fn = b.create<func::FuncOp>(module.getLoc(), fnName, fnTy);
  fn.setPublic();

  Region &bodyRegion = fn.getBody();
  b.createBlock(&bodyRegion, bodyRegion.end(), TypeRange{dynInputTy, i32Ty},
                SmallVector<Location>{fn.getLoc(), fn.getLoc()});
  Block *exitBlock = b.createBlock(&bodyRegion);
  exitBlock->addArgument(dynResultTy, fn.getLoc());
  return {fn, exitBlock};
}


// Emits the dispatcher switch from runtime layer id to one case block
// per outlined layer helper.

static SmallVector<Block *> createDispatcherCaseBlocks(Region &bodyRegion,
                                                       size_t numCases,
                                                       OpBuilder &builder) {
  SmallVector<Block *> caseBlocks;
  caseBlocks.reserve(numCases);
  for (size_t i = 0; i < numCases; ++i) {
    caseBlocks.push_back(builder.createBlock(&bodyRegion));
  }
  return caseBlocks;
}


// Emits one dispatcher case block that casts the input, calls the
// matching layer helper, and branches to the shared exit block.

static void emitDispatcherCaseBlock(Block *caseBlock, Block *exitBlock,
                                    const LayerCallInfo &info, Value inputArg,
                                    RankedTensorType dynResultTy,
                                    Location loc) {
  OpBuilder caseBuilder = OpBuilder::atBlockEnd(caseBlock);
  Value typedInput =
      caseBuilder.create<tensor::CastOp>(loc, info.inputTy, inputArg);
  auto call = caseBuilder.create<func::CallOp>(loc, info.callee,
                                               TypeRange{info.resultTy},
                                               ValueRange{typedInput});
  call->setAttr(kLayerIdAttr, caseBuilder.getI64IntegerAttr(info.layerId));
  Value dynResult =
      caseBuilder.create<tensor::CastOp>(loc, dynResultTy, call.getResult(0));
  caseBuilder.create<cf::BranchOp>(loc, exitBlock, ValueRange{dynResult});
}


// Emits the default dispatcher block that traps on invalid layer ids
// and forwards a fallback dynamic tensor to the exit block.

static void emitDispatcherDefaultBlock(Block *defaultBlock, Block *exitBlock,
                                       Value inputArg,
                                       RankedTensorType dynResultTy,
                                       Location loc) {
  OpBuilder defaultBuilder = OpBuilder::atBlockEnd(defaultBlock);
  Value isValid = defaultBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
  defaultBuilder.create<cf::AssertOp>(loc, isValid, "invalid analog layer-id");

  Value fallback = inputArg;
  if (inputArg.getType() != dynResultTy) {
    auto inputTy = cast<RankedTensorType>(inputArg.getType());
    if (inputTy.getRank() == dynResultTy.getRank()) {
      fallback =
          defaultBuilder.create<tensor::CastOp>(loc, dynResultTy, inputArg);
    } else {
      SmallVector<Value> zeroDims;
      zeroDims.reserve(dynResultTy.getRank());
      for (int64_t i = 0; i < dynResultTy.getRank(); ++i) {
        zeroDims.push_back(defaultBuilder.create<arith::ConstantIndexOp>(loc, 0));
      }
      fallback = defaultBuilder.create<tensor::EmptyOp>(
          loc, dynResultTy.getShape(), dynResultTy.getElementType(), zeroDims);
    }
  }
  defaultBuilder.create<cf::BranchOp>(loc, exitBlock, ValueRange{fallback});
}


// Builds the public `analog_run_layer` dispatcher that switches on the
// runtime layer id and invokes the matching outlined layer helper.

static func::FuncOp createLayerDispatcher(ModuleOp module,
                                          ArrayRef<LayerCallInfo> layers,
                                          StringRef fnName,
                                          RankedTensorType dynInputTy,
                                          RankedTensorType dynResultTy) {
  auto [fn, exitBlock] =
      createLayerDispatcherSkeleton(module, fnName, dynInputTy, dynResultTy);
  Block *entry = &fn.getBody().front();
  Region &bodyRegion = fn.getBody();
  Value inputArg = entry->getArgument(0);
  Value layerIdArg = entry->getArgument(1);
  Location loc = fn.getLoc();

  OpBuilder regionBuilder = OpBuilder::atBlockEnd(entry);
  SmallVector<Block *> caseBlocks =
      createDispatcherCaseBlocks(bodyRegion, layers.size(), regionBuilder);
  Block *defaultBlock = regionBuilder.createBlock(&bodyRegion);

  OpBuilder entryBuilder = OpBuilder::atBlockEnd(entry);
  SmallVector<int32_t> caseValues;
  caseValues.reserve(layers.size());
  SmallVector<ValueRange> caseOperands(layers.size(), ValueRange{});
  for (const LayerCallInfo &info : layers) {
    caseValues.push_back(static_cast<int32_t>(info.layerId));
  }
  entryBuilder.create<cf::SwitchOp>(loc, layerIdArg, defaultBlock, ValueRange{},
                                    caseValues, caseBlocks, caseOperands);

  for (auto [info, caseBlock] : llvm::zip(layers, caseBlocks)) {
    emitDispatcherCaseBlock(caseBlock, exitBlock, info, inputArg, dynResultTy,
                            loc);
  }
  emitDispatcherDefaultBlock(defaultBlock, exitBlock, inputArg, dynResultTy,
                             loc);

  OpBuilder exitBuilder = OpBuilder::atBlockEnd(exitBlock);
  exitBuilder.create<func::ReturnOp>(loc, ValueRange{exitBlock->getArgument(0)});
  return fn;
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef DispatchLayersPass::getArgument() const {
  return "analog-dispatch-layers";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef DispatchLayersPass::getDescription() const {
  return "Create analog_run_layer and rewrite layer calls to "
         "dispatch/wait layer hooks";
}


// Declares the dialects this pass may create while building runtime
// dispatch wrappers and tensor casts.

void DispatchLayersPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
}


// Collects layer helper calls, creates the unified runtime dispatcher,
// and rewrites call sites to use dispatch and wait hooks.

void DispatchLayersPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp forward = module.lookupSymbol<func::FuncOp>(kForwardFnName);
  if (!forward || forward.isExternal()) {
    return;
  }

  SmallVector<LayerCallInfo> layers;
  if (failed(collectLayerCalls(forward, layers))) {
    signalPassFailure();
    return;
  }

  if (layers.empty()) {
    return;
  }

  SmallVector<LayerCallInfo> layers2D;
  SmallVector<LayerCallInfo> layers3D;
  SmallVector<LayerCallInfo> layers4D;
  SmallVector<LayerCallInfo> layers5D;
  for (LayerCallInfo &info : layers) {
    switch (info.inputTy.getRank()) {
    case 2:
      layers2D.push_back(info);
      break;
    case 3:
      layers3D.push_back(info);
      break;
    case 4:
      layers4D.push_back(info);
      break;
    case 5:
      layers5D.push_back(info);
      break;
    default:
      info.call.emitError("unsupported tensor rank for layer dispatcher ABI");
      signalPassFailure();
      return;
    }
  }

  eraseSymbolsWithPrefix(module, kInvokeLayerPrefix);
  auto processGroup = [&](MutableArrayRef<LayerCallInfo> group) -> LogicalResult {
    if (group.empty())
      return success();

    auto maybeTypes = computeUnifiedDispatcherTypes(group);
    if (failed(maybeTypes)) {
      return failure();
    }
    DispatcherAbi abi = *maybeTypes;

    FailureOr<LayerRuntimeHooks> maybeHooks =
        getOrCreateRuntimeHooks(module, abi.dynInputTy, abi.dynResultTy);
    if (failed(maybeHooks)) {
      forward.emitError("unsupported tensor rank for layer dispatch hooks");
      return failure();
    }

    FailureOr<StringRef> maybeDispatcherName =
        getDispatcherNameForTypes(abi.dynInputTy, abi.dynResultTy);
    if (failed(maybeDispatcherName)) {
      forward.emitError("unsupported tensor rank for layer dispatcher");
      return failure();
    }

    createLayerDispatcher(module, group, *maybeDispatcherName, abi.dynInputTy,
                          abi.dynResultTy);
    rewriteLayerCallSites(group, *maybeHooks, abi.dynInputTy, abi.dynResultTy);
    return success();
  };

  if (failed(processGroup(MutableArrayRef<LayerCallInfo>(layers2D))) ||
      failed(processGroup(MutableArrayRef<LayerCallInfo>(layers3D))) ||
      failed(processGroup(MutableArrayRef<LayerCallInfo>(layers4D))) ||
      failed(processGroup(MutableArrayRef<LayerCallInfo>(layers5D)))) {
    signalPassFailure();
    return;
  }
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createDispatchLayersPass() {
  return std::make_unique<DispatchLayersPass>();
}

} // namespace analog
} // namespace mlir
