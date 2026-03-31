#include "analog-mlir/Dialect/Analog/Transforms/DispatchLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformAttrs.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

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

struct LayerCallInfo {
  func::CallOp call;
  int64_t layerId;
  FlatSymbolRefAttr callee;
  SmallVector<RankedTensorType> operandTys;
  RankedTensorType resultTy;
};

enum class LayerAbiClass {
  singleInputSingleResult,
  multiInputSingleResult,
};

struct LayerSignature {
  SmallVector<RankedTensorType> operandTys;
  SmallVector<RankedTensorType> resultTys;
  SmallVector<RankedTensorType> dynamicOperandTys;
  SmallVector<RankedTensorType> dynamicResultTys;

  bool isSingleInputSingleResult() const {
    return operandTys.size() == 1 && resultTys.size() == 1;
  }

  bool isMultiInputSingleResult() const {
    return operandTys.size() > 1 && resultTys.size() == 1;
  }

  bool isSupportedForCurrentDispatcherAbi() const {
    return resultTys.size() == 1 && operandTys.size() >= 1 &&
           operandTys.size() <= 2;
  }

  LayerAbiClass classifyAbi() const {
    return operandTys.size() == 1 ? LayerAbiClass::singleInputSingleResult
                                  : LayerAbiClass::multiInputSingleResult;
  }
};

struct LayerRuntimeHooks {
  func::FuncOp dispatchDecl;
  func::FuncOp waitDecl;
};

// Returns an existing external declaration for the runtime hook or
// creates a new private declaration if it is missing.
static func::FuncOp getOrCreateExternDecl(ModuleOp module, StringRef name,
                                          FunctionType type) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

// Removes a named function symbol from the module when it exists.
static void eraseSymbolIfPresent(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    fn.erase();
}

// Removes all function symbols whose names share the given prefix so
// stale generated entrypoints do not accumulate.
static void eraseSymbolsWithPrefix(ModuleOp module, StringRef prefix) {
  SmallVector<func::FuncOp> toErase;
  for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
    if (fn.getSymName().starts_with(prefix))
      toErase.push_back(fn);
  }
  for (func::FuncOp fn : toErase)
    fn.erase();
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
  return calleeName.starts_with("analog_dispatch_layer_") ||
         calleeName.starts_with("analog_run_layer_") ||
         calleeName.starts_with("analog_wait_layers_");
}

// Validates one candidate layer call and extracts the information
// needed for dispatcher generation.
static FailureOr<std::optional<LayerCallInfo>> analyzeLayerCall(func::CallOp call) {
  auto idAttr = call->getAttrOfType<IntegerAttr>(detail::kLayerIdAttr);
  if (!idAttr)
    return std::optional<LayerCallInfo>{};

  FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee) {
    call.emitError("expected direct func.call for layer-id call");
    return failure();
  }
  if (isRuntimeDispatchCallee(callee.getValue()))
    return std::optional<LayerCallInfo>{};

  if (call.getNumOperands() < 1 || call.getNumResults() != 1) {
    call.emitError("expected layer-id call to have at least 1 operand and exactly 1 result");
    return failure();
  }

  auto resultTy = dyn_cast<RankedTensorType>(call.getResult(0).getType());
  if (!resultTy) {
    call.emitError("expected tensor operand/result types on layer-id call");
    return failure();
  }

  SmallVector<RankedTensorType> operandTys;
  operandTys.reserve(call.getNumOperands());
  for (Value operand : call.getOperands()) {
    auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandTy) {
      call.emitError("expected tensor operand/result types on layer-id call");
      return failure();
    }
    if (operandTy.getElementType() != resultTy.getElementType()) {
      call.emitError("expected matching element type for layer dispatcher ABI");
      return failure();
    }
    operandTys.push_back(operandTy);
  }

  return std::optional<LayerCallInfo>{LayerCallInfo{
      call,
      idAttr.getValue().getSExtValue(),
      callee,
      operandTys,
      resultTy,
  }};
}

// Collects the layer helper calls inside `forward` and validates that
// they match the expected dispatcher ABI shape.
static LogicalResult collectLayerCalls(func::FuncOp forward,
                                       SmallVectorImpl<LayerCallInfo> &layers) {
  bool hadError = false;
  forward.walk([&](func::CallOp call) {
    if (hadError)
      return;

    FailureOr<std::optional<LayerCallInfo>> maybeLayer = analyzeLayerCall(call);
    if (failed(maybeLayer)) {
      hadError = true;
      return;
    }
    if (*maybeLayer)
      layers.push_back(**maybeLayer);
  });
  return failure(hadError);
}

// Computes the fully dynamic tensor types shared by the unified layer
// dispatcher across all collected layer calls.
static LayerSignature buildLayerSignature(const LayerCallInfo &info) {
  LayerSignature signature;
  signature.operandTys.append(info.operandTys.begin(), info.operandTys.end());
  signature.resultTys.push_back(info.resultTy);
  for (RankedTensorType operandTy : info.operandTys)
    signature.dynamicOperandTys.push_back(makeDynamicLike(operandTy));
  signature.dynamicResultTys.push_back(makeDynamicLike(info.resultTy));
  return signature;
}

static FailureOr<LayerSignature>
computeUnifiedLayerSignature(MutableArrayRef<LayerCallInfo> layers) {
  LayerSignature signature = buildLayerSignature(layers.front());
  for (LayerCallInfo &info : layers) {
    LayerSignature candidate = buildLayerSignature(info);
    if (candidate.dynamicOperandTys != signature.dynamicOperandTys ||
        candidate.dynamicResultTys != signature.dynamicResultTys) {
      info.call.emitError("incompatible layer signatures for unified analog_dispatch_layer ABI");
      return failure();
    }
  }
  return signature;
}

static FailureOr<unsigned> getSupportedTensorRank(RankedTensorType ty) {
  switch (ty.getRank()) {
  case 2:
  case 3:
  case 4:
  case 5:
    return ty.getRank();
  default:
    return failure();
  }
}

static FailureOr<std::string>
getDispatchHookNameForSignature(const LayerSignature &signature) {
  if (!signature.isSupportedForCurrentDispatcherAbi())
    return failure();

  FailureOr<unsigned> maybeResultRank =
      getSupportedTensorRank(signature.dynamicResultTys.front());
  if (failed(maybeResultRank))
    return failure();

  if (signature.isSingleInputSingleResult()) {
    FailureOr<unsigned> maybeInputRank =
        getSupportedTensorRank(signature.dynamicOperandTys.front());
    if (failed(maybeInputRank) || *maybeInputRank != *maybeResultRank)
      return failure();
    switch (*maybeResultRank) {
    case 2:
      return std::string(kDispatchLayer2DFnName);
    case 3:
      return std::string(kDispatchLayer3DFnName);
    case 4:
      return std::string(kDispatchLayer4DFnName);
    case 5:
      return std::string(kDispatchLayer5DFnName);
    default:
      return failure();
    }
  }

  if (signature.operandTys.size() != 2)
    return failure();
  FailureOr<unsigned> maybeInput0Rank =
      getSupportedTensorRank(signature.dynamicOperandTys[0]);
  FailureOr<unsigned> maybeInput1Rank =
      getSupportedTensorRank(signature.dynamicOperandTys[1]);
  if (failed(maybeInput0Rank) || failed(maybeInput1Rank))
    return failure();

  return ("analog_dispatch_layer_" + std::to_string(*maybeResultRank) +
          "d_from_" + std::to_string(*maybeInput0Rank) + "d_" +
          std::to_string(*maybeInput1Rank) + "d");
}

static FailureOr<std::string>
getWaitHookNameForSignature(const LayerSignature &signature) {
  if (!signature.isSupportedForCurrentDispatcherAbi())
    return failure();

  FailureOr<unsigned> maybeResultRank =
      getSupportedTensorRank(signature.dynamicResultTys.front());
  if (failed(maybeResultRank))
    return failure();

  if (signature.isSingleInputSingleResult()) {
    switch (*maybeResultRank) {
    case 2:
      return std::string(kWaitLayers2DFnName);
    case 3:
      return std::string(kWaitLayers3DFnName);
    case 4:
      return std::string(kWaitLayers4DFnName);
    case 5:
      return std::string(kWaitLayers5DFnName);
    default:
      return failure();
    }
  }

  if (signature.operandTys.size() != 2)
    return failure();
  FailureOr<unsigned> maybeInput0Rank =
      getSupportedTensorRank(signature.dynamicOperandTys[0]);
  FailureOr<unsigned> maybeInput1Rank =
      getSupportedTensorRank(signature.dynamicOperandTys[1]);
  if (failed(maybeInput0Rank) || failed(maybeInput1Rank))
    return failure();

  return ("analog_wait_layers_" + std::to_string(*maybeResultRank) +
          "d_from_" + std::to_string(*maybeInput0Rank) + "d_" +
          std::to_string(*maybeInput1Rank) + "d");
}

static FailureOr<std::string>
getDispatcherNameForSignature(const LayerSignature &signature) {
  if (!signature.isSupportedForCurrentDispatcherAbi())
    return failure();

  FailureOr<unsigned> maybeResultRank =
      getSupportedTensorRank(signature.dynamicResultTys.front());
  if (failed(maybeResultRank))
    return failure();

  if (signature.isSingleInputSingleResult()) {
    switch (*maybeResultRank) {
    case 2:
      return std::string(kRunLayer2DFnName);
    case 3:
      return std::string(kRunLayer3DFnName);
    case 4:
      return std::string(kRunLayer4DFnName);
    case 5:
      return std::string(kRunLayer5DFnName);
    default:
      return failure();
    }
  }

  if (signature.operandTys.size() != 2)
    return failure();
  FailureOr<unsigned> maybeInput0Rank =
      getSupportedTensorRank(signature.dynamicOperandTys[0]);
  FailureOr<unsigned> maybeInput1Rank =
      getSupportedTensorRank(signature.dynamicOperandTys[1]);
  if (failed(maybeInput0Rank) || failed(maybeInput1Rank))
    return failure();

  return ("analog_run_layer_" + std::to_string(*maybeResultRank) +
          "d_from_" + std::to_string(*maybeInput0Rank) + "d_" +
          std::to_string(*maybeInput1Rank) + "d");
}

// Ensures the external runtime hook declarations exist and tags
// them as required shims for downstream lowering.
static FailureOr<LayerRuntimeHooks>
getOrCreateRuntimeHooks(ModuleOp module, const LayerSignature &signature) {
  if (!signature.isSupportedForCurrentDispatcherAbi())
    return failure();

  RankedTensorType dynResultTy = signature.dynamicResultTys.front();
  auto dynResultMemRefTy =
      MemRefType::get(dynResultTy.getShape(), dynResultTy.getElementType());
  OpBuilder moduleBuilder(module.getBodyRegion());
  auto i32Ty = moduleBuilder.getI32Type();
  FailureOr<std::string> maybeDispatchName =
      getDispatchHookNameForSignature(signature);
  FailureOr<std::string> maybeWaitName =
      getWaitHookNameForSignature(signature);
  if (failed(maybeDispatchName) || failed(maybeWaitName))
    return failure();

  SmallVector<Type> dispatchInputs(signature.dynamicOperandTys.begin(),
                                   signature.dynamicOperandTys.end());
  dispatchInputs.push_back(dynResultMemRefTy);
  dispatchInputs.push_back(i32Ty);
  auto dispatchTy = moduleBuilder.getFunctionType(dispatchInputs, TypeRange{});
  auto waitTy = moduleBuilder.getFunctionType(TypeRange{}, TypeRange{});

  func::FuncOp dispatchDecl =
      getOrCreateExternDecl(module, *maybeDispatchName, dispatchTy);
  func::FuncOp waitDecl =
      getOrCreateExternDecl(module, *maybeWaitName, waitTy);
  dispatchDecl->setAttr(detail::kShimRequiredAttr, moduleBuilder.getUnitAttr());
  waitDecl->setAttr(detail::kShimRequiredAttr, moduleBuilder.getUnitAttr());
  return LayerRuntimeHooks{dispatchDecl, waitDecl};
}

// Replaces direct layer helper calls with dispatch and wait hook calls
// using the unified dynamic tensor ABI.
static LogicalResult rewriteLayerCallSites(MutableArrayRef<LayerCallInfo> layers,
                                           LayerRuntimeHooks hooks,
                                           const LayerSignature &signature) {
  RankedTensorType dynResultTy = signature.dynamicResultTys.front();
  auto dynResultMemRefTy =
      MemRefType::get(dynResultTy.getShape(), dynResultTy.getElementType());
  for (LayerCallInfo &info : layers) {
    if (!info.call || !info.call->getBlock())
      continue;

    if (!info.resultTy.hasStaticShape()) {
      info.call.emitError("wait-by-reference layer dispatch currently requires static result shapes");
      return failure();
    }

    OpBuilder b(info.call);
    Location loc = info.call.getLoc();
    Value idConst = b.create<arith::ConstantIntOp>(loc, info.layerId, 32);
    auto resultBufferTy =
        MemRefType::get(info.resultTy.getShape(), info.resultTy.getElementType());
    Value resultBuffer = b.create<memref::AllocOp>(loc, resultBufferTy);
    Value dynResultBuffer =
        b.create<memref::CastOp>(loc, dynResultMemRefTy, resultBuffer);

    SmallVector<Value> dispatchOperands;
    dispatchOperands.reserve(signature.dynamicOperandTys.size() + 2);
    for (auto [operand, dynOperandTy] :
         llvm::zip(info.call.getOperands(), signature.dynamicOperandTys)) {
      dispatchOperands.push_back(
          b.create<tensor::CastOp>(loc, dynOperandTy, operand));
    }
    dispatchOperands.push_back(dynResultBuffer);
    dispatchOperands.push_back(idConst);

    auto dispatch = b.create<func::CallOp>(loc, hooks.dispatchDecl.getSymName(),
                                           TypeRange{}, dispatchOperands);
    dispatch->setAttr(detail::kLayerIdAttr, b.getI64IntegerAttr(info.layerId));
    auto wait = b.create<func::CallOp>(loc, hooks.waitDecl.getSymName(),
                                       TypeRange{}, ValueRange{});
    wait->setAttr(detail::kLayerIdAttr, b.getI64IntegerAttr(info.layerId));
    auto typedResult =
        b.create<bufferization::ToTensorOp>(loc, info.resultTy, resultBuffer);
    typedResult->setAttr("restrict", b.getUnitAttr());

    info.call.getResult(0).replaceAllUsesWith(typedResult.getResult());
    info.call.erase();
  }
  return success();
}

// Creates the public dispatcher function with its entry block and
// shared exit block for void completion after filling the output buffer.
static std::pair<func::FuncOp, Block *> createLayerDispatcherSkeleton(
    ModuleOp module, StringRef fnName, ArrayRef<RankedTensorType> dynOperandTys,
    RankedTensorType dynResultTy) {
  eraseSymbolIfPresent(module, fnName);

  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());

  auto dynResultMemRefTy =
      MemRefType::get(dynResultTy.getShape(), dynResultTy.getElementType());
  auto i32Ty = b.getI32Type();
  SmallVector<Type> fnInputs(dynOperandTys.begin(), dynOperandTys.end());
  fnInputs.push_back(dynResultMemRefTy);
  fnInputs.push_back(i32Ty);
  auto fnTy = b.getFunctionType(fnInputs, TypeRange{});
  auto fn = b.create<func::FuncOp>(module.getLoc(), fnName, fnTy);
  fn.setPublic();

  Region &bodyRegion = fn.getBody();
  SmallVector<Type> entryArgTypes(dynOperandTys.begin(), dynOperandTys.end());
  entryArgTypes.push_back(dynResultMemRefTy);
  entryArgTypes.push_back(i32Ty);
  SmallVector<Location> entryArgLocs(entryArgTypes.size(), fn.getLoc());
  b.createBlock(&bodyRegion, bodyRegion.end(), entryArgTypes, entryArgLocs);
  Block *exitBlock = b.createBlock(&bodyRegion);
  return {fn, exitBlock};
}

// Emits the dispatcher switch from runtime layer id to one case block
// per outlined layer helper.
static SmallVector<Block *> createDispatcherCaseBlocks(Region &bodyRegion,
                                                       size_t numCases,
                                                       OpBuilder &builder) {
  SmallVector<Block *> caseBlocks;
  caseBlocks.reserve(numCases);
  for (size_t i = 0; i < numCases; ++i)
    caseBlocks.push_back(builder.createBlock(&bodyRegion));
  return caseBlocks;
}

// Emits one dispatcher case block that casts inputs, invokes the
// matching layer helper, copies its result into the provided output
// buffer, and branches to the shared exit block.
static void emitDispatcherCaseBlock(Block *caseBlock, Block *exitBlock,
                                    const LayerCallInfo &info,
                                    ArrayRef<Value> inputArgs,
                                    Value outBufferArg,
                                    Location loc) {
  OpBuilder caseBuilder = OpBuilder::atBlockEnd(caseBlock);
  SmallVector<Value> typedInputs;
  typedInputs.reserve(info.operandTys.size());
  for (auto [inputArg, operandTy] : llvm::zip(inputArgs, info.operandTys))
    typedInputs.push_back(
        caseBuilder.create<tensor::CastOp>(loc, operandTy, inputArg));
  auto call = caseBuilder.create<func::CallOp>(loc, info.callee,
                                               TypeRange{info.resultTy},
                                               typedInputs);
  call->setAttr(detail::kLayerIdAttr, caseBuilder.getI64IntegerAttr(info.layerId));

  auto resultBufferTy =
      MemRefType::get(info.resultTy.getShape(), info.resultTy.getElementType());
  Value sourceBuffer =
      caseBuilder.create<bufferization::ToBufferOp>(loc, resultBufferTy,
                                                    call.getResult(0));
  Value typedOutBuffer =
      caseBuilder.create<memref::CastOp>(loc, resultBufferTy, outBufferArg);
  caseBuilder.create<memref::CopyOp>(loc, sourceBuffer, typedOutBuffer);
  caseBuilder.create<cf::BranchOp>(loc, exitBlock);
}

// Emits the default dispatcher block that traps on invalid layer ids
// and branches to the shared void exit block.
static void emitDispatcherDefaultBlock(Block *defaultBlock, Block *exitBlock,
                                       Location loc) {
  OpBuilder defaultBuilder = OpBuilder::atBlockEnd(defaultBlock);
  Value isValid = defaultBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
  defaultBuilder.create<cf::AssertOp>(loc, isValid, "invalid analog layer-id");
  defaultBuilder.create<cf::BranchOp>(loc, exitBlock);
}

// Builds the public `analog_run_layer` dispatcher that switches on the
// runtime layer id and writes the matching outlined layer helper result
// into the provided output buffer.
static func::FuncOp createLayerDispatcher(ModuleOp module,
                                          ArrayRef<LayerCallInfo> layers,
                                          StringRef fnName,
                                          const LayerSignature &signature,
                                          RankedTensorType dynResultTy) {
  auto [fn, exitBlock] = createLayerDispatcherSkeleton(
      module, fnName, signature.dynamicOperandTys, dynResultTy);
  Block *entry = &fn.getBody().front();
  Region &bodyRegion = fn.getBody();
  ValueRange entryArgs = entry->getArguments();
  SmallVector<Value> inputArgs(entryArgs.begin(),
                               entryArgs.begin() + signature.dynamicOperandTys.size());
  Value outBufferArg = entryArgs[signature.dynamicOperandTys.size()];
  Value layerIdArg = entryArgs.back();
  Location loc = fn.getLoc();

  OpBuilder regionBuilder = OpBuilder::atBlockEnd(entry);
  SmallVector<Block *> caseBlocks =
      createDispatcherCaseBlocks(bodyRegion, layers.size(), regionBuilder);
  Block *defaultBlock = regionBuilder.createBlock(&bodyRegion);

  OpBuilder entryBuilder = OpBuilder::atBlockEnd(entry);
  SmallVector<int32_t> caseValues;
  caseValues.reserve(layers.size());
  SmallVector<ValueRange> caseOperands(layers.size(), ValueRange{});
  for (const LayerCallInfo &info : layers)
    caseValues.push_back(static_cast<int32_t>(info.layerId));
  entryBuilder.create<cf::SwitchOp>(loc, layerIdArg, defaultBlock, ValueRange{},
                                    caseValues, caseBlocks, caseOperands);

  for (auto [info, caseBlock] : llvm::zip(layers, caseBlocks))
    emitDispatcherCaseBlock(caseBlock, exitBlock, info, inputArgs, outBufferArg,
                            loc);
  emitDispatcherDefaultBlock(defaultBlock, exitBlock, loc);

  OpBuilder exitBuilder = OpBuilder::atBlockEnd(exitBlock);
  exitBuilder.create<func::ReturnOp>(loc, ValueRange{});
  return fn;
}

} // namespace

llvm::StringRef DispatchLayersPass::getArgument() const {
  return "analog-dispatch-layers";
}

llvm::StringRef DispatchLayersPass::getDescription() const {
  return "Create analog_run_layer and rewrite layer calls to "
         "dispatch/wait layer hooks";
}

void DispatchLayersPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
}

void DispatchLayersPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp forward = module.lookupSymbol<func::FuncOp>(kForwardFnName);
  if (!forward || forward.isExternal())
    return;

  SmallVector<LayerCallInfo> layers;
  if (failed(collectLayerCalls(forward, layers))) {
    signalPassFailure();
    return;
  }
  if (layers.empty())
    return;

  eraseSymbolsWithPrefix(module, kInvokeLayerPrefix);

  SmallVector<SmallVector<LayerCallInfo>> groups;
  SmallVector<LayerSignature> groupSignatures;
  for (LayerCallInfo &info : layers) {
    LayerSignature signature = buildLayerSignature(info);
    if (!signature.isSupportedForCurrentDispatcherAbi()) {
      info.call.emitError("unsupported layer signature for current dispatcher ABI");
      signalPassFailure();
      return;
    }

    bool inserted = false;
    for (auto [groupSig, group] : llvm::zip(groupSignatures, groups)) {
      if (groupSig.dynamicOperandTys == signature.dynamicOperandTys &&
          groupSig.dynamicResultTys == signature.dynamicResultTys) {
        group.push_back(info);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      groupSignatures.push_back(signature);
      groups.push_back(SmallVector<LayerCallInfo>{info});
    }
  }

  auto processGroup = [&](MutableArrayRef<LayerCallInfo> group) -> LogicalResult {
    if (group.empty())
      return success();

    for (LayerCallInfo &info : group) {
      if (!info.resultTy.hasStaticShape()) {
        info.call.emitError(
            "by-reference layer runtime ABI currently requires static result shapes");
        return failure();
      }
    }

    auto maybeSignature = computeUnifiedLayerSignature(group);
    if (failed(maybeSignature))
      return failure();
    LayerSignature signature = *maybeSignature;
    if (!signature.isSupportedForCurrentDispatcherAbi()) {
      forward.emitError("unsupported layer signature for current dispatcher ABI");
      return failure();
    }
    RankedTensorType dynResultTy = signature.dynamicResultTys.front();

    FailureOr<LayerRuntimeHooks> maybeHooks =
        getOrCreateRuntimeHooks(module, signature);
    if (failed(maybeHooks)) {
      forward.emitError("unsupported tensor rank for layer dispatch hooks");
      return failure();
    }

    FailureOr<std::string> maybeDispatcherName =
        getDispatcherNameForSignature(signature);
    if (failed(maybeDispatcherName)) {
      forward.emitError("unsupported tensor rank for layer dispatcher");
      return failure();
    }

    createLayerDispatcher(module, group, *maybeDispatcherName, signature,
                          dynResultTy);
    return rewriteLayerCallSites(group, *maybeHooks, signature);
  };

  for (SmallVector<LayerCallInfo> &group : groups) {
    if (failed(processGroup(MutableArrayRef<LayerCallInfo>(group)))) {
      signalPassFailure();
      return;
    }
  }
}

std::unique_ptr<mlir::Pass> createDispatchLayersPass() {
  return std::make_unique<DispatchLayersPass>();
}

} // namespace analog
} // namespace mlir
