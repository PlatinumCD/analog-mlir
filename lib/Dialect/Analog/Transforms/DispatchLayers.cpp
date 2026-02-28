#include "analog-mlir/Dialect/Analog/Transforms/DispatchLayers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/SmallVector.h"

#include <functional>

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

struct LayerCallInfo {
  func::CallOp call;
  int64_t layerId;
  FlatSymbolRefAttr callee;
  RankedTensorType inputTy;
  RankedTensorType resultTy;
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

static RankedTensorType makeDynamicLike(RankedTensorType ty) {
  SmallVector<int64_t> shape(ty.getRank(), ShapedType::kDynamic);
  return RankedTensorType::get(shape, ty.getElementType());
}

static func::FuncOp createLayerDispatcher(ModuleOp module,
                                          ArrayRef<LayerCallInfo> layers,
                                          RankedTensorType dynInputTy,
                                          RankedTensorType dynResultTy) {
  eraseSymbolIfPresent(module, "analog_run_layer");

  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(&module.getBodyRegion().front());

  auto i32Ty = b.getI32Type();
  auto fnTy = b.getFunctionType(TypeRange{dynInputTy, i32Ty},
                                TypeRange{dynResultTy});
  auto fn = b.create<func::FuncOp>(module.getLoc(), "analog_run_layer", fnTy);
  fn.setPublic();

  Block *entry = fn.addEntryBlock();
  OpBuilder body = OpBuilder::atBlockEnd(entry);
  Value inputArg = entry->getArgument(0);
  Value layerIdArg = entry->getArgument(1);
  Location loc = fn.getLoc();

  auto emitCallAsDyn = [&](OpBuilder &builder,
                           const LayerCallInfo &info) -> Value {
    Value typedInput =
        builder.create<tensor::CastOp>(loc, info.inputTy, inputArg);
    auto call = builder.create<func::CallOp>(loc, info.callee,
                                             TypeRange{info.resultTy},
                                             ValueRange{typedInput});
    call->setAttr("layer-id", builder.getI64IntegerAttr(info.layerId));
    return builder.create<tensor::CastOp>(loc, dynResultTy, call.getResult(0));
  };

  std::function<Value(OpBuilder &, size_t)> buildChain;
  buildChain = [&](OpBuilder &builder, size_t pos) -> Value {
    const LayerCallInfo &info = layers[pos];
    Value idConst = builder.create<arith::ConstantIntOp>(loc, info.layerId, 32);
    Value isMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  layerIdArg, idConst);
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{dynResultTy}, isMatch,
                                          /*withElseRegion=*/true);

    {
      OpBuilder thenBuilder = OpBuilder::atBlockEnd(&ifOp.getThenRegion().front());
      Value thenValue = emitCallAsDyn(thenBuilder, info);
      thenBuilder.create<scf::YieldOp>(loc, ValueRange{thenValue});
    }

    {
      OpBuilder elseBuilder = OpBuilder::atBlockEnd(&ifOp.getElseRegion().front());
      Value elseValue;
      if (pos + 1 == layers.size()) {
        // Fallback for unknown layer-id: execute the last known layer.
        elseValue = emitCallAsDyn(elseBuilder, info);
      } else {
        elseValue = buildChain(elseBuilder, pos + 1);
      }
      elseBuilder.create<scf::YieldOp>(loc, ValueRange{elseValue});
    }

    return ifOp.getResult(0);
  };

  Value result = buildChain(body, 0);
  body.create<func::ReturnOp>(loc, ValueRange{result});
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
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
}

void DispatchLayersPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp forward = module.lookupSymbol<func::FuncOp>("forward");
  if (!forward || forward.isExternal()) {
    return;
  }

  SmallVector<LayerCallInfo> layers;
  bool hadError = false;
  forward.walk([&](func::CallOp call) {
    if (hadError) {
      return;
    }

    auto idAttr = call->getAttrOfType<IntegerAttr>("layer-id");
    if (!idAttr) {
      return;
    }

    FlatSymbolRefAttr callee = call.getCalleeAttr();
    if (!callee) {
      call.emitError("expected direct func.call for layer-id call");
      hadError = true;
      return;
    }

    StringRef calleeName = callee.getValue();
    if (calleeName == "analog_dispatch_layer" ||
        calleeName == "analog_run_layer" ||
        calleeName == "analog_wait_layers") {
      return;
    }

    if (call.getNumOperands() != 1 || call.getNumResults() != 1) {
      call.emitError("expected layer-id call to have 1 operand and 1 result");
      hadError = true;
      return;
    }

    auto inputTy = dyn_cast<RankedTensorType>(call.getOperand(0).getType());
    auto resultTy = dyn_cast<RankedTensorType>(call.getResult(0).getType());
    if (!inputTy || !resultTy) {
      call.emitError("expected tensor operand/result types on layer-id call");
      hadError = true;
      return;
    }
    if (inputTy.getRank() != resultTy.getRank()) {
      call.emitError("expected matching tensor ranks for layer dispatcher ABI");
      hadError = true;
      return;
    }
    if (inputTy.getElementType() != resultTy.getElementType()) {
      call.emitError("expected matching element type for layer dispatcher ABI");
      hadError = true;
      return;
    }

    int64_t layerId = idAttr.getValue().getSExtValue();
    layers.push_back(LayerCallInfo{call, layerId, callee, inputTy, resultTy});
  });

  if (hadError) {
    signalPassFailure();
    return;
  }

  if (layers.empty()) {
    return;
  }

  RankedTensorType dynInputTy = makeDynamicLike(layers.front().inputTy);
  RankedTensorType dynResultTy = makeDynamicLike(layers.front().resultTy);
  for (LayerCallInfo &info : layers) {
    if (makeDynamicLike(info.inputTy) != dynInputTy ||
        makeDynamicLike(info.resultTy) != dynResultTy) {
      info.call.emitError("incompatible layer signatures for unified "
                          "analog_dispatch_layer ABI");
      signalPassFailure();
      return;
    }
  }

  OpBuilder moduleBuilder(module.getBodyRegion());
  auto i32Ty = moduleBuilder.getI32Type();
  auto dispatchTy =
      moduleBuilder.getFunctionType(TypeRange{dynInputTy, i32Ty}, TypeRange{});
  auto waitTy = moduleBuilder.getFunctionType(TypeRange{}, TypeRange{dynResultTy});

  func::FuncOp dispatchDecl = getOrCreateExternDecl(module, "analog_dispatch_layer",
                                                    dispatchTy);
  func::FuncOp waitDecl = getOrCreateExternDecl(module, "analog_wait_layers",
                                                waitTy);
  dispatchDecl->setAttr("analog-shim-required", moduleBuilder.getUnitAttr());
  waitDecl->setAttr("analog-shim-required", moduleBuilder.getUnitAttr());

  eraseSymbolsWithPrefix(module, "analog_invoke_layer_");
  createLayerDispatcher(module, layers, dynInputTy, dynResultTy);
  for (LayerCallInfo &info : layers) {
    if (!info.call || !info.call->getBlock()) {
      continue;
    }

    OpBuilder b(info.call);
    Location loc = info.call.getLoc();

    Value idConst = b.create<arith::ConstantIntOp>(loc, info.layerId, 32);
    Value dynInput =
        b.create<tensor::CastOp>(loc, dynInputTy, info.call.getOperand(0));
    auto dispatch = b.create<func::CallOp>(loc, dispatchDecl.getSymName(),
                                           TypeRange{},
                                           ValueRange{dynInput, idConst});
    dispatch->setAttr("layer-id", b.getI64IntegerAttr(info.layerId));
    auto wait =
        b.create<func::CallOp>(loc, waitDecl.getSymName(),
                               TypeRange{dynResultTy}, ValueRange{});
    wait->setAttr("layer-id", b.getI64IntegerAttr(info.layerId));
    Value typedResult =
        b.create<tensor::CastOp>(loc, info.resultTy, wait.getResult(0));

    info.call.getResult(0).replaceAllUsesWith(typedResult);
    info.call.erase();
  }
}

std::unique_ptr<mlir::Pass> createDispatchLayersPass() {
  return std::make_unique<DispatchLayersPass>();
}

} // namespace analog
} // namespace mlir
