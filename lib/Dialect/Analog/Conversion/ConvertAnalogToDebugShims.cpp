#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToDebugShims.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

constexpr StringLiteral kDebugSetName = "golem_debug_mvm_set";
constexpr StringLiteral kDebugLoadName = "golem_debug_mvm_load";
constexpr StringLiteral kDebugStoreName = "golem_debug_mvm_store";
constexpr StringLiteral kDebugComputeName = "golem_debug_mvm_compute";

struct ShimRewriteRule {
  StringRef sourceName;
  StringRef targetName;
  unsigned minOperands;
  bool passesPointer;
};


// Creates or reuses a private LLVM declaration for one debug shim so
// rewritten backend calls always have a valid callee.
static LLVM::LLVMFuncOp getOrCreateLLVMFunc(ModuleOp module, StringRef name,
                                            LLVM::LLVMFunctionType type) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return fn;
  }

  OpBuilder b(module.getBodyRegion());
  auto fn = b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}


// Recovers the logical data pointer from either exploded memref operands
// or extracted memref-descriptor pieces.
static Value getDataPtrOperand(LLVM::CallOp call) {
  auto buildPtrWithOffset = [&](Value basePtr, Value offset) -> Value {
    if (!basePtr || !offset)
      return basePtr;
    if (!llvm::isa<IntegerType>(offset.getType()))
      return basePtr;
    OpBuilder b(call);
    auto elemTy = Float32Type::get(call.getContext());
    return b.create<LLVM::GEPOp>(call.getLoc(), basePtr.getType(), elemTy,
                                 basePtr, ValueRange{offset});
  };

  // For exploded memref calls, operands are typically:
  //   [allocated_ptr, aligned_ptr, offset, sizes..., strides..., array_id]
  // and the first logical element base pointer is the aligned pointer.
  if (call.getNumOperands() >= 3 &&
      llvm::isa<LLVM::LLVMPointerType>(call.getOperand(1).getType()))
    return buildPtrWithOffset(call.getOperand(1), call.getOperand(2));

  Value ptr = call.getOperand(0);
  if (auto extract = ptr.getDefiningOp<LLVM::ExtractValueOp>()) {
    if (extract.getPosition().size() == 1 &&
        (extract.getPosition()[0] == 0 || extract.getPosition()[0] == 1)) {
      auto structTy =
          llvm::dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
      if (structTy && structTy.getBody().size() >= 3 &&
          llvm::isa<LLVM::LLVMPointerType>(structTy.getBody()[1])) {
        OpBuilder b(call);
        Value aligned = extract.getPosition()[0] == 1
                            ? ptr
                            : b.create<LLVM::ExtractValueOp>(
                                  call.getLoc(), extract.getContainer(),
                                  ArrayRef<int64_t>{1});
        Value offset = b.create<LLVM::ExtractValueOp>(
            call.getLoc(), extract.getContainer(), ArrayRef<int64_t>{2});
        return buildPtrWithOffset(aligned, offset);
      }
    }
  }

  return ptr;
}


// Declares all debug shim entrypoints that this pass may rewrite calls to
// so later replacements can refer to stable symbols.
static void ensureDebugShimDecls(ModuleOp module, MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  getOrCreateLLVMFunc(
      module, kDebugSetName,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, kDebugLoadName,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, kDebugStoreName,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, kDebugComputeName,
      LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false));
}


// Returns the debug-shim rewrite rule for a recognized backend call name
// and leaves unrelated calls untouched.
static const ShimRewriteRule *findShimRewriteRule(StringRef callee) {
  static constexpr ShimRewriteRule kRules[] = {
      {"golem_analog_mvm_set", kDebugSetName, 2, true},
      {"llvm.riscv.golem.analog.mvm.set", kDebugSetName, 2, true},
      {"golem_analog_mvm_load", kDebugLoadName, 2, true},
      {"llvm.riscv.golem.analog.mvm.load", kDebugLoadName, 2, true},
      {"golem_analog_mvm_store", kDebugStoreName, 2, true},
      {"llvm.riscv.golem.analog.mvm.store", kDebugStoreName, 2, true},
      {"golem_analog_mvm_compute", kDebugComputeName, 1, false},
      {"llvm.riscv.golem.analog.mvm", kDebugComputeName, 1, false},
  };

  for (const ShimRewriteRule &rule : kRules) {
    if (callee == rule.sourceName)
      return &rule;
  }

  return nullptr;
}


// Rewrites one backend call to the matching debug shim and forwards the
// extracted data pointer when the intrinsic operates on a buffer.
static bool rewriteCallToDebugShim(LLVM::CallOp call, MLIRContext *ctx) {
  auto calleeAttr = call.getCalleeAttr();
  if (!calleeAttr)
    return false;

  const ShimRewriteRule *rule = findShimRewriteRule(calleeAttr.getValue());
  if (!rule || call.getNumOperands() < rule->minOperands)
    return false;

  SmallVector<Value> operands;
  if (rule->passesPointer)
    operands.push_back(getDataPtrOperand(call));
  operands.push_back(call.getOperand(call.getNumOperands() - 1));

  OpBuilder b(call);
  b.create<LLVM::CallOp>(call.getLoc(), TypeRange{},
                         SymbolRefAttr::get(ctx, rule->targetName), operands);
  call.erase();
  return true;
}


// Deletes obsolete backend declarations after all call sites have been
// redirected to their debug-shim equivalents.
static void eraseUnusedBackendDecls(ModuleOp module) {
  for (StringRef oldName : {
           "golem_analog_mvm_set",
           "golem_analog_mvm_load",
           "golem_analog_mvm_store",
           "golem_analog_mvm_compute",
           "llvm.riscv.golem.analog.mvm.set",
           "llvm.riscv.golem.analog.mvm.load",
           "llvm.riscv.golem.analog.mvm.store",
           "llvm.riscv.golem.analog.mvm",
       }) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(oldName)) {
      if (fn.use_empty())
        fn.erase();
    }
  }
}

} // namespace


// Returns the command-line pipeline name used to invoke this conversion
// pass from tests and tooling.
llvm::StringRef ConvertAnalogToDebugShimsPass::getArgument() const {
  return "analog-convert-to-debug-shims";
}


// Describes that this pass redirects backend calls to debug shim targets
// for simulation and instrumentation.
llvm::StringRef ConvertAnalogToDebugShimsPass::getDescription() const {
  return "Rewrite analog backend calls to debug shim call targets for simulation/instrumentation";
}


// Registers the LLVM dialect because the pass rewrites and creates LLVM
// call operations and declarations.
void ConvertAnalogToDebugShimsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}


// Rewrites recognized backend calls to debug shims, then removes any old
// backend declarations left without uses.
void ConvertAnalogToDebugShimsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  ensureDebugShimDecls(module, ctx);

  module.walk([&](LLVM::CallOp call) {
    rewriteCallToDebugShim(call, ctx);
  });

  eraseUnusedBackendDecls(module);
}


// Creates the pass instance used by registration and conversion pipelines
// that target the debug shim backend.
std::unique_ptr<mlir::Pass> createConvertAnalogToDebugShimsPass() {
  return std::make_unique<ConvertAnalogToDebugShimsPass>();
}

} // namespace analog
} // namespace mlir
