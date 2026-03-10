#include "analog-mlir/Dialect/Analog/Conversion/FinalizeGolemIntrinsics.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

constexpr StringLiteral kWrapperSetName = "golem_analog_mvm_set";
constexpr StringLiteral kWrapperLoadName = "golem_analog_mvm_load";
constexpr StringLiteral kWrapperStoreName = "golem_analog_mvm_store";
constexpr StringLiteral kWrapperComputeName = "golem_analog_mvm_compute";
constexpr StringLiteral kIntrinsicSetName = "llvm.riscv.golem.analog.mvm.set";
constexpr StringLiteral kIntrinsicLoadName = "llvm.riscv.golem.analog.mvm.load";
constexpr StringLiteral kIntrinsicStoreName = "llvm.riscv.golem.analog.mvm.store";
constexpr StringLiteral kIntrinsicComputeName = "llvm.riscv.golem.analog.mvm";

struct IntrinsicRewriteRule {
  StringRef sourceName;
  StringRef targetName;
  unsigned minOperands;
  bool passesPointer;
};


// Creates or reuses a private LLVM declaration for one final intrinsic so
// rewritten calls always have a valid symbol target.
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
    if (!basePtr || !offset) {
      return basePtr;
    }

    if (!llvm::isa<IntegerType>(offset.getType())) {
      return basePtr;
    }

    OpBuilder b(call);
    auto elemTy = Float32Type::get(call.getContext());
    return b.create<LLVM::GEPOp>(call.getLoc(), basePtr.getType(), elemTy,
                                 basePtr, ValueRange{offset});
  };

  // For exploded memref calls, operands are typically:
  //   [allocated_ptr, aligned_ptr, offset, sizes..., strides..., array_id]
  // and the first logical element base pointer is the aligned pointer.
  if (call.getNumOperands() >= 3 &&
      llvm::isa<LLVM::LLVMPointerType>(call.getOperand(1).getType())) {
    return buildPtrWithOffset(call.getOperand(1), call.getOperand(2));
  }

  Value ptr = call.getOperand(0);
  if (auto extract = ptr.getDefiningOp<LLVM::ExtractValueOp>()) {
    if (extract.getPosition().size() == 1 &&
        (extract.getPosition()[0] == 0 || extract.getPosition()[0] == 1)) {
      auto structTy = llvm::dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
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


// Declares the final LLVM golem intrinsics that wrapper calls will be
// rewritten to during this pass.
static void ensureIntrinsicDecls(ModuleOp module, MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  getOrCreateLLVMFunc(
      module, kIntrinsicSetName,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, kIntrinsicLoadName,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, kIntrinsicStoreName,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, kIntrinsicComputeName,
      LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false));
}


// Returns the rewrite rule for a recognized wrapper call and leaves other
// LLVM calls untouched.
static const IntrinsicRewriteRule *findIntrinsicRewriteRule(StringRef callee) {
  static constexpr IntrinsicRewriteRule kRules[] = {
      {kWrapperSetName, kIntrinsicSetName, 2, true},
      {kWrapperLoadName, kIntrinsicLoadName, 2, true},
      {kWrapperStoreName, kIntrinsicStoreName, 2, true},
      {kWrapperComputeName, kIntrinsicComputeName, 1, false},
  };

  for (const IntrinsicRewriteRule &rule : kRules) {
    if (callee == rule.sourceName)
      return &rule;
  }

  return nullptr;
}


// Rewrites one wrapper call to its final intrinsic and forwards the data
// pointer when the intrinsic operates on a buffer.
static bool rewriteWrapperCall(LLVM::CallOp call, MLIRContext *ctx) {
  auto calleeAttr = call.getCalleeAttr();
  if (!calleeAttr)
    return false;

  const IntrinsicRewriteRule *rule =
      findIntrinsicRewriteRule(calleeAttr.getValue());
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


// Removes wrapper declarations that become dead after all call sites have
// been redirected to final intrinsics.
static void eraseUnusedWrapperDecls(ModuleOp module) {
  for (StringRef oldName : {kWrapperSetName, kWrapperLoadName, kWrapperStoreName,
                            kWrapperComputeName}) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(oldName)) {
      if (fn.use_empty())
        fn.erase();
    }
  }
}

} // namespace


// Returns the command-line name used to invoke this finalization pass in
// pipelines and tests.
llvm::StringRef FinalizeGolemIntrinsicsPass::getArgument() const {
  return "analog-finalize-golem-intrinsics";
}


// Describes that this pass rewrites wrapper calls into the final LLVM
// golem intrinsic entrypoints.
llvm::StringRef FinalizeGolemIntrinsicsPass::getDescription() const {
  return "Rewrite golem wrapper calls into final LLVM RISC-V golem intrinsic calls";
}


// Registers the LLVM dialect because this pass inspects and creates LLVM
// declarations and call operations.
void FinalizeGolemIntrinsicsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}


// Rewrites golem wrapper calls to their final intrinsic targets, then
// erases any obsolete wrapper declarations left unused.
void FinalizeGolemIntrinsicsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  ensureIntrinsicDecls(module, ctx);

  module.walk([&](LLVM::CallOp call) {
    rewriteWrapperCall(call, ctx);
  });

  eraseUnusedWrapperDecls(module);
}


// Creates the pass instance used by registration and lowering pipelines
// that target final golem intrinsics.
std::unique_ptr<mlir::Pass> createFinalizeGolemIntrinsicsPass() {
  return std::make_unique<FinalizeGolemIntrinsicsPass>();
}

} // namespace analog
} // namespace mlir
