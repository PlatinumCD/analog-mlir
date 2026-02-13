#include "analog-mlir/Dialect/Analog/Conversion/FinalizeGolemIntrinsics.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

static LLVM::LLVMFuncOp getOrCreateLLVMFunc(ModuleOp module, StringRef name,
                                            LLVM::LLVMFunctionType type,
                                            bool makePrivate = true) {

  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return fn;
  }

  OpBuilder b(module.getBodyRegion());
  auto fn = b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
  if (makePrivate) {
    fn.setPrivate();
  }
  return fn;
}

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
  //   [allocated_ptr, aligned_ptr, offset, sizes..., strides..., tile_id]
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

} // namespace

llvm::StringRef FinalizeGolemIntrinsicsPass::getArgument() const {
  return "analog-finalize-golem-intrinsics";
}

llvm::StringRef FinalizeGolemIntrinsicsPass::getDescription() const {
  return "Rewrite golem wrapper calls into final LLVM RISC-V golem intrinsic calls";
}

void FinalizeGolemIntrinsicsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void FinalizeGolemIntrinsicsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);

  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm.set",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrTy, i32Ty},
                                  false));
  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm.load",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrTy, i32Ty},
                                  false));
  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm.store",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrTy, i32Ty},
                                  false));
  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {i32Ty}, false));
  auto tileIdPassthrough = getOrCreateLLVMFunc(
      module, "golem_analog_tileid_passthrough",
      LLVM::LLVMFunctionType::get(i32Ty, {i32Ty}, false),
      /*makePrivate=*/false);

  module.walk([&](LLVM::CallOp call) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr) {
      return;
    }

    StringRef callee = calleeAttr.getValue();

    if (callee == "golem_analog_mvm_set" || 
        callee == "golem_analog_mvm_load" ||
        callee == "golem_analog_mvm_store") {

      if (call.getNumOperands() < 2) {
        return;
      }

      Value ptr = getDataPtrOperand(call);
      Value tileId = call.getOperand(call.getNumOperands() - 1);
      OpBuilder b(call);
      tileId = b
                   .create<LLVM::CallOp>(
                       call.getLoc(), TypeRange{i32Ty},
                       SymbolRefAttr::get(ctx, tileIdPassthrough.getName()),
                       SmallVector<Value>{tileId})
                   .getResult();
      StringRef dst = callee == "golem_analog_mvm_set"
                          ? "llvm.riscv.golem.analog.mvm.set"
                          : (callee == "golem_analog_mvm_load"
                                 ? "llvm.riscv.golem.analog.mvm.load"
                                 : "llvm.riscv.golem.analog.mvm.store");

      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, dst),
          SmallVector<Value>{ptr, tileId}
      );

      call.erase();
      return;
    }

    if (callee == "golem_analog_mvm_compute") {

      if (call.getNumOperands() < 1) {
        return;
      }

      OpBuilder b(call);
      Value tileId = call.getOperand(call.getNumOperands() - 1);
      tileId = b
                   .create<LLVM::CallOp>(
                       call.getLoc(), TypeRange{i32Ty},
                       SymbolRefAttr::get(ctx, tileIdPassthrough.getName()),
                       SmallVector<Value>{tileId})
                   .getResult();

      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, "llvm.riscv.golem.analog.mvm"),
          SmallVector<Value>{tileId}
      );

      call.erase();
      return;
    }
  });

  for (StringRef oldName : {"golem_analog_mvm_set",
                            "golem_analog_mvm_load",
                            "golem_analog_mvm_store",
                            "golem_analog_mvm_compute"}) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(oldName)) {
      if (fn.use_empty()) {
        fn.erase();
      }
    }
  }
}

std::unique_ptr<mlir::Pass> createFinalizeGolemIntrinsicsPass() {
  return std::make_unique<FinalizeGolemIntrinsicsPass>();
}

} // namespace analog
} // namespace mlir
