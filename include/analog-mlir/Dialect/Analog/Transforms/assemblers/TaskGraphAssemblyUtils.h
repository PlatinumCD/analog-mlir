#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ASSEMBLERS_TASKGRAPHASSEMBLYUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ASSEMBLERS_TASKGRAPHASSEMBLYUTILS_H

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace analog {
namespace assembler_utils {

inline constexpr llvm::StringLiteral kTaskGraphFuncAttrName(
    "analog.assembly.task_graph_func");
inline constexpr llvm::StringLiteral kForwardInputIndexAttrName(
    "analog.assembly.forward_input_index");
inline constexpr llvm::StringLiteral kForwardOutputIndexAttrName(
    "analog.assembly.forward_output_index");
inline constexpr llvm::StringLiteral kForwardCallIndexAttrName(
    "analog.assembly.forward_call_index");
inline constexpr llvm::StringLiteral kForwardResultIndexAttrName(
    "analog.assembly.forward_result_index");
inline constexpr llvm::StringLiteral kPersistentSymbolAttrName(
    "analog.assembly.persistent_symbol");

inline void setGeneratedTaskGraphFunc(mlir::func::FuncOp forward,
                                      mlir::func::FuncOp taskGraphFunc) {
  forward->setAttr(
      kTaskGraphFuncAttrName,
      mlir::FlatSymbolRefAttr::get(taskGraphFunc.getContext(),
                                   taskGraphFunc.getSymName()));
}

inline mlir::func::FuncOp
lookupGeneratedTaskGraphFunc(mlir::ModuleOp module, mlir::func::FuncOp forward) {
  auto taskGraphAttr =
      forward->getAttrOfType<mlir::FlatSymbolRefAttr>(kTaskGraphFuncAttrName);
  if (!taskGraphAttr)
    return {};

  return module.lookupSymbol<mlir::func::FuncOp>(taskGraphAttr.getValue());
}

inline mlir::analog::TaskGraphCreateOp
findTaskGraphCreateOp(mlir::func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc || taskGraphFunc.getBody().empty())
    return {};

  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    if (auto graph = llvm::dyn_cast<mlir::analog::TaskGraphCreateOp>(&op))
      return graph;
  }

  return {};
}

inline void clearAssemblyAttrs(mlir::func::FuncOp forward,
                               mlir::func::FuncOp taskGraphFunc) {
  forward->removeAttr(kTaskGraphFuncAttrName);
  if (!taskGraphFunc || taskGraphFunc.getBody().empty())
    return;

  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    op.removeAttr(kForwardInputIndexAttrName);
    op.removeAttr(kForwardOutputIndexAttrName);
    op.removeAttr(kForwardCallIndexAttrName);
    op.removeAttr(kForwardResultIndexAttrName);
    op.removeAttr(kPersistentSymbolAttrName);
  }
}

} // namespace assembler_utils
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ASSEMBLERS_TASKGRAPHASSEMBLYUTILS_H
