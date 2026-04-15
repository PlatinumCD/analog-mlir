#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ASSEMBLERS_TASKGRAPHASSEMBLYSTEP_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ASSEMBLERS_TASKGRAPHASSEMBLYSTEP_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <vector>

namespace mlir {
namespace analog {

class TaskGraphAssemblyStep {
public:
  virtual ~TaskGraphAssemblyStep() = default;

  virtual StringRef getName() const = 0;

  virtual LogicalResult assemble(ModuleOp module, func::FuncOp forward) const = 0;
};

using TaskGraphAssemblySteps =
    std::vector<std::unique_ptr<TaskGraphAssemblyStep>>;

void registerForwardTaskOutliner(TaskGraphAssemblySteps &steps);
void registerTaskGraphGeneratorAssembler(TaskGraphAssemblySteps &steps);
void registerTaskGraphResourceAssembler(TaskGraphAssemblySteps &steps);
void registerTaskGraphTaskAssembler(TaskGraphAssemblySteps &steps);
void registerTaskGraphExecutionPlanAssembler(TaskGraphAssemblySteps &steps);

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ASSEMBLERS_TASKGRAPHASSEMBLYSTEP_H
