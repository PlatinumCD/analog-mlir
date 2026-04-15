#include "analog-mlir/Dialect/Analog/Transforms/AssembleTaskGraph.h"
#include "analog-mlir/Dialect/Analog/Transforms/assemblers/TaskGraphAssemblyStep.h"
#include "analog-mlir/Dialect/Analog/Transforms/optimizers/CoreScheduleLinearOptimizer.h"
#include "analog-mlir/Dialect/Analog/Transforms/optimizers/FrontloadWeightTasksOptimizer.h"

#include "analog-mlir/Dialect/Analog/Transforms/assemblers/TaskGraphAssemblyUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

mlir::LogicalResult assembleTaskGraph(mlir::ModuleOp module,
                                      mlir::func::FuncOp forward,
                                      mlir::func::FuncOp &taskGraphFunc) {
  mlir::analog::TaskGraphAssemblySteps steps;
  mlir::analog::registerForwardTaskOutliner(steps);
  mlir::analog::registerTaskGraphGeneratorAssembler(steps);
  mlir::analog::registerTaskGraphResourceAssembler(steps);
  mlir::analog::registerTaskGraphTaskAssembler(steps);

  taskGraphFunc = {};
  for (const auto &step : steps) {
    if (failed(step->assemble(module, forward))) {
      forward.emitError("failed to apply task graph assembly step '")
          << step->getName() << "'";
      return mlir::failure();
    }

    if (!taskGraphFunc) {
      taskGraphFunc =
          mlir::analog::assembler_utils::lookupGeneratedTaskGraphFunc(module,
                                                                      forward);
    }
  }

  if (!taskGraphFunc) {
    forward.emitError(
        "expected task graph assembly pipeline to create a generator function");
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult assembleExecutablePlan(mlir::ModuleOp module,
                                           mlir::func::FuncOp forward) {
  mlir::analog::TaskGraphAssemblySteps steps;
  mlir::analog::registerTaskGraphExecutionPlanAssembler(steps);

  for (const auto &step : steps) {
    if (failed(step->assemble(module, forward))) {
      forward.emitError("failed to apply task graph assembly step '")
          << step->getName() << "'";
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult optimizeSymbolTaskGraph(mlir::func::FuncOp taskGraphFunc) {
  mlir::analog::SymbolTaskGraphOptimizers optimizers;
  mlir::analog::registerFrontloadWeightTasksOptimizer(optimizers);
  mlir::analog::registerCoreScheduleLinearOptimizer(optimizers);

  for (const auto &optimizer : optimizers) {
    if (failed(optimizer->optimize(taskGraphFunc))) {
      taskGraphFunc.emitError("failed to apply symbol task graph optimizer '")
          << optimizer->getName() << "'";
      return mlir::failure();
    }
  }

  return mlir::success();
}

} // namespace

namespace mlir {
namespace analog {

void AssembleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    mlir::func::FuncOp taskGraphFunc;
    if (failed(assembleTaskGraph(module, func, taskGraphFunc)) ||
        failed(optimizeSymbolTaskGraph(taskGraphFunc)) ||
        failed(assembleExecutablePlan(module, func))) {
      signalPassFailure();
      return;
    }
  }
}

void registerAssembleTaskGraphPass() {
  PassRegistration<AssembleTaskGraphPass>();
}

} // namespace analog
} // namespace mlir
