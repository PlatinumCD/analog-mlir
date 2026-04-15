#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_FRONTLOADWEIGHTTASKSOPTIMIZER_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_FRONTLOADWEIGHTTASKSOPTIMIZER_H

#include "analog-mlir/Dialect/Analog/Transforms/optimizers/optimizerUtils.h"

namespace mlir {
namespace analog {

// Installs the optimizer that schedules weight initialization tasks first.
void registerFrontloadWeightTasksOptimizer(
    SymbolTaskGraphOptimizers &optimizers);

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_FRONTLOADWEIGHTTASKSOPTIMIZER_H
