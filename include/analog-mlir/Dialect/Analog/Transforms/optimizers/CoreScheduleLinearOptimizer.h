#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_CORESCHEDULELINEAROPTIMIZER_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_CORESCHEDULELINEAROPTIMIZER_H

#include "analog-mlir/Dialect/Analog/Transforms/optimizers/optimizerUtils.h"

namespace mlir {
namespace analog {

// Installs the core linear scheduling optimizer into the symbolic graph
// optimizer pipeline.
void registerCoreScheduleLinearOptimizer(
    SymbolTaskGraphOptimizers &optimizers);

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_CORESCHEDULELINEAROPTIMIZER_H
