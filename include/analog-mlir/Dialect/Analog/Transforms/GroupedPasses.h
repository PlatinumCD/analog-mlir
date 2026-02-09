#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_GROUPED_PASSES_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_GROUPED_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace analog {

void registerMaterializePipeline();
void registerPartitionPipeline();
void registerPlacePipeline();

} // namespace analog
} // namespace mlir

#endif
