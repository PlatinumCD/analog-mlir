#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_OPTIMIZERUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_OPTIMIZERUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <vector>

namespace mlir {
namespace analog {

// Defines the extension point for passes that refine a generated symbolic task
// graph function after it has been assembled.
class SymbolTaskGraphOptimizer {
public:
  // Allows optimizer implementations to be owned through the base interface.
  virtual ~SymbolTaskGraphOptimizer() = default;

  // Returns the optimizer name used when reporting task graph failures.
  virtual StringRef getName() const = 0;

  // Mutates the generated task graph function or emits an error on invalid IR.
  virtual LogicalResult optimize(func::FuncOp taskGraphFunc) const = 0;
};

// Owns symbolic task graph optimizers in the order they should run.
using SymbolTaskGraphOptimizers =
    std::vector<std::unique_ptr<SymbolTaskGraphOptimizer>>;

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OPTIMIZERS_OPTIMIZERUTILS_H
