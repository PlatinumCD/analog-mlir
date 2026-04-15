#ifndef ANALOG_MLIR_DIALECT_ANALOG_IR_ANALOGOPS_H
#define ANALOG_MLIR_DIALECT_ANALOG_IR_ANALOGOPS_H

#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// Analog Ops
//===----------------------------------------------------------------------===//

// Exposes the TableGen-generated operation classes for analog storage,
// placement, task-graph, and array-execution IR.
#define GET_OP_CLASSES
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h.inc"

#endif // ANALOG_MLIR_DIALECT_ANALOG_IR_ANALOGOPS_H
