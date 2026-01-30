#include "analog-mlir/Dialect/Analog/IR/AnalogDialect.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"

#define DEBUG_TYPE "analog-ops"

using namespace mlir;
using namespace mlir::analog;

#define GET_OP_CLASSES
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.cpp.inc"

void AnalogDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.cpp.inc"
        >();
}

