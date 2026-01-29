#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OpAsmSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "analog-types"

using namespace mlir;
using namespace mlir::analog;

#define GET_TYPEDEF_CLASSES
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.cpp.inc"

void AnalogDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// WeightsType - verify
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::analog::WeightsType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType,
    mlir::StringAttr layout,
    int64_t stride) {

  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !mlir::isa<mlir::FloatType>(elementType))
    return emitError() << "elementType must be a float type";

  if (!layout)
    return emitError() << "layout must be a string attribute";

  if (shape.size() != 2)
    return emitError() << "expected rank-2 weights, got rank " << shape.size();

  if (stride <= 0)
    return emitError() << "stride must be > 0";

  if (stride < shape[1])
    return emitError() << "stride must be >= inner dimension (" << shape[1] << ")";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// WeightsType - parseShapeAndElt
//===----------------------------------------------------------------------===//

mlir::ParseResult
mlir::analog::WeightsType::parseShapeAndElt(
    mlir::AsmParser &parser,
    llvm::SmallVector<int64_t> &shape,
    mlir::Type &elementType) {

  shape.clear();

  while (true) {
    int64_t dim;
    mlir::OptionalParseResult maybeInt =
        parser.parseOptionalInteger(dim);
    if (!maybeInt.has_value())
      break;

    shape.push_back(dim);

    if (parser.parseXInDimensionList())
      return mlir::failure();
  }

  if (parser.parseType(elementType))
    return mlir::failure();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// WeightsType - printShapeAndElt
//===----------------------------------------------------------------------===//

void
mlir::analog::WeightsType::printShapeAndElt(
    mlir::AsmPrinter &printer,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {

  for (auto d : shape)
    printer << d << 'x';

  printer.printType(elementType);
}

//===----------------------------------------------------------------------===//
// WeightsType - parse
//===----------------------------------------------------------------------===//

mlir::Type
mlir::analog::WeightsType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  StringAttr layout;
  int64_t stride;

  // Parse 8x8xf32
  if (parseShapeAndElt(parser, shape, elementType))
    return Type();

  // , layout="T"
  if (parser.parseComma())
    return Type();

  if (parser.parseKeyword("layout"))
    return Type();

  if (parser.parseEqual())
    return Type();

  if (parser.parseAttribute(layout))
    return Type();

  // , stride=8
  if (parser.parseComma())
    return Type();

  if (parser.parseKeyword("stride"))
    return Type();

  if (parser.parseEqual())
    return Type();

  if (parser.parseInteger(stride))
    return Type();

  return get(parser.getContext(), shape, elementType, layout, stride);
}

//===----------------------------------------------------------------------===//
// WeightsType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::WeightsType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
  printer << ", layout=";
  printer.printAttribute(getLayout());
  printer << ", stride=" << getStride();
}


