#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OpAsmSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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
    mlir::Type elementType) {

  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !mlir::isa<mlir::FloatType>(elementType))
    return emitError() << "elementType must be a float type";

  if (shape.size() != 2)
    return emitError() << "expected rank-2 weights, got rank " << shape.size();

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

  // Parse 8x8xf32
  if (parseShapeAndElt(parser, shape, elementType))
    return Type();

  return get(parser.getContext(), shape, elementType);
}

//===----------------------------------------------------------------------===//
// WeightsType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::WeightsType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
}




//===----------------------------------------------------------------------===//
// TileType - verify
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::analog::TileType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType,
    int64_t stride,
    int64_t base) {

  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !mlir::isa<mlir::FloatType>(elementType))
    return emitError() << "elementType must be a float type";

  if (shape.size() != 2)
    return emitError() << "expected rank-2 weights, got rank " << shape.size();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TileType - parseShapeAndElt
//===----------------------------------------------------------------------===//

mlir::ParseResult
mlir::analog::TileType::parseShapeAndElt(
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
// TileType - printShapeAndElt
//===----------------------------------------------------------------------===//

void
mlir::analog::TileType::printShapeAndElt(
    mlir::AsmPrinter &printer,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {

  for (auto d : shape)
    printer << d << 'x';

  printer.printType(elementType);
}

//===----------------------------------------------------------------------===//
// TileType - parse
//===----------------------------------------------------------------------===//

mlir::Type
mlir::analog::TileType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  uint64_t stride;
  uint64_t base;

  // Parse 8x8xf32
  if (parseShapeAndElt(parser, shape, elementType))
    return Type();

  // Expect: , stride = <int>
  if (parser.parseComma() ||
      parser.parseKeyword("stride") ||
      parser.parseEqual() ||
      parser.parseInteger(stride))
    return Type();

  // Expect: , base = <int>
  if (parser.parseComma() ||
      parser.parseKeyword("base") ||
      parser.parseEqual() ||
      parser.parseInteger(base))
    return Type();

  return get(parser.getContext(), shape, elementType, stride, base);
}

//===----------------------------------------------------------------------===//
// TileType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::TileType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
  printer << ", stride=" << getStride();
  printer << ", base=" << getBase();
}
