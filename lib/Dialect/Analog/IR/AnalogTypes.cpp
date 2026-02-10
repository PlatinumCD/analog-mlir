#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
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
// MatrixType - verify
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::analog::MatrixType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {

  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !mlir::isa<mlir::FloatType>(elementType))
    return emitError() << "elementType must be a float type";

  if (shape.size() != 2)
    return emitError() << "expected rank-2 matrix, got rank " << shape.size();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MatrixType - parseShapeAndElt
//===----------------------------------------------------------------------===//

mlir::ParseResult
mlir::analog::MatrixType::parseShapeAndElt(
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
// MatrixType - printShapeAndElt
//===----------------------------------------------------------------------===//

void
mlir::analog::MatrixType::printShapeAndElt(
    mlir::AsmPrinter &printer,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {

  for (auto d : shape)
    printer << d << 'x';

  printer.printType(elementType);
}

//===----------------------------------------------------------------------===//
// MatrixType - parse
//===----------------------------------------------------------------------===//

mlir::Type
mlir::analog::MatrixType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;

  // Parse 8x8xf32
  if (parseShapeAndElt(parser, shape, elementType))
    return Type();

  return get(parser.getContext(), shape, elementType);
}

//===----------------------------------------------------------------------===//
// MatrixType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::MatrixType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
}





//===----------------------------------------------------------------------===//
// VectorType - verify
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::analog::VectorType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {

  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !mlir::isa<mlir::FloatType>(elementType))
    return emitError() << "elementType must be a float type";

  if (shape.size() != 2)
    return emitError() << "expected rank-2 matrix, got rank " << shape.size();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// VectorType - parseShapeAndElt
//===----------------------------------------------------------------------===//

mlir::ParseResult
mlir::analog::VectorType::parseShapeAndElt(
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
// VectorType - printShapeAndElt
//===----------------------------------------------------------------------===//

void
mlir::analog::VectorType::printShapeAndElt(
    mlir::AsmPrinter &printer,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {

  for (auto d : shape)
    printer << d << 'x';

  printer.printType(elementType);
}

//===----------------------------------------------------------------------===//
// VectorType - parse
//===----------------------------------------------------------------------===//

mlir::Type
mlir::analog::VectorType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;

  // Parse 1x8xf32
  if (parseShapeAndElt(parser, shape, elementType))
    return Type();

  return get(parser.getContext(), shape, elementType);
}

//===----------------------------------------------------------------------===//
// VectorType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::VectorType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
}






//===----------------------------------------------------------------------===//
// TileGridType - verify
//===----------------------------------------------------------------------===//

llvm::LogicalResult
mlir::analog::TileGridType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> gridShape,
    llvm::ArrayRef<int64_t> tileShape,
    mlir::analog::MatrixType matrix) {

  if (tileShape.size() != 2) {
    return emitError() << "tile_shape must have exactly 2 dimensions";
  }

  if (tileShape[0] <= 0 || tileShape[1] <= 0) {
    return emitError() << "tile_shape dimensions must be positive";
  }

  if (tileShape.size() != 2) {
    return emitError() << "tile_shape must have exactly 2 dimensions";
  }

  if (tileShape[0] <= 0 || tileShape[1] <= 0) {
    return emitError() << "tile_shape dimensions must be positive";
  }

  auto matrixShape = matrix.getShape();
  if (matrixShape.size() != 2) {
    return emitError() << "matrix must be rank-2";
  }

/*  // Optional but sane invariant
  if (matrixShape[0] % tileShape[0] != 0 ||
      matrixShape[1] % tileShape[1] != 0) {
    return emitError() << "matrix shape must be divisible by tile_shape";
  }
*/

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TileGridType - parse
//===----------------------------------------------------------------------===//

mlir::Type
mlir::analog::TileGridType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> tileShape;
  mlir::Type matrixType;

  if (parser.parseLess()) {
    return Type();
  }

  // grid shape
  if (parser.parseKeyword("grid_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return Type();
  }

  int64_t gridDim;
  if (parser.parseInteger(gridDim)) {
    return Type();
  }
  gridShape.push_back(gridDim);

  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(gridDim)) {
      return Type();
    }
    gridShape.push_back(gridDim);
  }

  if (parser.parseRSquare()) {
    return Type();
  }

  if (parser.parseComma()) {
    return Type();
  }

  // tile shape
  if (parser.parseKeyword("tile_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return Type();
  }

  int64_t tileDim;
  if (parser.parseInteger(tileDim)) {
    return Type();
  }
  tileShape.push_back(tileDim);

  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(tileDim)) {
      return Type();
    }
    tileShape.push_back(tileDim);
  }

  if (parser.parseRSquare()) {
    return Type();
  }

  if (parser.parseComma()) {
    return Type();
  }

  if (parser.parseType(matrixType)) {
    return Type();
  }

  if (parser.parseGreater()) {
    return Type();
  }

  auto matrix = llvm::dyn_cast<mlir::analog::MatrixType>(matrixType);
  if (!matrix) {
    parser.emitError(parser.getCurrentLocation(), "expected analog.matrix type");
    return Type();
  }

  return get(parser.getContext(), gridShape, tileShape, matrix);
}


//===----------------------------------------------------------------------===//
// TileGridType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::TileGridType::print(mlir::AsmPrinter &printer) const {
  printer << "grid_shape=[";
  llvm::interleaveComma(getGridShape(), printer);
  printer << "], tile_shape=[";
  llvm::interleaveComma(getTileShape(), printer);
  printer << "], ";
  printer.printType(getMatrix());
}



//===----------------------------------------------------------------------===//
// VTileSliceType - verify
//===----------------------------------------------------------------------===//

llvm::LogicalResult
mlir::analog::VTileSliceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> gridShape,
    llvm::ArrayRef<int64_t> tileShape,
    mlir::analog::VectorType vector) {

  if (gridShape.size() != 2) {
    return emitError() << "grid_shape must be 2-dimensional";
  }

  if (gridShape[0] <= 0 || gridShape[1] <= 0) {
    return emitError() << "grid_shape dimension must be positive";
  }


  if (tileShape.size() != 2) {
    return emitError() << "tile_shape must be 2-dimensional";
  }

  if (tileShape[0] <= 0 || tileShape[1] <= 0) {
    return emitError() << "tile_shape dimension must be positive";
  }

  auto vecShape = vector.getShape();
  if (vecShape.size() != 2) {
    return emitError() << "vector must be rank-2 (1xN)";
  }

  if (vecShape[0] != 1) {
    return emitError() << "vector must have leading dimension 1";
  }

/*
  if (vecShape[1] % tileShape[0] != 0) {
    return emitError() << "vector length must be divisible by tile_shape";
  }
*/

  return mlir::success();
}


//===----------------------------------------------------------------------===//
// VTileSliceType - parse
//===----------------------------------------------------------------------===//

mlir::Type
mlir::analog::VTileSliceType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> tileShape;
  mlir::Type vectorType;

  if (parser.parseLess()) {
    return Type();
  }

  // grid shape
  if (parser.parseKeyword("grid_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return Type();
  }

  int64_t gridDim;
  if (parser.parseInteger(gridDim)) {
    return Type();
  }
  gridShape.push_back(gridDim);

  if (parser.parseRSquare()) {
    return Type();
  }

  if (parser.parseComma()) {
    return Type();
  }
  
  // tile shape
  if (parser.parseKeyword("tile_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return Type();
  }

  int64_t tileDim;
  if (parser.parseInteger(tileDim)) {
    return Type();
  }
  tileShape.push_back(tileDim);

  if (parser.parseRSquare()) {
    return Type();
  }

  if (parser.parseComma()) {
    return Type();
  }

  if (parser.parseType(vectorType)) {
    return Type();
  }

  if (parser.parseGreater()) {
    return Type();
  }

  auto vector = llvm::dyn_cast<mlir::analog::VectorType>(vectorType);
  if (!vector) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected analog.vector type");
    return Type();
  }

  return get(parser.getContext(), gridShape, tileShape, vector);
}


//===----------------------------------------------------------------------===//
// VTileSliceType - print
//===----------------------------------------------------------------------===//

void
mlir::analog::VTileSliceType::print(mlir::AsmPrinter &printer) const {
  printer << "grid_shape=[";
  llvm::interleaveComma(getGridShape(), printer);
  printer << "], tile_shape=[";
  llvm::interleaveComma(getTileShape(), printer);
  printer << "], ";
  printer.printType(getVector());
}
