#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "analog-base"

using namespace mlir;
using namespace mlir::analog;

//===- Generated implementation -------------------------------------------===//

#include "analog-mlir/Dialect/Analog/IR/AnalogBase.cpp.inc"

void AnalogDialect::initialize() {
  registerTypes();
  registerOps();
}

Type AnalogDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return {};

  if (parser.parseLess())
    return {};

  Type result;
  if (mnemonic == MatrixType::getMnemonic()) {
    result = MatrixType::parse(parser);
  } else if (mnemonic == VectorType::getMnemonic()) {
    result = VectorType::parse(parser);
  } else if (mnemonic == TileGridType::getMnemonic()) {
    result = TileGridType::parse(parser);
  } else if (mnemonic == VTileSliceType::getMnemonic()) {
    result = VTileSliceType::parse(parser);
  } else {
    parser.emitError(parser.getNameLoc())
        << "unknown analog type: " << mnemonic;
    return {};
  }

  if (parser.parseGreater())
    return {};

  return result;
}

void AnalogDialect::printType(Type type,
                              DialectAsmPrinter &printer) const {
  if (auto w = llvm::dyn_cast<MatrixType>(type)) {
    printer << MatrixType::getMnemonic() << "<";
    w.print(printer);
    printer << ">";
    return;
  }

  if (auto w = llvm::dyn_cast<VectorType>(type)) {
    printer << VectorType::getMnemonic() << "<";
    w.print(printer);
    printer << ">";
    return;
  }

  if (auto w = llvm::dyn_cast<TileGridType>(type)) {
    printer << TileGridType::getMnemonic() << "<";
    w.print(printer);
    printer << ">";
    return;
  }

  if (auto w = llvm::dyn_cast<VTileSliceType>(type)) {
    printer << VTileSliceType::getMnemonic() << "<";
    w.print(printer);
    printer << ">";
    return;
  }

  llvm_unreachable("unknown analog type");
}
