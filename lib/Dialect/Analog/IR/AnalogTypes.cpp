

// =====------------------------------------------------------=====
//    WeightsType - verify
// =====------------------------------------------------------=====
::mlir::LogicalResult
WeightsType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                    ::llvm::ArrayRef<int64_t> shape,
                    ::mlir::Type elementType,
                    ::mlir::StringAttr layout,
                    int64_t stride) {
  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !elementType.isa<::mlir::FloatType>())
    return emitError() << "elementType must be a float type";

  if (!layout)
    return emitError() << "layout must be a string attribute";

  // Optional: enforce 2D weights for your use-case.
  if (shape.size() != 2)
    return emitError() << "expected rank-2 weights, got rank " << shape.size();

  if (stride <= 0)
    return emitError() << "stride must be > 0";

  // Optional: enforce stride equals the “logical row length”.
  if (stride < shape[1])
    return emitError() << "stride must be >= inner dimension (" << shape[1] << ")";

  return ::mlir::success();
}


// =====------------------------------------------------------=====
//    WeightsType - parseShapeAndElt
// =====------------------------------------------------------=====

::mlir::ParseResult
WeightsType::parseShapeAndElt(::mlir::AsmParser &parser,
                             ::llvm::SmallVector<int64_t> &shape,
                             ::mlir::Type &elementType) {
  // Parse: 8x8xf32  (general rank: d0xd1x...x<elt>)
  shape.clear();

  // Parse one or more "Nx" prefixes.
  while (true) {
    int64_t dim = 0;
    auto maybeInt = parser.parseOptionalInteger(dim);
    if (maybeInt.has_value()) {
      shape.push_back(dim);
      // Expect 'x' after each dim.
      if (parser.parseXInDimensionList())
        return ::mlir::failure();

      // If next token is an integer, loop again; otherwise break and parse elt.
      if (parser.getToken().is(::mlir::AsmToken::Integer))
        continue;
    } else {
      // No integer where we expected dims => error.
      return parser.emitError(parser.getCurrentLocation(),
                              "expected dimension list like 8x8x...");
    }
    break;
  }

  // Parse element type (e.g., f32).
  if (parser.parseType(elementType))
    return ::mlir::failure();

  return ::mlir::success();
}


// =====------------------------------------------------------=====
//    WeightsType - parseShapeAndElt
// =====------------------------------------------------------=====
void WeightsType::printShapeAndElt(::mlir::AsmPrinter &printer,
                                  ::llvm::ArrayRef<int64_t> shape,
                                  ::mlir::Type elementType) {
  for (int64_t d : shape)
    printer << d << 'x';
  printer << elementType;
}
