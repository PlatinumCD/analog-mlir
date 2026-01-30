
#ifndef ANALOG_TYPES
#define ANALOG_TYPES

include "analog-mlir/Dialect/Analog/IR/AnalogBase.td"

// =====-------------------------------------------===== 
//       WeightsType
// =====-------------------------------------------===== 

def Analog_WeightsType : Analog_Type<"Weights", "weights"> {
  let summary = "Analog weights container type";

  let description = [{
    Represents a weight matrix intended for analog computation,
    with an explicit layout tag and a logical stride (in elements).
  }];

  // Parameters are what is stored in the type
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [8,8])">:$shape,
    "Type":$elementType,
    "StringAttr":$layout
  );

  let hasCustomAssemblyFormat = 1;

  // Generate Verify Declarations (We write this function)
  let genVerifyDecl = 1;

  // Add additional Declarations (We write these functions)
  let extraClassDeclaration = [{
    static ::mlir::ParseResult
    parseShapeAndElt(::mlir::AsmParser &parser,
                     ::llvm::SmallVector<int64_t> &shape,
                     ::mlir::Type &elementType);

    static void
    printShapeAndElt(::mlir::AsmPrinter &printer,
                     ::llvm::ArrayRef<int64_t> shape,
                     ::mlir::Type elementType);

  }];
}

#endif
