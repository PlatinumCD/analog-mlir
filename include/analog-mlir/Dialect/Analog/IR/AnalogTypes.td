
#ifndef ANALOG_TYPES
#define ANALOG_TYPES

include "analog-mlir/Dialect/Analog/IR/AnalogBase.td"

// =====-------------------------------------------===== 
//       MatrixType
// =====-------------------------------------------===== 

def Analog_MatrixType : Analog_Type<"Matrix", "matrix"> {
  let summary = "Analog matrix container type";

  let description = [{
    Represents a matrix intended for analog computation.
  }];

  // Parameters are what is stored in the type
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [8,8])">:$shape,
    "Type":$elementType
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

// =====-------------------------------------------===== 
//       VectorType
// =====-------------------------------------------===== 

def Analog_VectorType : Analog_Type<"Vector", "vector"> {
  let summary = "Analog vector container type";

  let description = [{
    Represents a vector intended for analog computation.
  }];

  // Parameters are what is stored in the type
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$shape,
    "Type":$elementType
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

// =====-------------------------------------------===== 
//       MatrixGridType
// =====-------------------------------------------===== 

def Analog_MatrixGridType : Analog_Type<"MatrixGrid", "matrix.grid"> {
  let summary = "Analog array grid view type";

  let description = [{
    Represents a 2D grid of analog arrays, each array having a fixed shape
    and element type.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$gridShape,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$arrayShape,
    Analog_MatrixType:$matrix
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


// =====-------------------------------------------===== 
//       VectorSliceType
// =====-------------------------------------------===== 

def Analog_VectorSliceType : Analog_Type<"VectorSlice", "vector.slice"> {
  let summary = "Analog array slice view type";

  let description = [{
    Represents a 1D view of analog varrays, each varray having a fixed shape
    and element type.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$gridShape,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$arrayShape,
    Analog_VectorType:$vector
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


#endif
