
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
//       TileGridType
// =====-------------------------------------------===== 

def Analog_TileGridType : Analog_Type<"TileGrid", "tile.grid"> {
  let summary = "Analog tile grid view type";

  let description = [{
    Represents a 2D grid of analog tiles, each tile having a fixed shape
    and element type.
  }];

  let parameters = (ins
    "int64_t":$tiles,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$tileShape,
    Analog_MatrixType:$matrix
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

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
//       VTileSliceType
// =====-------------------------------------------===== 

def Analog_VTileSliceType : Analog_Type<"VTileSlice", "vtile.slice"> {
  let summary = "Analog tile slice view type";

  let description = [{
    Represents a 1D view of analog vtiles, each vtile having a fixed shape
    and element type.
  }];

  let parameters = (ins
    "int64_t":$tiles,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$tileShape,
    Analog_VectorType:$vector
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


#endif
