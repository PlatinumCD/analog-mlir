#ifndef ANALOG_OPS_TD
#define ANALOG_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===- AnalogOps.td -------------------------------------------------------===//
//
//  Minimal analog ops for: matrix -> arrays -> set/load/compute/store
//
//===----------------------------------------------------------------------===//

include "analog-mlir/Dialect/Analog/IR/AnalogBase.td"
include "analog-mlir/Dialect/Analog/IR/AnalogTypes.td"

//===----------------------------------------------------------------------===//
// analog.matrix.from_tensor
// Import a constant tensor into analog matrix storage
//===----------------------------------------------------------------------===//

def Analog_MatrixFromTensorOp
    : Analog_Op<"matrix.from_tensor"> {

  let summary = "Materialize analog matrix from a tensor";

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    Analog_MatrixType:$result
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.matrix.partition
// Create an analog array grid from matrix
//===----------------------------------------------------------------------===//

def Analog_MatrixPartitionOp
    : Analog_Op<"matrix.partition"> {

  let summary = "Create a array partition view over an analog matrix";

  let description = [{
    Declares that an analog matrix tensor is realized as a grid of
    fixed-size analog arrays. The grid shape is derived from the
    matrix shape and array dimensions.
  }];

  let arguments = (ins
    Analog_MatrixType:$matrix
  );

  let results = (outs
    Analog_MatrixGridType:$result
  );

  let assemblyFormat = [{
    $matrix attr-dict
    `:` qualified(type($matrix)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.array.matrix.place
// Place analog partition into accelerator 
//===----------------------------------------------------------------------===//

def Analog_ArrayMatrixPlaceOp
    : Analog_Op<"array.matrix.place"> {

  let summary = "Place an analog partition into accelerator";

  let description = [{
    Given a grid of analog arrays, this op places a single array.
  }];

  let arguments = (ins
    Analog_MatrixGridType:$input,
    Index:$rowIndex,
    Index:$colIndex,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $input `[` $rowIndex `,` $colIndex `]` `,` `array` `[` $indices `]`
    attr-dict
    `:` qualified(type($input)) 
  }];
}


//===----------------------------------------------------------------------===//
// analog.vector.from_tensor
// Import a constant tensor into analog vector storage
//===----------------------------------------------------------------------===//

def Analog_VectorFromTensorOp
    : Analog_Op<"vector.from_tensor"> {

  let summary = "Materialize analog vector from a tensor";

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    Analog_VectorType:$result
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.vector.partition
// Create an analog varray slice from a vector
//===----------------------------------------------------------------------===//

def Analog_VectorPartitionOp
    : Analog_Op<"vector.partition"> {

  let summary = "Partition an analog vector into slices.";

  let description = [{
    create an analog vector partition view for tiling
  }];

  let arguments = (ins
    Analog_VectorType:$vector
  );

  let results = (outs
    Analog_VectorSliceType:$result
  );

  let assemblyFormat = [{
    $vector attr-dict
    `:` qualified(type($vector)) `->` qualified(type($result))
  }];
}


//===----------------------------------------------------------------------===//
// analog.array.vector.place
// Set the varray in the accelerator
//===----------------------------------------------------------------------===//

def Analog_ArrayVectorPlaceOp
  : Analog_Op<"array.vector.place"> {

  let summary = "Place an analog varray into an accelerator";

  let arguments = (ins
    Analog_VectorSliceType:$input,
    Index:$sliceIndex,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $input `[` $sliceIndex `]` `,` `array` `[` $indices `]`
    attr-dict
    `:` qualified(type($input)) 
  }];
}


//===----------------------------------------------------------------------===//
// analog.array.execute
// Execute a array in the accelerator
//===----------------------------------------------------------------------===//

def Analog_ArrayExecuteOp
  : Analog_Op<"array.execute"> {

  let summary = "Execute a array in the accelerator";

  let arguments = (ins
    Variadic<Index>:$indices
  );

  let results = (outs
    Analog_MatrixGridType:$grid
  );

  let assemblyFormat = [{
    `array` `[` $indices `]` attr-dict `:` qualified(type($grid)) 
  }];
}

//===----------------------------------------------------------------------===//
// analog.array.store
// store the results of a array in the accelerator into a memory buffer
//===----------------------------------------------------------------------===//

def Analog_ArrayStoreOp
  : Analog_Op<"array.store"> {

  let summary = "Store the results of a array into a memory buffer";

  let arguments = (ins
    Analog_MatrixGridType:$grid,
    AnyMemRef:$dest,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $grid `,` $dest `[` $indices `]`
    attr-dict `:` qualified(type($grid)) `,` qualified(type($dest))
  }];
}

#endif // ANALOG_OPS_TD
