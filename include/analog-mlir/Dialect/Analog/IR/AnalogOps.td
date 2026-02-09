#ifndef ANALOG_OPS_TD
#define ANALOG_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===- AnalogOps.td -------------------------------------------------------===//
//
//  Minimal analog ops for: matrix -> tiles -> set/load/compute/store
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
// analog.tile.partition
// Create an analog tile grid from matrix
//===----------------------------------------------------------------------===//

def Analog_TilePartitionOp
    : Analog_Op<"tile.partition"> {

  let summary = "Create a tile partition view over an analog matrix";

  let description = [{
    Declares that an analog matrix tensor is realized as a grid of
    fixed-size analog tiles. The grid shape is derived from the
    matrix shape and tile dimensions.
  }];

  let arguments = (ins
    Analog_MatrixType:$matrix,
    I64ArrayAttr:$tile
  );

  let results = (outs
    Analog_TileGridType:$result
  );

  let assemblyFormat = [{
    $matrix attr-dict
    `:` qualified(type($matrix)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.tile.place
// Place analog partition into accelerator 
//===----------------------------------------------------------------------===//

def Analog_TilePlaceOp
    : Analog_Op<"tile.place"> {

  let summary = "Place an analog partition into accelerator";

  let description = [{
    Given a grid of analog tiles, this op places a single tile.
  }];

  let arguments = (ins
    Analog_TileGridType:$input,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $input `[` $indices `]`
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
// analog.vtile.partition
// Create an analog vtile slice from a vector
//===----------------------------------------------------------------------===//

def Analog_VTilePartitionOp
    : Analog_Op<"vtile.partition"> {

  let summary = "Partition an analog vector into slices.";

  let description = [{
    create an analog vector partition view for tiling
  }];

  let arguments = (ins
    Analog_VectorType:$vector,
    I64ArrayAttr:$tile
  );

  let results = (outs
    Analog_VTileSliceType:$result
  );

  let assemblyFormat = [{
    $vector attr-dict
    `:` qualified(type($vector)) `->` qualified(type($result))
  }];
}


//===----------------------------------------------------------------------===//
// analog.vtile.place
// Set the vtile in the accelerator
//===----------------------------------------------------------------------===//

def Analog_VTilePlaceOp
  : Analog_Op<"vtile.place"> {

  let summary = "Place an analog vtile into an accelerator";

  let arguments = (ins
    Analog_VTileSliceType:$input,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $input `[` $indices `]`
    attr-dict
    `:` qualified(type($input)) 
  }];
}

#endif // ANALOG_OPS_TD
