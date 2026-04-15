#ifndef ANALOG_OPS_TD
#define ANALOG_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
// Create an analog array grid from a matrix
//===----------------------------------------------------------------------===//

def Analog_MatrixPartitionOp
    : Analog_Op<"matrix.partition"> {

  let summary = "Create an array partition view over an analog matrix";

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
// Place an analog partition into the accelerator
//===----------------------------------------------------------------------===//

def Analog_ArrayMatrixPlaceOp
    : Analog_Op<"array.matrix.place"> {

  let summary = "Place an analog partition into the accelerator";

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
// Create an analog vector-array slice from a vector
//===----------------------------------------------------------------------===//

def Analog_VectorPartitionOp
    : Analog_Op<"vector.partition"> {

  let summary = "Partition an analog vector into slices";

  let description = [{
    Creates an analog vector partition view for tiling.
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
// Place a vector-array slice in the accelerator
//===----------------------------------------------------------------------===//

def Analog_ArrayVectorPlaceOp
  : Analog_Op<"array.vector.place"> {

  let summary = "Place an analog vector-array slice into an accelerator";

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
// analog.task_graph.create
// Create a task graph handle
//===----------------------------------------------------------------------===//

def Analog_TaskGraphCreateOp
  : Analog_Op<"task_graph.create"> {

  let summary = "Create an analog task graph handle";

  let results = (outs
    Analog_TaskGraphType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.task_graph.input
// Create a graph input resource handle
//===----------------------------------------------------------------------===//

def Analog_TaskGraphInputOp
  : Analog_Op<"task_graph.input"> {

  let summary = "Create a graph input resource handle";

  let arguments = (ins
    Analog_TaskGraphType:$graph
  );

  let results = (outs
    Analog_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.task_graph.output
// Create a graph output resource handle
//===----------------------------------------------------------------------===//

def Analog_TaskGraphOutputOp
  : Analog_Op<"task_graph.output"> {

  let summary = "Create a graph output resource handle";

  let arguments = (ins
    Analog_TaskGraphType:$graph
  );

  let results = (outs
    Analog_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.task_graph.temporary
// Create an internal graph temporary resource handle
//===----------------------------------------------------------------------===//

def Analog_TaskGraphTemporaryOp
  : Analog_Op<"task_graph.temporary"> {

  let summary = "Create an internal graph temporary resource handle";

  let arguments = (ins
    Analog_TaskGraphType:$graph
  );

  let results = (outs
    Analog_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.task_graph.persistent
// Create a graph persistent resource handle
//===----------------------------------------------------------------------===//

def Analog_TaskGraphPersistentOp
  : Analog_Op<"task_graph.persistent"> {

  let summary = "Create a graph persistent resource handle";

  let arguments = (ins
    Analog_TaskGraphType:$graph
  );

  let results = (outs
    Analog_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// analog.task.create
// Create a task node in a task graph
//===----------------------------------------------------------------------===//

def Analog_TaskCreateOp
  : Analog_Op<"task.create"> {

  let summary = "Create a task in an analog task graph";

  let description = [{
    Creates one task node associated with a task graph. The task records the
    callee it will invoke, its layer type, its explicit dependencies, and
    whether it runs in the analog or digital domain.
  }];

  let arguments = (ins
    Analog_TaskGraphType:$graph,
    FlatSymbolRefAttr:$callee,
    StrAttr:$domain,
    OptionalAttr<StrAttr>:$layerType,
    Variadic<Analog_TaskResourceType>:$inputs,
    Variadic<Analog_TaskResourceType>:$outputs,
    Variadic<Analog_TaskType>:$dependencies
  );

  let results = (outs
    Analog_TaskType:$result
  );

  let hasVerifier = 1;
  let traits = [AttrSizedOperandSegments];

  let assemblyFormat = [{
    $graph `,` $callee `,` `domain` `=` $domain
    (`,` `layer_type` `=` $layerType^)?
    `,`
    `inputs` `[` $inputs `]`
    `,`
    `outputs` `[` $outputs `]`
    `,`
    `deps` `[` $dependencies `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// analog.array.execute
// Execute an array in the accelerator
//===----------------------------------------------------------------------===//

def Analog_ArrayExecuteOp
  : Analog_Op<"array.execute"> {

  let summary = "Execute an array in the accelerator";

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
// Store the results of an array in the accelerator into a memory buffer
//===----------------------------------------------------------------------===//

def Analog_ArrayStoreOp
  : Analog_Op<"array.store"> {

  let summary = "Store the results of an array into a memory buffer";

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
