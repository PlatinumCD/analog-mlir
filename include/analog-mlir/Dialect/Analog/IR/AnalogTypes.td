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

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [8,8])">:$shape,
    "Type":$elementType
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
//       VectorType
// =====-------------------------------------------=====

def Analog_VectorType : Analog_Type<"Vector", "vector"> {
  let summary = "Analog vector container type";

  let description = [{
    Represents a vector intended for analog computation.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$shape,
    "Type":$elementType
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
    Represents a 1D view of analog vector arrays, each with a fixed shape and
    element type.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$gridShape,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$arrayShape,
    Analog_VectorType:$vector
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// =====-------------------------------------------=====
//       TaskGraphType
// =====-------------------------------------------=====

def Analog_TaskGraphType : Analog_Type<"TaskGraph", "task_graph"> {
  let summary = "Analog task graph handle type";

  let description = [{
    Represents a lightweight handle to a task graph under construction in the
    analog IR.
  }];
}

// =====-------------------------------------------=====
//       TaskType
// =====-------------------------------------------=====

def Analog_TaskType : Analog_Type<"Task", "task"> {
  let summary = "Analog task handle type";

  let description = [{
    Represents a single scheduled task node inside an analog task graph.
  }];
}

// =====-------------------------------------------=====
//       RuntimeHandleType
// =====-------------------------------------------=====

def Analog_RuntimeHandleType : Analog_Type<"RuntimeHandle", "runtime_handle"> {
  let summary = "Analog runtime-owned opaque handle type";

  let description = [{
    Represents runtime-owned persistent state materialized by init tasks and
    consumed by later graph tasks through explicit persistent resource slots.
  }];
}

// =====-------------------------------------------=====
//       TaskResourceType
// =====-------------------------------------------=====

def Analog_TaskResourceType : Analog_Type<"TaskResource", "task_resource"> {
  let summary = "Analog task graph resource handle type";

  let description = [{
    Represents one graph-owned resource slot carrying a concrete payload type
    used as a task input, task output, graph boundary value, or internal
    temporary.
  }];

  let parameters = (ins
    "Type":$valueType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // ANALOG_TYPES
