include "Analog.td"
include "mlir/IR/AttrTypeBase.td"

class Analog_Type<string name, string mnemonic, list<Trait> traits = []>
    : TypeDef<Analog_Dialect, name, traits> {
  let mnemonic = mnemonic;
}


// =====-------------------------------------------===== 
//       WeightsType
// =====-------------------------------------------===== 

def Analog_WeightsType<"Weights", "weights"> {
  let summary = "Analog weights container type";

  let description = [{
    Represents a weight matrix intended for analog computation,
    with an explicit layout tag and a logical stride (in elements).
  }];

  // Parameters are what is stored in the type
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [8,8])">:$shape,
    "Type":$elementType,
    "StringAttr":$layout,
    "int64_t":$stride
  );

  // AssemblyFormat is how it prints and parses
  let assemblyFormat = [{
    < custom<ShapeAndElt>($shape, $elementType),
      layout = $layout,
      stride = $stride >
  }];

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
