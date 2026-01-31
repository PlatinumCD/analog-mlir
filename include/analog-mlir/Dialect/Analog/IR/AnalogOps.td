#ifndef ANALOG_OPS_TD
#define ANALOG_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

//===- AnalogOps.td -------------------------------------------------------===//
//
//  Minimal analog ops for: weights -> tiles -> set/load/compute/store
//
//===----------------------------------------------------------------------===//

include "analog-mlir/Dialect/Analog/IR/AnalogBase.td"
include "analog-mlir/Dialect/Analog/IR/AnalogTypes.td"

//===----------------------------------------------------------------------===//
// analog.weights.from_const
// Import a constant tensor into analog weight storage
//===----------------------------------------------------------------------===//

def Analog_WeightsFromConstOp
    : Analog_Op<"weights.from_const"> {

  let summary = "Materialize analog weights from a constant tensor";

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$layer
  );

  let results = (outs
    Analog_WeightsType:$result
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` qualified(type($result))
  }];
}

#endif // ANALOG_OPS_TD
