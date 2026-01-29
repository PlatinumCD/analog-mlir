
include "mlir/IR/DialectBase.td"

def Analog_Dialect : Dialect {
  let name = "analog";
  let cppNamespace = "::analog";
}
