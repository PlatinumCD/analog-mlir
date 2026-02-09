#ifndef ANALOG_BASE_TD
#define ANALOG_BASE_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Analog_Dialect : Dialect {
  let name = "analog";
  let cppNamespace = "::mlir::analog";

  let extraClassDeclaration = [{
    void registerTypes();
    void registerOps();
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
  }];
}


class Analog_Type<string name, string mnem, list<Trait> traits = []>
    : TypeDef<Analog_Dialect, name, traits> {
  let mnemonic = mnem;
}


class Analog_Op<string mnem, list<Trait>traits = []>
    : Op<Analog_Dialect, mnem, traits> {
  let hasCustomAssemblyFormat = 0;
}

#endif // ANALOG_BASE_TD
