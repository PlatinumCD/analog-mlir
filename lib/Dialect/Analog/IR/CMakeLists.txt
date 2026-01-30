add_mlir_dialect_library(AnalogIR
  AnalogBase.cpp
  AnalogTypes.cpp
  AnalogOps.cpp

  DEPENDS
  AnalogIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
