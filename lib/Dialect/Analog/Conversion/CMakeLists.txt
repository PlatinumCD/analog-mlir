add_mlir_library(MLIRAnalogConversion
  Passes.cpp
  ConvertAnalogToGolemBackend.cpp
  FinalizeGolemIntrinsics.cpp

  LINK_LIBS PUBLIC
  AnalogIR
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
