add_mlir_library(MLIRAnalogConversion
  ConvertAnalogToGolemBackend.cpp
  EmitRuntimeGraph.cpp
  FinalizeGolemIntrinsics.cpp
  Passes.cpp
  golem/GolemUtils.cpp
  golem/LowerExecute.cpp
  golem/LowerMatrix.cpp
  golem/LowerStore.cpp
  golem/LowerVector.cpp

  LINK_LIBS PUBLIC
  AnalogIR
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
