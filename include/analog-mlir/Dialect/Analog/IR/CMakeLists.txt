message(STATUS "ENTERING: ${CMAKE_CURRENT_SOURCE_DIR}")

set(LLVM_TARGET_DEFINITIONS AnalogDialect.td)

mlir_tablegen(AnalogBase.h.inc -gen-dialect-decls)
mlir_tablegen(AnalogBase.cpp.inc -gen-dialect-defs)
mlir_tablegen(AnalogTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AnalogTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(AnalogIRIncGen)

add_dependencies(AnalogIncGen AnalogIRIncGen)
