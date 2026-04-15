execute_process(
  COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY') or '')"
  OUTPUT_VARIABLE GOLEM_SIM_PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"
  OUTPUT_VARIABLE GOLEM_SIM_PYTHON_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBPL') or '')"
  OUTPUT_VARIABLE GOLEM_SIM_PYTHON_LIBPL
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND python3 -c "import sysconfig; print((sysconfig.get_config_var('LDLIBRARY') or '').removeprefix('lib').removesuffix('.so'))"
  OUTPUT_VARIABLE GOLEM_SIM_PYTHON_LIBNAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_library(GOLEM_SIM_PYTHON_LIBRARY
  NAMES ${GOLEM_SIM_PYTHON_LIBNAME}
  HINTS ${GOLEM_SIM_PYTHON_LIBDIR} ${GOLEM_SIM_PYTHON_LIBPL}
)

if(NOT GOLEM_SIM_PYTHON_LIBRARY)
  message(FATAL_ERROR "Failed to locate embedded Python library")
endif()

add_llvm_library(MLIRAnalogRuntime
  lib/TaskGraphRuntime.cpp
  lib/SimBridge.cpp
  lib/GolemShims.cpp
)

target_include_directories(MLIRAnalogRuntime PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/golem-simulator/include
)

target_link_libraries(MLIRAnalogRuntime PUBLIC
  ${GOLEM_SIM_PYTHON_LIBRARY}
)

target_compile_definitions(MLIRAnalogRuntime PRIVATE
  ANALOG_GOLEM_SIM_PYTHON_DIR="${PROJECT_SOURCE_DIR}/tools/golem-simulator/python"
)

set_target_properties(MLIRAnalogRuntime PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/runtime"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/runtime"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/runtime"
)
