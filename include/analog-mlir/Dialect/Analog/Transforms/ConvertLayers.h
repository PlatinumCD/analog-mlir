#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_CONVERTLAYERS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_CONVERTLAYERS_H

#include "analog-mlir/Dialect/Analog/IR/AnalogDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;

namespace analog {

// Defines the extension point for rewriting one extracted digital layer into
// analog IR.
class LayerConverter {
public:
  // Allows converter implementations to be owned through the base interface.
  virtual ~LayerConverter() = default;

  // Identifies the layer_type this converter claims in dispatch maps.
  virtual StringRef getName() const = 0;

  // Converts an extracted digital layer in place for the requested array shape.
  virtual void convert(func::FuncOp func, int64_t arrayRows,
                       int64_t arrayCols) const = 0;
};

// Owns converter instances while dispatch tables keep non-owning pointers.
using LayerConverters = std::vector<std::unique_ptr<LayerConverter>>;

// Maps layer_type strings to the converter that can lower them to analog IR.
using LayerConverterMap = llvm::StringMap<const LayerConverter *>;

// Scans forward's layer calls and delegates still-digital layer functions to a
// matching converter.
struct ConvertLayersPass
    : public mlir::PassWrapper<ConvertLayersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLayersPass)

  // Carries the target analog array row count to converter implementations.
  Option<int64_t> arrayRows{
      *this, "array-rows",
      llvm::cl::desc("Number of rows in the analog array"),
      llvm::cl::init(0)};

  // Carries the target analog array column count to converter implementations.
  Option<int64_t> arrayCols{
      *this, "array-cols",
      llvm::cl::desc("Number of columns in the analog array"),
      llvm::cl::init(0)};

  // Uses the default option values for command-line pass construction.
  ConvertLayersPass() = default;

  // Copies pass options when MLIR clones the pass for a pass manager run.
  ConvertLayersPass(const ConvertLayersPass &pass)
      : PassWrapper(pass),
        arrayRows(*this, "array-rows",
                  llvm::cl::desc("Number of rows in the analog array"),
                  llvm::cl::init(0)),
        arrayCols(*this, "array-cols",
                  llvm::cl::desc("Number of columns in the analog array"),
                  llvm::cl::init(0)) {
    arrayRows = pass.arrayRows;
    arrayCols = pass.arrayCols;
  }

  // Provides the stable command-line name used to schedule this transform.
  mlir::StringRef getArgument() const final {
    return "analog-convert-layers";
  }

  // Summarizes the pass behavior for MLIR pass registration and help text.
  mlir::StringRef getDescription() const final {
    return "Convert extracted layers to analog";
  }

  // Ensures analog and loop/buffer operations introduced by converters exist.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::analog::AnalogDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  // Converts layer functions called by forward when their layer_type is known.
  void runOnOperation() override;
};

// Installs the built-in linear converter and every layer_type alias it handles.
void registerLinearConverter(LayerConverters &converters,
                             LayerConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in Conv1D converter and every layer_type alias it handles.
void registerConv1DConverter(LayerConverters &converters,
                             LayerConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in Conv2D converter and every layer_type alias it handles.
void registerConv2DConverter(LayerConverters &converters,
                             LayerConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in grouped Conv2D converter and every layer_type alias it
// handles.
void registerConv2DGroupedConverter(LayerConverters &converters,
                                    LayerConverterMap &converterMap,
                                    MLIRContext *context);

// Installs the built-in Conv3D converter and every layer_type alias it handles.
void registerConv3DConverter(LayerConverters &converters,
                             LayerConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in RNN cell converter and every layer_type alias it
// handles.
void registerRNNCellConverter(LayerConverters &converters,
                              LayerConverterMap &converterMap,
                              MLIRContext *context);

// Registers the layer conversion pass with MLIR's global pass registry.
void registerConvertLayersPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_CONVERTLAYERS_H
