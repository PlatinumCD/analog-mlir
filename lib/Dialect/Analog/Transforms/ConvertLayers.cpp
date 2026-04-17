#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassRegistry.h"

namespace converter_utils = mlir::analog::converter_utils;

namespace mlir {
namespace analog {

// Dispatches each still-digital extracted layer called by forward to its
// layer_type converter.
void ConvertLayersPass::runOnOperation() {
  // Build the converter registry once so calls can dispatch by layer_type.
  mlir::analog::LayerConverters converters;
  mlir::analog::LayerConverterMap converterMap;
  mlir::analog::registerLinearConverter(converters, converterMap, &getContext());
  mlir::analog::registerConv1DConverter(converters, converterMap, &getContext());
  mlir::analog::registerConv2DConverter(converters, converterMap, &getContext());
  mlir::analog::registerConv2DGroupedConverter(converters, converterMap,
                                               &getContext());
  mlir::analog::registerConv3DConverter(converters, converterMap, &getContext());
  mlir::analog::registerRNNCellConverter(converters, converterMap,
                                         &getContext());
  mlir::analog::registerLSTMCellConverter(converters, converterMap,
                                          &getContext());

  // Only forward owns the executable layer call sequence for this pass.
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    func.walk([&](mlir::func::CallOp call) {
      // Resolve the callee to an extracted layer function with usable metadata.
      auto calleeAttr = call->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
      if (!calleeAttr)
        return;

      auto layerFunc =
          getOperation().lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue());
      if (!layerFunc)
        return;

      auto layerType = converter_utils::getLayerType(layerFunc);
      if (!layerType)
        return;

      if (!converter_utils::isDigitalLayer(layerFunc))
        return;

      // Leave unknown layer types untouched so other converters can be added
      // without changing the traversal contract.
      auto converterIt = converterMap.find(layerType.getValue());
      if (converterIt == converterMap.end())
        return;

      const mlir::analog::LayerConverter *converter = converterIt->second;
      converter->convert(layerFunc, arrayRows, arrayCols);
    });
  }
}

// Registers the layer conversion pass with MLIR's global pass registry.
void registerConvertLayersPass() {
  PassRegistration<ConvertLayersPass>();
}

} // namespace analog
} // namespace mlir
