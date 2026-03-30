#include "analog-mlir/Dialect/Analog/Transforms/TensorConstantUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace detail {

FailureOr<SmallVector<float>> extractFloatElements(arith::ConstantOp op) {
  if (!op)
    return failure();

  TypedAttr attr = op.getValue();
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (APFloat value : denseAttr.getValues<APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (auto resourceAttr = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    if (std::optional<ArrayRef<float>> values = resourceAttr.tryGetAsArrayRef())
      return SmallVector<float>(values->begin(), values->end());
  }

  return failure();
}

arith::ConstantOp createDenseF32ResourceConstant(OpBuilder &builder, Location loc,
                                                 RankedTensorType type,
                                                 StringRef resourceName,
                                                 ArrayRef<float> values) {
  Attribute attr = DenseF32ResourceElementsAttr::get(
      type, resourceName,
      HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(values));
  return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
}

std::string makeNumberedResourceName(StringRef prefix, unsigned &counter) {
  return (prefix + Twine(counter++)).str();
}

void eraseIfDead(arith::ConstantOp op) {
  if (op && op->use_empty())
    op.erase();
}

} // namespace detail
} // namespace analog
} // namespace mlir
