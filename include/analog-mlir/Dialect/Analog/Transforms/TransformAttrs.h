#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TRANSFORM_ATTRS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TRANSFORM_ATTRS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace analog {
namespace detail {

inline constexpr llvm::StringLiteral kMatrixSourceIdAttr =
    "analog.matrix_source_id";
inline constexpr llvm::StringLiteral kDeleteInFuturePassAttr =
    "analog.delete_in_future_pass";
inline constexpr llvm::StringLiteral kSlidingWindowMatmulAttr =
    "analog.sliding_window_matmul";
inline constexpr llvm::StringLiteral kSlidingWindowBiasAddAttr =
    "analog.sliding_window_bias_add";
inline constexpr llvm::StringLiteral kSlidingWindowPatchAttr =
    "analog.sliding_window_patch";
inline constexpr llvm::StringLiteral kOutputChannelAssemblyAttr =
    "analog.output_channel_assembly";
inline constexpr llvm::StringLiteral kRecurrentPatternAttr =
    "analog.recurrent_pattern";
inline constexpr llvm::StringLiteral kRecurrentActivationAttr =
    "analog.recurrent_activation";
inline constexpr llvm::StringLiteral kRecurrentInputSizeAttr =
    "analog.recurrent_input_size";
inline constexpr llvm::StringLiteral kRecurrentHiddenSizeAttr =
    "analog.recurrent_hidden_size";
inline constexpr llvm::StringLiteral kRecurrentStepsAttr =
    "analog.recurrent_steps";
inline constexpr llvm::StringLiteral kPreparedForAnalogAttr =
    "analog.prepared_for_analog";
inline constexpr llvm::StringLiteral kMatmulExecIdAttr =
    "analog.matmul_exec_id";
inline constexpr llvm::StringLiteral kShimRequiredAttr =
    "analog-shim-required";
inline constexpr llvm::StringLiteral kLayerIdAttr = "layer-id";
inline constexpr llvm::StringLiteral kWeightIdAttr = "weight-id";

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TRANSFORM_ATTRS_H
