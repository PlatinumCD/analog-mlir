#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_REWRITEUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_REWRITEUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

#include <cctype>
#include <string>

namespace mlir {
namespace analog {
namespace rewrite_utils {

// Builds a symbol-friendly lowercase stem from an extractor layer label.
inline std::string makeFunctionBaseName(StringRef layerType) {
  std::string baseName;
  baseName.reserve(layerType.size());

  for (char ch : layerType) {
    unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c))
      baseName.push_back(static_cast<char>(std::tolower(c)));
    else if (std::isspace(c))
      baseName.push_back('_');
  }

  if (baseName.empty())
    baseName = "layer";

  return baseName;
}

// Chooses a module-local function name derived from the layer label.
inline std::string makeUniqueFunctionName(ModuleOp module,
                                          StringRef layerType) {
  unsigned functionIndex = 0;
  std::string baseName = makeFunctionBaseName(layerType);
  std::string functionName = baseName + "_" + std::to_string(functionIndex);
  while (module.lookupSymbol(functionName)) {
    ++functionIndex;
    functionName = baseName + "_" + std::to_string(functionIndex);
  }
  return functionName;
}

// Mirrors SSA value types into the signature vectors used by outlined funcs.
inline SmallVector<Type> collectValueTypes(llvm::ArrayRef<Value> values) {
  SmallVector<Type> types;
  types.reserve(values.size());
  for (Value value : values)
    types.push_back(value.getType());
  return types;
}

// Outlines a matched subgraph into a new function and replaces the root with
// a call using the caller-provided input and output boundaries.
inline void extractToFunction(Operation *root, llvm::ArrayRef<Operation *> ops,
                              llvm::ArrayRef<Value> inputs,
                              llvm::ArrayRef<Value> outputs,
                              RewriterBase &rewriter, StringRef layerType) {
  // Refuse partial matches that cannot produce a valid outlined function.
  if (!root || ops.empty() || outputs.empty())
    return;

  auto module = root->getParentOfType<ModuleOp>();
  if (!module)
    return;

  // Materialize the function signature and metadata in the parent module.
  SmallVector<Type> inputTypes = collectValueTypes(inputs);
  SmallVector<Type> outputTypes = collectValueTypes(outputs);
  std::string functionName = makeUniqueFunctionName(module, layerType);

  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc = rewriter.create<func::FuncOp>(
      root->getLoc(), functionName, functionType);
  extractedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));
  extractedFunc->setAttr("layer_domain", rewriter.getStringAttr("digital"));

  Block *entryBlock = extractedFunc.addEntryBlock();
  IRMapping mapping;
  for (unsigned i = 0; i < inputs.size(); ++i)
    mapping.map(inputs[i], entryBlock->getArgument(i));

  // Clone the matched body with external values remapped to block arguments.
  rewriter.setInsertionPointToStart(entryBlock);
  for (Operation *op : ops)
    rewriter.clone(*op, mapping);

  // Return the cloned values that correspond to the original output boundary.
  SmallVector<Value> mappedOutputs;
  for (Value output : outputs)
    mappedOutputs.push_back(mapping.lookupOrDefault(output));
  rewriter.create<func::ReturnOp>(root->getLoc(), mappedOutputs);

  // Replace the original subgraph boundary with a call to the outlined func.
  rewriter.setInsertionPoint(root);
  auto call = rewriter.create<func::CallOp>(root->getLoc(),
                                            extractedFunc.getSymName(),
                                            outputTypes, inputs);

  for (unsigned i = 0; i < outputs.size(); ++i) {
    Value output = outputs[i];
    output.replaceAllUsesWith(call.getResult(i));
  }

  // Drop now-dead matched ops from consumers back to producers.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    if ((*it)->use_empty())
      rewriter.eraseOp(*it);
  }
}

} // namespace rewrite_utils
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_REWRITEUTILS_H
