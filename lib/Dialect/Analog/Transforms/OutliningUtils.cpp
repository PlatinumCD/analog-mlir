#include "analog-mlir/Dialect/Analog/Transforms/OutliningUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace detail {

std::string buildOutlinedFunctionName(StringRef prefix, int64_t id) {
  return (prefix + std::to_string(id)).str();
}

void cloneOutlinedOpsIntoBuilder(Block &source, OpBuilder &builder,
                                 IRMapping &mapper) {
  for (Operation &op : source) {
    if (isa<cf::BranchOp, scf::YieldOp>(op))
      continue;
    Operation *cloned = builder.clone(op, mapper);
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
      mapper.map(oldRes, newRes);
  }
}

FailureOr<SmallVector<Value>> getOutlinedLayerReturns(Block &bodyBlk,
                                                      Block &exitBlk,
                                                      IRMapping &mapper) {
  auto br = dyn_cast<cf::BranchOp>(bodyBlk.getTerminator());
  if (!br || br.getDest() != &exitBlk)
    return failure();

  SmallVector<Value> returns;
  returns.reserve(br.getNumOperands());
  for (Value v : br.getDestOperands())
    returns.push_back(mapper.lookupOrDefault(v));
  return returns;
}

} // namespace detail
} // namespace analog
} // namespace mlir
