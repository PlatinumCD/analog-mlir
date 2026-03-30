#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REGION_UTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REGION_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analog {
namespace detail {

Operation *findTopLevelOwner(Operation *op);

bool allUsesStayWithinTopLevelOwners(
    Operation *producer, const llvm::DenseSet<Operation *> &segmentOwners);

llvm::DenseSet<Operation *> collectSegmentClosure(llvm::ArrayRef<Operation *> segment);

bool isRegionInsideRegion(Region *candidate, Region &target);

bool isOpInsideRegion(Operation *op, Region &region);

bool isValueDefinedInsideRegion(Value value, Region &region);

void computeEscapingResults(llvm::ArrayRef<Operation *> segment,
                            llvm::SmallVectorImpl<Value> &escapingResults);

void collectExternalValuesForRegion(scf::ExecuteRegionOp exec,
                                    llvm::SmallVectorImpl<Value> &externalValues);

bool isValueAvailableAtOp(Value value, Operation *op);

void filterUndominatedExternalValues(scf::ExecuteRegionOp exec,
                                     llvm::SmallVectorImpl<Value> &values);

bool allUsesInsideRegion(arith::ConstantOp cst, Region &region);

bool allUsesInsideRegion(Operation *op, Region &region);

void replaceUsesInsideRegion(Value oldValue, Value newValue, Region &region);

void moveSegmentIntoBlock(llvm::ArrayRef<Operation *> segment, Block *body);

std::pair<Block *, Block *> createExecuteRegionBlocks(scf::ExecuteRegionOp exec);

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REGION_UTILS_H
