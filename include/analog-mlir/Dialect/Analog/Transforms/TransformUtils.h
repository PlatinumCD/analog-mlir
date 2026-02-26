#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TRANSFORM_UTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TRANSFORM_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include <cstdint>

namespace mlir {
namespace analog {
namespace detail {

struct GridTiling2D {
  int64_t rows;
  int64_t cols;
};

inline GridTiling2D computeGridTiling2D(int64_t matrixRows, int64_t matrixCols,
                                        int64_t arrayRows, int64_t arrayCols) {
  return {
      (matrixRows + arrayRows - 1) / arrayRows,
      (matrixCols + arrayCols - 1) / arrayCols,
  };
}

template <typename BodyBuilderFn>
inline void build2DIndexLoopNest(OpBuilder &builder, Location loc,
                                 int64_t upperRows, int64_t upperCols,
                                 BodyBuilderFn &&emitBody) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value ubRow = builder.create<arith::ConstantIndexOp>(loc, upperRows);
  Value ubCol = builder.create<arith::ConstantIndexOp>(loc, upperCols);

  builder.create<scf::ForOp>(
      loc, zero, ubRow, one, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value rowIdx, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, zero, ubCol, one, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value colIdx,
                ValueRange) {
              emitBody(colBuilder, colLoc, rowIdx, colIdx);
              colBuilder.create<scf::YieldOp>(colLoc);
            });

        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TRANSFORM_UTILS_H
