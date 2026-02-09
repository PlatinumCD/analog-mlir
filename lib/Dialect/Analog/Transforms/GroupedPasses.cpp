#include "analog-mlir/Dialect/Analog/Transforms/GroupedPasses.h"

#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceTiles.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceVTiles.h"

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::analog;

//===----------------------------------------------------------------------===//
// analog-materialize 
//===----------------------------------------------------------------------===//

struct MaterializePipelineOptions
    : public PassPipelineOptions<MaterializePipelineOptions> {};

void mlir::analog::registerMaterializePipeline() {
  PassPipelineRegistration<MaterializePipelineOptions>(
      "analog-materialize",
      "Materialize tensors into analog matrix and vector IR",
      [](OpPassManager &pm,
         const MaterializePipelineOptions &) {
        pm.addPass(createMaterializeMatrixFromTensorPass());
        pm.addPass(createMaterializeVectorFromTensorPass());
      });
}

//===----------------------------------------------------------------------===//
// analog-partition pipeline
//===----------------------------------------------------------------------===//

struct PartitionPipelineOptions
    : public PassPipelineOptions<PartitionPipelineOptions> {

  Option<int64_t> tileRows{
      *this, "tile-rows",
      llvm::cl::desc("Number of rows per analog tile"),
      llvm::cl::init(16)};

  Option<int64_t> tileCols{
      *this, "tile-cols",
      llvm::cl::desc("Number of cols per analog tile"),
      llvm::cl::init(16)};
};

void mlir::analog::registerPartitionPipeline() {
  PassPipelineRegistration<PartitionPipelineOptions>(
      "analog-partition",
      "Partition analog matrices and vectors into tiles",
      [](OpPassManager &pm,
         const PartitionPipelineOptions &opts) {
        pm.addPass(createPartitionMatrixPass(
            opts.tileRows, opts.tileCols));
        pm.addPass(createPartitionVectorPass(
            opts.tileRows, opts.tileCols));
      });
}

//===----------------------------------------------------------------------===//
// analog-place pipeline 
//===----------------------------------------------------------------------===//

struct PlacePipelineOptions
    : public PassPipelineOptions<PlacePipelineOptions> {};

void mlir::analog::registerPlacePipeline() {
  PassPipelineRegistration<PlacePipelineOptions>(
      "analog-place",
      "Extract and place analog tiles and vector tiles",
      [](OpPassManager &pm,
         const PlacePipelineOptions &) {
        pm.addPass(createPlaceTilesPass());
        pm.addPass(createPlaceVTilesPass());
      });
}

