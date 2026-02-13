#include "analog-mlir/Dialect/Analog/Transforms/GroupedPasses.h"

#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceMatrices.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceVectors.h"

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

  Option<int64_t> arrayRows{
      *this, "array-rows",
      llvm::cl::desc("Number of rows per analog array"),
      llvm::cl::init(16)};

  Option<int64_t> arrayCols{
      *this, "array-cols",
      llvm::cl::desc("Number of cols per analog array"),
      llvm::cl::init(16)};
};

void mlir::analog::registerPartitionPipeline() {
  PassPipelineRegistration<PartitionPipelineOptions>(
      "analog-partition",
      "Partition analog matrices and vectors into arrays",
      [](OpPassManager &pm,
         const PartitionPipelineOptions &opts) {
        pm.addPass(createPartitionMatrixPass(
            opts.arrayRows, opts.arrayCols));
        pm.addPass(createPartitionVectorPass(
            opts.arrayRows, opts.arrayCols));
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
      "Extract and place analog arrays and vector arrays",
      [](OpPassManager &pm,
         const PlacePipelineOptions &) {
        pm.addPass(createPlaceMatricesPass());
        pm.addPass(createPlaceVectorsPass());
      });
}

