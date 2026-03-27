#include "analog-mlir/Dialect/Analog/Transforms/GroupedPasses.h"

#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/IdentifyRecurrentPatterns.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceMatrices.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceVectors.h"
#include "analog-mlir/Dialect/Analog/Transforms/PrepareRNNForAnalog.h"
#include "analog-mlir/Dialect/Analog/Transforms/PrepareRNNCellForAnalog.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv1DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv2DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteGroupedConv2DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv3DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/ExecuteArray.h"
#include "analog-mlir/Dialect/Analog/Transforms/ReduceResults.h"
#include "analog-mlir/Dialect/Analog/Transforms/ReplaceMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/IsolateLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/DispatchWeights.h"
#include "analog-mlir/Dialect/Analog/Transforms/DispatchLayers.h"

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::analog;

//===----------------------------------------------------------------------===//
// analog-rewrite-conv-to-matmul
//===----------------------------------------------------------------------===//

struct RewriteConvToMatmulPipelineOptions
    : public PassPipelineOptions<RewriteConvToMatmulPipelineOptions> {};

void mlir::analog::registerRewriteConvToMatmulPipeline() {
  PassPipelineRegistration<RewriteConvToMatmulPipelineOptions>(
      "analog-rewrite-conv-to-matmul",
      "Rewrite supported conv3d, conv2d, grouped conv2d, and conv1d ops into a matmul-oriented form",
      [](OpPassManager &pm,
         const RewriteConvToMatmulPipelineOptions &) {
        OpPassManager &funcPM = pm.nest<func::FuncOp>();
        funcPM.addPass(createRewriteConv2DToMatmulPass());
        funcPM.addPass(createRewriteGroupedConv2DToMatmulPass());
        funcPM.addPass(createRewriteConv1DToMatmulPass());
        funcPM.addPass(createRewriteConv3DToMatmulPass());
      });
}

//===----------------------------------------------------------------------===//
// analog-rewrite-recurrent-to-matmul
//===----------------------------------------------------------------------===//

struct RewriteRecurrentToMatmulPipelineOptions
    : public PassPipelineOptions<RewriteRecurrentToMatmulPipelineOptions> {};

void mlir::analog::registerRewriteRecurrentToMatmulPipeline() {
  PassPipelineRegistration<RewriteRecurrentToMatmulPipelineOptions>(
      "analog-rewrite-recurrent-to-matmul",
      "Identify supported recurrent patterns and prepare RNN cells for matmul-oriented lowering",
      [](OpPassManager &pm,
         const RewriteRecurrentToMatmulPipelineOptions &) {
        OpPassManager &funcPM = pm.nest<func::FuncOp>();
        funcPM.addPass(createIdentifyRecurrentPatternsPass());
        funcPM.addPass(createPrepareRNNForAnalogPass());
        funcPM.addPass(createPrepareRNNCellForAnalogPass());
      });
}

//===----------------------------------------------------------------------===//
// analog-materialize-and-place
//===----------------------------------------------------------------------===//

struct MaterializeAndPlacePipelineOptions
    : public PassPipelineOptions<MaterializeAndPlacePipelineOptions> {

  Option<int64_t> arrayRows{
      *this, "array-rows",
      llvm::cl::desc("Number of rows per analog array"),
      llvm::cl::init(16)};

  Option<int64_t> arrayCols{
      *this, "array-cols",
      llvm::cl::desc("Number of cols per analog array"),
      llvm::cl::init(16)};
};

void mlir::analog::registerMaterializeAndPlacePipeline() {
  PassPipelineRegistration<MaterializeAndPlacePipelineOptions>(
      "analog-materialize-and-place",
      "Materialize analog tensors, partition them into arrays, and place them",
      [](OpPassManager &pm,
         const MaterializeAndPlacePipelineOptions &opts) {
        OpPassManager &funcPM = pm.nest<func::FuncOp>();
        funcPM.addPass(createMaterializeMatrixFromTensorPass());
        funcPM.addPass(createMaterializeVectorFromTensorPass());
        funcPM.addPass(createPartitionMatrixPass(
            opts.arrayRows, opts.arrayCols));
        funcPM.addPass(createPartitionVectorPass(
            opts.arrayRows, opts.arrayCols));
        funcPM.addPass(createPlaceMatricesPass());
        funcPM.addPass(createPlaceVectorsPass());
      });
}

//===----------------------------------------------------------------------===//
// analog-execute-and-replace
//===----------------------------------------------------------------------===//

struct ExecuteAndReplacePipelineOptions
    : public PassPipelineOptions<ExecuteAndReplacePipelineOptions> {};

void mlir::analog::registerExecuteAndReplacePipeline() {
  PassPipelineRegistration<ExecuteAndReplacePipelineOptions>(
      "analog-execute-and-replace",
      "Execute placed analog arrays, reduce results, and replace source matmuls",
      [](OpPassManager &pm,
         const ExecuteAndReplacePipelineOptions &) {
        OpPassManager &funcPM = pm.nest<func::FuncOp>();
        funcPM.addPass(createExecuteArrayPass());
        funcPM.addPass(createReduceResultsPass());
        funcPM.addPass(createReplaceMatmulPass());
      });
}

//===----------------------------------------------------------------------===//
// analog-dispatch-runtime
//===----------------------------------------------------------------------===//

struct DispatchRuntimePipelineOptions
    : public PassPipelineOptions<DispatchRuntimePipelineOptions> {};

void mlir::analog::registerDispatchRuntimePipeline() {
  PassPipelineRegistration<DispatchRuntimePipelineOptions>(
      "analog-dispatch-runtime",
      "Isolate analog routines and create weight/layer runtime dispatch entrypoints",
      [](OpPassManager &pm,
         const DispatchRuntimePipelineOptions &) {
        OpPassManager &funcPM = pm.nest<func::FuncOp>();
        funcPM.addPass(createIsolateLayersPass());
        pm.addPass(createDispatchWeightsPass());
        pm.addPass(createDispatchLayersPass());
      });
}
