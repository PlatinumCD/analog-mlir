#include "analog-mlir/Dialect/Analog/Transforms/OpenMPDispatch.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"

#include <algorithm>
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

static void ensureOmpTerminator(Region &region, Location loc) {
  if (region.empty()) {
    region.push_back(new Block());
  }

  Block &block = region.front();
  if (!block.empty() && isa<omp::TerminatorOp>(block.back())) {
    return;
  }

  OpBuilder::atBlockEnd(&block).create<omp::TerminatorOp>(loc);
}

static void lowerDenseStageToOmpSections(MutableArrayRef<scf::ExecuteRegionOp> denseRegions,
                                         scf::ExecuteRegionOp joinBarrier) {
  if (denseRegions.empty()) {
    return;
  }

  Location denseLoc = denseRegions.front().getLoc();

  OpBuilder b(denseRegions.front());
  auto parallel = b.create<omp::ParallelOp>(denseLoc);
  parallel->setAttr("omp_dispatch", b.getStringAttr("dense_sections"));
  parallel->setAttr("dispatch_phase", b.getStringAttr("parallel_dense"));

  Block *parallelBlock = new Block();
  parallel.getRegion().push_back(parallelBlock);
  OpBuilder pb = OpBuilder::atBlockEnd(parallelBlock);

  auto sections = pb.create<omp::SectionsOp>(
      denseLoc, ValueRange{}, ValueRange{}, /*nowait=*/false, ValueRange{},
      ArrayAttr{}, /*private_needs_barrier=*/false,
      omp::ReductionModifierAttr{}, ValueRange{}, DenseBoolArrayAttr{},
      ArrayAttr{});
  sections->setAttr("dispatch_phase", b.getStringAttr("parallel_dense"));

  Block *sectionsBlock = new Block();
  sections.getRegion().push_back(sectionsBlock);
  OpBuilder sb = OpBuilder::atBlockEnd(sectionsBlock);

  for (scf::ExecuteRegionOp dense : denseRegions) {
    // Dense-resource values are consumed by later tensor-level ops outside the
    // OpenMP region. Provide a same-typed placeholder so SSA remains valid while
    // moving the dense resource pipeline into the first OpenMP body.
    if (dense->getNumResults() == 1) {
      auto resultTy = dyn_cast<RankedTensorType>(dense.getResult(0).getType());
      if (resultTy) {
        OpBuilder replBuilder(dense);
        auto empty = replBuilder.create<tensor::EmptyOp>(dense.getLoc(), resultTy.getShape(), resultTy.getElementType());
        dense.getResult(0).replaceAllUsesWith(empty.getResult());
      }
    }

    auto section = sb.create<omp::SectionOp>(dense.getLoc());
    section->setAttrs(dense->getAttrs());

    Block *sectionBlock = new Block();
    section.getRegion().push_back(sectionBlock);
    OpBuilder secb = OpBuilder::atBlockEnd(sectionBlock);
    dense->moveBefore(sectionBlock, sectionBlock->end());
    secb.create<omp::TerminatorOp>(dense.getLoc());
  }

  ensureOmpTerminator(sections.getRegion(), denseLoc);
  ensureOmpTerminator(parallel.getRegion(), denseLoc);

  if (joinBarrier) {
    joinBarrier.erase();
  }
}

static int64_t getCoreCount(ArrayRef<scf::ExecuteRegionOp> layerRegions) {
  int64_t maxCore = -1;
  for (scf::ExecuteRegionOp layer : layerRegions) {
    auto coreAttr = layer->getAttrOfType<IntegerAttr>("core_id");
    if (!coreAttr) {
      continue;
    }

    maxCore = std::max(maxCore, coreAttr.getInt());
  }
  return maxCore >= 0 ? (maxCore + 1) : 2;
}

static void lowerOrderedLayerRegionToOmpSections(scf::ExecuteRegionOp layer, int64_t coreCount) {
  Location loc = layer.getLoc();
  OpBuilder b(layer);

  // Keep SSA in the parent block by replacing the region result with a
  // same-typed placeholder before moving the region into OpenMP.
  if (layer->getNumResults() == 1) {
    auto resultTy = dyn_cast<RankedTensorType>(layer.getResult(0).getType());
    if (resultTy) {
      auto empty = b.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
      layer.getResult(0).replaceAllUsesWith(empty.getResult());
    }
  }

  auto parallel = b.create<omp::ParallelOp>(loc);
  parallel->setAttr("dispatch_phase", b.getStringAttr("ordered_layer"));
  parallel->setAttr("omp_dispatch", b.getStringAttr("ordered_layer_sections"));

  Block *parallelBlock = new Block();
  parallel.getRegion().push_back(parallelBlock);
  OpBuilder pb = OpBuilder::atBlockEnd(parallelBlock);

  auto sections = pb.create<omp::SectionsOp>(
      loc, ValueRange{}, ValueRange{}, /*nowait=*/false, ValueRange{},
      ArrayAttr{}, /*private_needs_barrier=*/false,
      omp::ReductionModifierAttr{}, ValueRange{}, DenseBoolArrayAttr{},
      ArrayAttr{});
  sections->setAttr("dispatch_phase", b.getStringAttr("ordered_layer"));

  Block *sectionsBlock = new Block();
  sections.getRegion().push_back(sectionsBlock);
  OpBuilder sb = OpBuilder::atBlockEnd(sectionsBlock);

  auto coreAttr = layer->getAttrOfType<IntegerAttr>("core_id");
  int64_t targetCore = coreAttr ? coreAttr.getInt() : 0;

  for (int64_t core = 0; core < coreCount; ++core) {
    auto section = sb.create<omp::SectionOp>(loc);
    section->setAttr("core_id", b.getI64IntegerAttr(core));
    section->setAttr("dispatch_phase", b.getStringAttr("ordered_layer"));
    section->setAttr("dispatch_scope", b.getStringAttr("private"));
    section->setAttr("layer_group", layer->getAttr("layer_group"));

    Block *sectionBlock = new Block();
    section.getRegion().push_back(sectionBlock);
    OpBuilder secb = OpBuilder::atBlockEnd(sectionBlock);

    if (core == targetCore) {
      layer->moveBefore(sectionBlock, sectionBlock->end());
      section->setAttr("group_kind", b.getStringAttr("layer"));
      section->setAttr("section_mode", b.getStringAttr("compute"));
    } else {
      section->setAttr("group_kind", b.getStringAttr("layer_wait"));
      section->setAttr("section_mode", b.getStringAttr("wait"));
    }

    secb.create<omp::TerminatorOp>(loc);
  }

  ensureOmpTerminator(sections.getRegion(), loc);
  ensureOmpTerminator(parallel.getRegion(), loc);
}

} // namespace

llvm::StringRef OpenMPDispatchPass::getArgument() const {
  return "analog-openmp-dispatch";
}

llvm::StringRef OpenMPDispatchPass::getDescription() const {
  return "Lower grouped layer metadata to OpenMP scheduling regions";
}

void OpenMPDispatchPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<omp::OpenMPDialect>();
  registry.insert<scf::SCFDialect>();
}

void OpenMPDispatchPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (func.empty()) {
    return;
  }

  Block &entry = func.getBody().front();

  SmallVector<scf::ExecuteRegionOp> denseRegions;
  SmallVector<scf::ExecuteRegionOp> layerRegions;
  scf::ExecuteRegionOp joinBarrier;

  for (Operation &op : llvm::make_early_inc_range(entry)) {
    auto region = dyn_cast<scf::ExecuteRegionOp>(&op);
    if (!region) {
      continue;
    }

    auto kind = region->getAttrOfType<StringAttr>("group_kind");
    if (!kind) {
      continue;
    }

    if (kind.getValue() == "dense_resource") {
      denseRegions.push_back(region);
    } else if (kind.getValue() == "layer") {
      layerRegions.push_back(region);
    } else if (kind.getValue() == "barrier") {
      joinBarrier = region;
    }
  }

  if (!denseRegions.empty()) {
    lowerDenseStageToOmpSections(denseRegions, joinBarrier);
  }

  int64_t coreCount = getCoreCount(layerRegions);
  for (scf::ExecuteRegionOp layer : layerRegions) {
    if (layer && layer->getBlock()) {
      lowerOrderedLayerRegionToOmpSections(layer, coreCount);
    }
  }
}

std::unique_ptr<mlir::Pass> createOpenMPDispatchPass() {
  return std::make_unique<OpenMPDispatchPass>();
}

} // namespace analog
} // namespace mlir
