//===- Liveness.cpp - Liveness analysis for MLIR --------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Implementation of the liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Builds and holds block information during the construction phase.
struct BlockInfoBuilder {
  using ValueSetT = Liveness::ValueSetT;

  /// Fills the block builder with initial liveness information.
  void build(Block *block) {
    // Store block for further processing.
    this->block = block;

    // Mark all block arguments (phis) as defined.
    for (BlockArgument *argument : block->getArguments())
      defValues.insert(argument);

    // Check all result values and whether their uses
    // are inside this block or not (see outValues).
    for (Operation &operation : *block)
      for (Value *result : operation.getResults()) {
        // Mark as defined
        defValues.insert(result);

        // Check whether this value will be in the outValues
        // set (its uses escape this block). Due to the SSA
        // properties of the program, the uses must occur after
        // the definition. Therefore, we do not have to check
        // additional conditions to detect an escaping value.
        for (OpOperand &use : result->getUses())
          if (use.getOwner()->getBlock() != block) {
            outValues.insert(result);
            break;
          }
      }

    // Check all operations for used operands.
    for (Operation &operation : block->getOperations())
      for (Value *operand : operation.getOperands())
        useValues.insert(operand);
  }

  /// Updates live-in information of the current block.
  /// To do so it uses the default liveness-computation formula:
  /// newIn = use union out \ def.
  /// The methods returns true, if the set has changed (newIn != in),
  /// false otherwise.
  bool updateLiveIn() {
    ValueSetT newIn = useValues;
    newIn.set_union(outValues);
    newIn.set_subtract(defValues);

    if (newIn.size() == inValues.size())
      return false;

    inValues = newIn;
    return true;
  }

  /// Updates live-out information of the current block.
  /// It iterates over all successors and unifies their live-in
  /// values with the current live-out values.
  template <typename SourceT>
  void updateLiveOut(SourceT &source) {
    for (Block *succ : block->getSuccessors()) {
      BlockInfoBuilder &builder = source[succ];
      outValues.set_union(builder.inValues);
    }
  }

  /// Gets a set of all values that are either live-in, live-out or defined.
  ValueSetT resolveValues() const {
    ValueSetT result = inValues;
    result.set_union(outValues);
    result.set_union(defValues);
    return result;
  }

  /// Gets the start operation for the given value
  /// (must be referenced in this block).
  Operation *getStartOperation(Value *value) const {
    Operation *definingOp = value->getDefiningOp();
    // The given value is either live-in or is defined
    // in the scope of this block.
    if (inValues.count(value) || !definingOp)
      return &block->front();
    return definingOp;
  }

  /// Gets the end operation for the given value using the start operation
  /// provided (must be referenced in this block).
  Operation *getEndOperation(Value *value, Operation *startOperation) const {
    // The given value is either dying in this block or live-out.
    if (outValues.count(value))
      return &block->back();

    // Resolve the last operation (must exist by definition).
    Operation *endOperation = startOperation;
    for (OpOperand &use : value->getUses()) {
      Operation *useOperation = use.getOwner();
      // Check whether the use is in our block and after
      // the current end operation.
      if (useOperation->getBlock() == block &&
          endOperation->isBeforeInBlock(useOperation))
        endOperation = useOperation;
    }
    return endOperation;
  }

  /// The current block.
  Block *block;

  /// The set of all live in values.
  ValueSetT inValues;

  /// The set of all live out values.
  ValueSetT outValues;

  /// The set of all defined values.
  ValueSetT defValues;

  /// The set of all used values.
  ValueSetT useValues;
};

/// Builds the internal liveness block mapping.
static void
buildBlockMapping(MutableArrayRef<Region> regions,
                  llvm::DenseMap<Block *, BlockInfoBuilder> &builders) {
  llvm::SetVector<Block *> toProcess;

  // Initialize all block structures
  for (Region &region : regions)
    for (Block &block : region) {
      auto &blockBuilder = builders[&block];
      blockBuilder.build(&block);

      if (blockBuilder.updateLiveIn())
        toProcess.insert(block.pred_begin(), block.pred_end());
    }

  // Propagate the in and out-value sets (fixpoint iteration)
  while (!toProcess.empty()) {
    Block *current = toProcess.pop_back_val();
    BlockInfoBuilder &builder = builders[current];

    // Update the current out values.
    builder.updateLiveOut(builders);

    // Compute (potentially) updated live in values.
    if (builder.updateLiveIn())
      toProcess.insert(current->pred_begin(), current->pred_end());
  }
}

/// Builds the internal operation mapping.
static void buildOperationMapping(MutableArrayRef<Region> regions,
                                  Liveness::OperationIdMapT &operationIdMapping,
                                  Liveness::OperationListT &operationList) {
  // Build unique id mapping.
  // TODO: we might want to exploit the presence of `closed` regions.
  for (Region &region : regions)
    for (Block &block : region)
      for (Operation &operation : block) {
        operationIdMapping[&operation] = operationList.size();
        operationList.push_back(&operation);
      }
}

/// Builds the internal liveness value mapping.
static void
buildValueMapping(llvm::DenseMap<Block *, BlockInfoBuilder> &builders,
                  Liveness::OperationIdMapT &operationIdMapping,
                  Liveness::ValueMapT &valueMapping) {
  for (auto &entry : builders) {
    BlockInfoBuilder &builder = entry.second;

    // Iterate over all values.
    for (Value *value : builder.resolveValues()) {
      // Resolve start and end operation for the current block.
      Operation *startOperation = builder.getStartOperation(value);
      Operation *endOperation = builder.getEndOperation(value, startOperation);

      // Resolve unique operation ids and register live range.
      size_t startId = operationIdMapping[startOperation];
      size_t endId = operationIdMapping[endOperation];

      llvm::BitVector &bitSet = valueMapping[value];
      bitSet.resize(operationIdMapping.size());
      bitSet.set(startId, endId + 1);
    }
  }
}

//===----------------------------------------------------------------------===//
// Liveness
//===----------------------------------------------------------------------===//

/// Creates a new Liveness analysis that computes liveness
/// information for all associated regions.
Liveness::Liveness(Operation *op) : operation(op) { build(op->getRegions()); }

/// Initializes the internal mappings.
void Liveness::build(MutableArrayRef<Region> regions) {

  // Build internal block mapping.
  DenseMap<Block *, BlockInfoBuilder> builders;
  buildBlockMapping(regions, builders);

  // Build internal value-id mapping.
  buildOperationMapping(regions, operationIdMapping, operationList);

  // Build internal value mapping.
  buildValueMapping(builders, operationIdMapping, valueMapping);

  // Store internal block data.
  for (auto &entry : builders) {
    BlockInfoBuilder &builder = entry.second;
    LivenessBlockInfo &info = blockMapping[entry.first];

    info.inValues = std::move(builder.inValues);
    info.outValues = std::move(builder.outValues);
  }
}

/// Gets liveness info (if any) for the given value.
Liveness::OperationListT Liveness::resolveLiveness(Value *value) const {
  OperationListT result;
  auto it = valueMapping.find(value);

  // No value entry found.
  if (it == valueMapping.end())
    return result;

  // Iterate over all active bits and resolve the corresponding operation.
  result.reserve(it->second.size());
  for (auto bit : it->second.set_bits())
    result.push_back(operationList[bit]);
  return result;
}

/// Gets liveness info (if any) for the block.
const LivenessBlockInfo *Liveness::getLiveness(Block *block) const {
  auto it = blockMapping.find(block);
  return it == blockMapping.end() ? nullptr : &it->second;
}

/// Returns a reference to a set containing live-in values.
const Liveness::ValueSetT &Liveness::getLiveIn(Block *block) const {
  return getLiveness(block)->in();
}

/// Returns a reference to a set containing live-out values.
const Liveness::ValueSetT &Liveness::getLiveOut(Block *block) const {
  return getLiveness(block)->out();
}

/// Returns true if the given operation represent the last use of the
/// given value.
bool Liveness::isLastUse(Value *value, Operation *operation) {
  Block *block = operation->getBlock();
  const ValueSetT &liveOut = getLiveOut(block);
  // The given value escapes the associated block.
  if (liveOut.count(value))
    return false;

  size_t id = operationIdMapping[operation];
  llvm::BitVector &bitSet = valueMapping[value];
  // If the given operation is not the last one in this block and
  // the value is still alive after this operation...
  if (&block->back() != operation && bitSet.test(id + 1))
    return false;

  // The operation dies at this point.
  return true;
}

/// Dumps the liveness information in a human readable format.
void Liveness::dump() const { print(llvm::errs()); }

/// Dumps the liveness information to the given stream.
void Liveness::print(raw_ostream &os) const {
  os << "// ---- Liveness -----\n";

  // Build a unique block mapping for testing purposes.
  DenseMap<Block *, size_t> blockIds;
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      blockIds[&block] = blockIds.size();

  // Local printing helper to create desired output.
  auto printValueRef = [&](Value *value) {
    if (Operation *defOp = value->getDefiningOp())
      os << "val_" << defOp->getName();
    else {
      auto blockArg = cast<BlockArgument>(value);
      os << "arg" << blockArg->getArgNumber() << "@"
         << blockIds[blockArg->getOwner()];
    }
    os << " ";
  };

  // Dump information about in and out values.
  for (Region &region : operation->getRegions())
    for (Block &block : region) {
      os << "// - Block: " << blockIds[&block] << "\n";
      auto liveness = getLiveness(&block);
      os << "// --- LiveIn: ";
      for (Value *liveIn : liveness->inValues)
        printValueRef(liveIn);
      os << "\n// --- LiveOut: ";
      for (Value *liveOut : liveness->outValues)
        printValueRef(liveOut);
      os << "\n";

      // Print liveness intervals.
      os << "// --- BeginLiveness";
      for (Operation &op : block) {
        if (op.getNumResults() < 1)
          continue;
        os << "\n";
        for (Value *result : op.getResults()) {
          os << "// ";
          printValueRef(result);
          os << ":";
          auto liveOperations = resolveLiveness(result);
          for (Operation *operation : liveOperations) {
            os << "\n//     ";
            operation->print(os);
          }
        }
      }
      os << "\n// --- EndLiveness\n";
    }
  os << "// -------------------\n";
}
