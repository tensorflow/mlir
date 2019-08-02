//===- Block.cpp - MLIR Block Class ---------------------------------------===//
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

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// BlockArgument
//===----------------------------------------------------------------------===//

/// Returns the number of this argument.
unsigned BlockArgument::getArgNumber() {
  // Arguments are not stored in place, so we have to find it within the list.
  auto argList = getOwner()->getArguments();
  return std::distance(argList.begin(), llvm::find(argList, this));
}

//===----------------------------------------------------------------------===//
// Block
//===----------------------------------------------------------------------===//

Block::~Block() {
  assert(!verifyInstOrder() && "Expected valid operation ordering.");
  clear();

  for (auto *arg : arguments)
    if (!arg->use_empty())
      arg->user_begin()->dump();

  llvm::DeleteContainerPointers(arguments);
}

Region *Block::getParent() { return parentValidInstOrderPair.getPointer(); }

/// Returns the closest surrounding operation that contains this block or
/// nullptr if this is a top-level operation block.
Operation *Block::getContainingOp() {
  return getParent() ? getParent()->getContainingOp() : nullptr;
}

/// Return if this block is the entry block in the parent region.
bool Block::isEntryBlock() { return this == &getParent()->front(); }

/// Insert this block (which must not already be in a region) right before the
/// specified block.
void Block::insertBefore(Block *block) {
  assert(!getParent() && "already inserted into a block!");
  assert(block->getParent() && "cannot insert before a block without a parent");
  block->getParent()->getBlocks().insert(Region::iterator(block), this);
}

/// Unlink this Block from its parent Region and delete it.
void Block::erase() {
  assert(getParent() && "Block has no parent");
  getParent()->getBlocks().erase(this);
}

/// Returns 'op' if 'op' lies in this block, or otherwise finds the
/// ancestor operation of 'op' that lies in this block. Returns nullptr if
/// the latter fails.
Operation *Block::findAncestorInstInBlock(Operation &op) {
  // Traverse up the operation hierarchy starting from the owner of operand to
  // find the ancestor operation that resides in the block of 'forInst'.
  auto *currInst = &op;
  while (currInst->getBlock() != this) {
    currInst = currInst->getParentOp();
    if (!currInst)
      return nullptr;
  }
  return currInst;
}

/// This drops all operand uses from operations within this block, which is
/// an essential step in breaking cyclic dependences between references when
/// they are to be deleted.
void Block::dropAllReferences() {
  for (Operation &i : *this)
    i.dropAllReferences();
}

void Block::dropAllDefinedValueUses() {
  for (auto *arg : getArguments())
    arg->dropAllUses();
  for (auto &op : *this)
    op.dropAllDefinedValueUses();
  dropAllUses();
}

/// Returns true if the ordering of the child operations is valid, false
/// otherwise.
bool Block::isInstOrderValid() { return parentValidInstOrderPair.getInt(); }

/// Invalidates the current ordering of operations.
void Block::invalidateInstOrder() {
  // Validate the current ordering.
  assert(!verifyInstOrder());
  parentValidInstOrderPair.setInt(false);
}

/// Verifies the current ordering of child operations. Returns false if the
/// order is valid, true otherwise.
bool Block::verifyInstOrder() {
  // The order is already known to be invalid.
  if (!isInstOrderValid())
    return false;
  // The order is valid if there are less than 2 operations.
  if (operations.empty() || std::next(operations.begin()) == operations.end())
    return false;

  Operation *prev = nullptr;
  for (auto &i : *this) {
    // The previous operation must have a smaller order index than the next as
    // it appears earlier in the list.
    if (prev && prev->orderIndex >= i.orderIndex)
      return true;
    prev = &i;
  }
  return false;
}

/// Recomputes the ordering of child operations within the block.
void Block::recomputeInstOrder() {
  parentValidInstOrderPair.setInt(true);

  // TODO(riverriddle) Have non-congruent indices to reduce the number of times
  // an insert invalidates the list.
  unsigned orderIndex = 0;
  for (auto &op : *this)
    op.orderIndex = orderIndex++;
}

//===----------------------------------------------------------------------===//
// Argument list management.
//===----------------------------------------------------------------------===//

BlockArgument *Block::addArgument(Type type) {
  auto *arg = new BlockArgument(type, this);
  arguments.push_back(arg);
  return arg;
}

/// Add one argument to the argument list for each type specified in the list.
auto Block::addArguments(ArrayRef<Type> types)
    -> llvm::iterator_range<args_iterator> {
  arguments.reserve(arguments.size() + types.size());
  auto initialSize = arguments.size();
  for (auto type : types) {
    addArgument(type);
  }
  return {arguments.data() + initialSize, arguments.data() + arguments.size()};
}

void Block::eraseArgument(unsigned index, bool updatePredTerms) {
  assert(index < arguments.size());

  // Delete the argument.
  delete arguments[index];
  arguments.erase(arguments.begin() + index);

  // If we aren't updating predecessors, there is nothing left to do.
  if (!updatePredTerms)
    return;

  // Erase this argument from each of the predecessor's terminator.
  for (auto predIt = pred_begin(), predE = pred_end(); predIt != predE;
       ++predIt) {
    auto *predTerminator = (*predIt)->getTerminator();
    predTerminator->eraseSuccessorOperand(predIt.getSuccessorIndex(), index);
  }
}

//===----------------------------------------------------------------------===//
// Terminator management
//===----------------------------------------------------------------------===//

/// Get the terminator operation of this block. This function asserts that
/// the block has a valid terminator operation.
Operation *Block::getTerminator() {
  assert(!empty() && !back().isKnownNonTerminator());
  return &back();
}

/// Return true if this block has no predecessors.
bool Block::hasNoPredecessors() { return pred_begin() == pred_end(); }

// Indexed successor access.
unsigned Block::getNumSuccessors() {
  return empty() ? 0 : back().getNumSuccessors();
}

Block *Block::getSuccessor(unsigned i) {
  assert(i < getNumSuccessors());
  return getTerminator()->getSuccessor(i);
}

/// If this block has exactly one predecessor, return it.  Otherwise, return
/// null.
///
/// Note that multiple edges from a single block (e.g. if you have a cond
/// branch with the same block as the true/false destinations) is not
/// considered to be a single predecessor.
Block *Block::getSinglePredecessor() {
  auto it = pred_begin();
  if (it == pred_end())
    return nullptr;
  auto *firstPred = *it;
  ++it;
  return it == pred_end() ? firstPred : nullptr;
}

//===----------------------------------------------------------------------===//
// Operation Walkers
//===----------------------------------------------------------------------===//

void Block::walk(llvm::function_ref<void(Operation *)> callback) {
  walk(begin(), end(), callback);
}

/// Walk the operations in the specified [begin, end) range of this block,
/// calling the callback for each operation.
void Block::walk(Block::iterator begin, Block::iterator end,
                 llvm::function_ref<void(Operation *)> callback) {
  for (auto &op : llvm::make_early_inc_range(llvm::make_range(begin, end)))
    op.walk(callback);
}

//===----------------------------------------------------------------------===//
// Other
//===----------------------------------------------------------------------===//

/// Split the block into two blocks before the specified operation or
/// iterator.
///
/// Note that all operations BEFORE the specified iterator stay as part of
/// the original basic block, and the rest of the operations in the original
/// block are moved to the new block, including the old terminator.  The
/// original block is left without a terminator.
///
/// The newly formed Block is returned, and the specified iterator is
/// invalidated.
Block *Block::splitBlock(iterator splitBefore) {
  // Start by creating a new basic block, and insert it immediate after this
  // one in the containing region.
  auto newBB = new Block();
  getParent()->getBlocks().insert(std::next(Region::iterator(this)), newBB);

  // Move all of the operations from the split point to the end of the region
  // into the new block.
  newBB->getOperations().splice(newBB->end(), getOperations(), splitBefore,
                                end());
  return newBB;
}

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

Block *PredecessorIterator::unwrap(BlockOperand &value) {
  return value.getOwner()->getBlock();
}

/// Get the successor number in the predecessor terminator.
unsigned PredecessorIterator::getSuccessorIndex() const {
  return I->getOperandNumber();
}
