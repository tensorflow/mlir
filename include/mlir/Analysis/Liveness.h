//===- Liveness.h - Liveness analysis for MLIR ------------------*- C++ -*-===//
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
// This file contains an analysis for computing liveness information from a
// given top-level operation. The current version of the analysis uses a
// traditional algorithm to resolve detailed live-range information about all
// values within the specified regions. It is also possible to query liveness
// information on block level.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LIVENESS_H
#define MLIR_ANALYSIS_LIVENESS_H

#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class Block;
class Region;
class Value;
class Operation;

class LivenessBlockInfo;

/// Represents an analysis for computing liveness information from a
/// given top-level operation.
class Liveness {
public:
  using OperationIdMapT = llvm::DenseMap<Operation *, size_t>;
  using OperationListT = std::vector<Operation *>;
  using ValueMapT = llvm::DenseMap<Value *, llvm::BitVector>;
  using BlockMapT = llvm::DenseMap<Block *, LivenessBlockInfo>;
  using ValueSetT = llvm::SmallSetVector<Value *, 16>;

public:
  /// Creates a new Liveness analysis that computes liveness
  /// information for all associated regions.
  Liveness(Operation *op);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Resolves liveness info (if any) for the given value.
  /// This includes all operations in which the given value is live.
  OperationListT resolveLiveness(Value *value) const;

  /// Resolves liveness info (if any) for the block.
  const LivenessBlockInfo *getLiveness(Block *block) const;

  /// Returns a reference to a set containing live-in values.
  const ValueSetT &getLiveIn(Block *block) const;

  /// Returns a reference to a set containing live-out values.
  const ValueSetT &getLiveOut(Block *block) const;

  /// Returns true if the given operation (and its associated operand index)
  /// represent the last use of the given value.
  bool isLastUse(Value *value, Operation *operation, unsigned operandIndex);

  /// Dumps the liveness information in a human readable format.
  void dump() const;

  /// Dumps the liveness information to the given stream.
  void print(llvm::raw_ostream &os) const;

private:
  /// Initializes the internal mappings.
  void build(llvm::MutableArrayRef<Region> regions);

private:
  Operation *operation;

  /// Maps operations to unique identifiers (for fast bit-vector lookups).
  OperationIdMapT operationIdMapping;

  /// Ordered list of all operations that can be used to map bit indices
  /// (from bit-vector lookups) to actual operations.
  OperationListT operationList;

  /// Maps values to internal liveness information.
  ValueMapT valueMapping;

  /// Maps blocks to internal liveness information.
  BlockMapT blockMapping;
};

/// This class represents liveness information on block level.
class LivenessBlockInfo {
public:
  /// A typedef declaration of a value set.
  using ValueSetT = Liveness::ValueSetT;

public:
  /// Returns all values that are live at the beginning
  // of the block.
  const ValueSetT &in() const { return inValues; }

  /// Returns all values that are live at the end
  // of the block.
  const ValueSetT &out() const { return outValues; }

private:
  /// The set of all live in values.
  ValueSetT inValues;

  /// The set of all live out values.
  ValueSetT outValues;

  friend class Liveness;
};
} // end namespace mlir

#endif // MLIR_ANALYSIS_LIVENESS_H
