//===- ConvertControlFlowToCFG.h - Pass entrypoint --------------*- C++ -*-===//
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

#ifndef MLIR_CONVERSION_CONTROLFLOWTOCFG_CONVERTCONTROLFLOWTOCFG_H_
#define MLIR_CONVERSION_CONTROLFLOWTOCFG_CONVERTCONTROLFLOWTOCFG_H_

#include <memory>
#include <vector>

namespace mlir {
class FuncOp;
class FunctionPassBase;
struct LogicalResult;
class MLIRContext;
class RewritePattern;

// Owning list of rewriting patterns.
using OwningRewritePatternList = std::vector<std::unique_ptr<RewritePattern>>;

/// Collect a set of patterns to lower from loop.for, loop.if, and
/// loop.terminator to CFG operations within the Standard dialect, in particular
/// convert structured control flow into CFG branch-based control flow.
void populateLoopToStdConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);

/// Creates a pass to convert loop.for, loop.if and loop.terminator ops to CFG.
FunctionPassBase *createConvertToCFGPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CONTROLFLOWTOCFG_CONVERTCONTROLFLOWTOCFG_H_
