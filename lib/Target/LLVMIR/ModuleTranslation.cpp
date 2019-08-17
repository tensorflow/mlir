//===- ModuleTranslation.cpp - MLIR to LLVM conversion --------------------===//
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
// This file implements the translation between an MLIR LLVM dialect module and
// the corresponding LLVMIR module. It only handles core LLVM IR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace mlir {
namespace LLVM {

// Convert an MLIR function type to LLVM IR.  Arguments of the function must of
// MLIR LLVM IR dialect types.  Use `loc` as a location when reporting errors.
// Return nullptr on errors.
static llvm::FunctionType *convertFunctionType(llvm::LLVMContext &llvmContext,
                                               FunctionType type, Location loc,
                                               bool isVarArgs) {
  assert(type && "expected non-null type");
  if (type.getNumResults() > 1)
    return emitError(loc, "LLVM functions can only have 0 or 1 result"),
           nullptr;

  SmallVector<llvm::Type *, 8> argTypes;
  argTypes.reserve(type.getNumInputs());
  for (auto t : type.getInputs()) {
    auto wrappedLLVMType = t.dyn_cast<LLVM::LLVMType>();
    if (!wrappedLLVMType)
      return emitError(loc, "non-LLVM function argument type"), nullptr;
    argTypes.push_back(wrappedLLVMType.getUnderlyingType());
  }

  if (type.getNumResults() == 0)
    return llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), argTypes,
                                   isVarArgs);

  auto wrappedResultType = type.getResult(0).dyn_cast<LLVM::LLVMType>();
  if (!wrappedResultType)
    return emitError(loc, "non-LLVM function result"), nullptr;

  return llvm::FunctionType::get(wrappedResultType.getUnderlyingType(),
                                 argTypes, isVarArgs);
}

// Create an LLVM IR constant of `llvmType` from the MLIR attribute `attr`.
// This currently supports integer, floating point, splat and dense element
// attributes and combinations thereof.  In case of error, report it to `loc`
// and return nullptr.
llvm::Constant *ModuleTranslation::getLLVMConstant(llvm::Type *llvmType,
                                                   Attribute attr,
                                                   Location loc) {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return llvm::ConstantInt::get(llvmType, intAttr.getValue());
  if (auto floatAttr = attr.dyn_cast<FloatAttr>())
    return llvm::ConstantFP::get(llvmType, floatAttr.getValue());
  if (auto funcAttr = attr.dyn_cast<SymbolRefAttr>())
    return functionMapping.lookup(funcAttr.getValue());
  if (auto boolAttr = attr.dyn_cast<BoolAttr>())
    return llvm::ConstantInt::get(llvmType, boolAttr.getValue() ? 1 : 0);
  if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
    auto *vectorType = cast<llvm::VectorType>(llvmType);
    auto *child = getLLVMConstant(vectorType->getElementType(),
                                  splatAttr.getSplatValue(), loc);
    return llvm::ConstantVector::getSplat(vectorType->getNumElements(), child);
  }
  if (auto denseAttr = attr.dyn_cast<DenseElementsAttr>()) {
    auto *vectorType = cast<llvm::VectorType>(llvmType);
    SmallVector<llvm::Constant *, 8> constants;
    uint64_t numElements = vectorType->getNumElements();
    constants.reserve(numElements);
    for (auto n : denseAttr.getAttributeValues()) {
      constants.push_back(
          getLLVMConstant(vectorType->getElementType(), n, loc));
      if (!constants.back())
        return nullptr;
    }
    return llvm::ConstantVector::get(constants);
  }
  if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
    return llvm::ConstantDataArray::get(
        llvmModule->getContext(), ArrayRef<char>{stringAttr.getValue().data(),
                                                 stringAttr.getValue().size()});
  }
  emitError(loc, "unsupported constant value");
  return nullptr;
}

// Convert MLIR integer comparison predicate to LLVM IR comparison predicate.
static llvm::CmpInst::Predicate getLLVMCmpPredicate(ICmpPredicate p) {
  switch (p) {
  case LLVM::ICmpPredicate::eq:
    return llvm::CmpInst::Predicate::ICMP_EQ;
  case LLVM::ICmpPredicate::ne:
    return llvm::CmpInst::Predicate::ICMP_NE;
  case LLVM::ICmpPredicate::slt:
    return llvm::CmpInst::Predicate::ICMP_SLT;
  case LLVM::ICmpPredicate::sle:
    return llvm::CmpInst::Predicate::ICMP_SLE;
  case LLVM::ICmpPredicate::sgt:
    return llvm::CmpInst::Predicate::ICMP_SGT;
  case LLVM::ICmpPredicate::sge:
    return llvm::CmpInst::Predicate::ICMP_SGE;
  case LLVM::ICmpPredicate::ult:
    return llvm::CmpInst::Predicate::ICMP_ULT;
  case LLVM::ICmpPredicate::ule:
    return llvm::CmpInst::Predicate::ICMP_ULE;
  case LLVM::ICmpPredicate::ugt:
    return llvm::CmpInst::Predicate::ICMP_UGT;
  case LLVM::ICmpPredicate::uge:
    return llvm::CmpInst::Predicate::ICMP_UGE;
  }
  llvm_unreachable("incorrect comparison predicate");
}

static llvm::CmpInst::Predicate getLLVMCmpPredicate(FCmpPredicate p) {
  switch (p) {
  case LLVM::FCmpPredicate::_false:
    return llvm::CmpInst::Predicate::FCMP_FALSE;
  case LLVM::FCmpPredicate::oeq:
    return llvm::CmpInst::Predicate::FCMP_OEQ;
  case LLVM::FCmpPredicate::ogt:
    return llvm::CmpInst::Predicate::FCMP_OGT;
  case LLVM::FCmpPredicate::oge:
    return llvm::CmpInst::Predicate::FCMP_OGE;
  case LLVM::FCmpPredicate::olt:
    return llvm::CmpInst::Predicate::FCMP_OLT;
  case LLVM::FCmpPredicate::ole:
    return llvm::CmpInst::Predicate::FCMP_OLE;
  case LLVM::FCmpPredicate::one:
    return llvm::CmpInst::Predicate::FCMP_ONE;
  case LLVM::FCmpPredicate::ord:
    return llvm::CmpInst::Predicate::FCMP_ORD;
  case LLVM::FCmpPredicate::ueq:
    return llvm::CmpInst::Predicate::FCMP_UEQ;
  case LLVM::FCmpPredicate::ugt:
    return llvm::CmpInst::Predicate::FCMP_UGT;
  case LLVM::FCmpPredicate::uge:
    return llvm::CmpInst::Predicate::FCMP_UGE;
  case LLVM::FCmpPredicate::ult:
    return llvm::CmpInst::Predicate::FCMP_ULT;
  case LLVM::FCmpPredicate::ule:
    return llvm::CmpInst::Predicate::FCMP_ULE;
  case LLVM::FCmpPredicate::une:
    return llvm::CmpInst::Predicate::FCMP_UNE;
  case LLVM::FCmpPredicate::uno:
    return llvm::CmpInst::Predicate::FCMP_UNO;
  case LLVM::FCmpPredicate::_true:
    return llvm::CmpInst::Predicate::FCMP_TRUE;
  }
  llvm_unreachable("incorrect comparison predicate");
}

// A helper to look up remapped operands in the value remapping table.
template <typename Range>
SmallVector<llvm::Value *, 8> ModuleTranslation::lookupValues(Range &&values) {
  SmallVector<llvm::Value *, 8> remapped;
  remapped.reserve(llvm::size(values));
  for (Value *v : values) {
    remapped.push_back(valueMapping.lookup(v));
  }
  return remapped;
}

// Given a single MLIR operation, create the corresponding LLVM IR operation
// using the `builder`.  LLVM IR Builder does not have a generic interface so
// this has to be a long chain of `if`s calling different functions with a
// different number of arguments.
LogicalResult ModuleTranslation::convertOperation(Operation &opInst,
                                                  llvm::IRBuilder<> &builder) {
  auto extractPosition = [](ArrayAttr attr) {
    SmallVector<unsigned, 4> position;
    position.reserve(attr.size());
    for (Attribute v : attr)
      position.push_back(v.cast<IntegerAttr>().getValue().getZExtValue());
    return position;
  };

#include "mlir/LLVMIR/LLVMConversions.inc"

  // Emit function calls.  If the "callee" attribute is present, this is a
  // direct function call and we also need to look up the remapped function
  // itself.  Otherwise, this is an indirect call and the callee is the first
  // operand, look it up as a normal value.  Return the llvm::Value representing
  // the function result, which may be of llvm::VoidTy type.
  auto convertCall = [this, &builder](Operation &op) -> llvm::Value * {
    auto operands = lookupValues(op.getOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    if (auto attr = op.getAttrOfType<SymbolRefAttr>("callee")) {
      return builder.CreateCall(functionMapping.lookup(attr.getValue()),
                                operandsRef);
    } else {
      return builder.CreateCall(operandsRef.front(), operandsRef.drop_front());
    }
  };

  // Emit calls.  If the called function has a result, remap the corresponding
  // value.  Note that LLVM IR dialect CallOp has either 0 or 1 result.
  if (isa<LLVM::CallOp>(opInst)) {
    llvm::Value *result = convertCall(opInst);
    if (opInst.getNumResults() != 0) {
      valueMapping[opInst.getResult(0)] = result;
      return success();
    }
    // Check that LLVM call returns void for 0-result functions.
    return success(result->getType()->isVoidTy());
  }

  // Emit branches.  We need to look up the remapped blocks and ignore the block
  // arguments that were transformed into PHI nodes.
  if (auto brOp = dyn_cast<LLVM::BrOp>(opInst)) {
    builder.CreateBr(blockMapping[brOp.getSuccessor(0)]);
    return success();
  }
  if (auto condbrOp = dyn_cast<LLVM::CondBrOp>(opInst)) {
    builder.CreateCondBr(valueMapping.lookup(condbrOp.getOperand(0)),
                         blockMapping[condbrOp.getSuccessor(0)],
                         blockMapping[condbrOp.getSuccessor(1)]);
    return success();
  }

  // Emit addressof.  We need to look up the global value referenced by the
  // operation and store it in the MLIR-to-LLVM value mapping.  This does not
  // emit any LLVM instruction.
  if (auto addressOfOp = dyn_cast<LLVM::AddressOfOp>(opInst)) {
    LLVM::GlobalOp global = addressOfOp.getGlobal();
    // The verifier should not have allowed this.
    assert(global && "referencing an undefined global");

    valueMapping[addressOfOp.getResult()] = globalsMapping.lookup(global);
    return success();
  }

  return opInst.emitError("unsupported or non-LLVM operation: ")
         << opInst.getName();
}

// Convert block to LLVM IR.  Unless `ignoreArguments` is set, emit PHI nodes
// to define values corresponding to the MLIR block arguments.  These nodes
// are not connected to the source basic blocks, which may not exist yet.
LogicalResult ModuleTranslation::convertBlock(Block &bb, bool ignoreArguments) {
  llvm::IRBuilder<> builder(blockMapping[&bb]);

  // Before traversing operations, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first block have been already made available through the remapping of
  // LLVM function arguments.
  if (!ignoreArguments) {
    auto predecessors = bb.getPredecessors();
    unsigned numPredecessors =
        std::distance(predecessors.begin(), predecessors.end());
    for (auto *arg : bb.getArguments()) {
      auto wrappedType = arg->getType().dyn_cast<LLVM::LLVMType>();
      if (!wrappedType)
        return emitError(bb.front().getLoc(),
                         "block argument does not have an LLVM type");
      llvm::Type *type = wrappedType.getUnderlyingType();
      llvm::PHINode *phi = builder.CreatePHI(type, numPredecessors);
      valueMapping[arg] = phi;
    }
  }

  // Traverse operations.
  for (auto &op : bb) {
    if (failed(convertOperation(op, builder)))
      return failure();
  }

  return success();
}

// Create named global variables that correspond to llvm.global definitions.
void ModuleTranslation::convertGlobals() {
  for (auto op : mlirModule.getOps<LLVM::GlobalOp>()) {
    llvm::Constant *cst;
    llvm::Type *type;
    // String attributes are treated separately because they cannot appear as
    // in-function constants and are thus not supported by getLLVMConstant.
    if (auto strAttr = op.value().dyn_cast<StringAttr>()) {
      cst = llvm::ConstantDataArray::getString(
          llvmModule->getContext(), strAttr.getValue(), /*AddNull=*/false);
      type = cst->getType();
    } else {
      type = op.getType().getUnderlyingType();
      cst = getLLVMConstant(type, op.value(), op.getLoc());
    }

    auto *var = new llvm::GlobalVariable(*llvmModule, type, op.constant(),
                                         llvm::GlobalValue::InternalLinkage,
                                         cst, op.sym_name());
    globalsMapping.try_emplace(op, var);
  }
}

// Get the SSA value passed to the current block from the terminator operation
// of its predecessor.
static Value *getPHISourceValue(Block *current, Block *pred,
                                unsigned numArguments, unsigned index) {
  auto &terminator = *pred->getTerminator();
  if (isa<LLVM::BrOp>(terminator)) {
    return terminator.getOperand(index);
  }

  // For conditional branches, we need to check if the current block is reached
  // through the "true" or the "false" branch and take the relevant operands.
  auto condBranchOp = dyn_cast<LLVM::CondBrOp>(terminator);
  assert(condBranchOp &&
         "only branch operations can be terminators of a block that "
         "has successors");
  assert((condBranchOp.getSuccessor(0) != condBranchOp.getSuccessor(1)) &&
         "successors with arguments in LLVM conditional branches must be "
         "different blocks");

  return condBranchOp.getSuccessor(0) == current
             ? terminator.getSuccessorOperand(0, index)
             : terminator.getSuccessorOperand(1, index);
}

void ModuleTranslation::connectPHINodes(FuncOp func) {
  // Skip the first block, it cannot be branched to and its arguments correspond
  // to the arguments of the LLVM function.
  for (auto it = std::next(func.begin()), eit = func.end(); it != eit; ++it) {
    Block *bb = &*it;
    llvm::BasicBlock *llvmBB = blockMapping.lookup(bb);
    auto phis = llvmBB->phis();
    auto numArguments = bb->getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto &numberedPhiNode : llvm::enumerate(phis)) {
      auto &phiNode = numberedPhiNode.value();
      unsigned index = numberedPhiNode.index();
      for (auto *pred : bb->getPredecessors()) {
        phiNode.addIncoming(valueMapping.lookup(getPHISourceValue(
                                bb, pred, numArguments, index)),
                            blockMapping.lookup(pred));
      }
    }
  }
}

// TODO(mlir-team): implement an iterative version
static void topologicalSortImpl(llvm::SetVector<Block *> &blocks, Block *b) {
  blocks.insert(b);
  for (Block *bb : b->getSuccessors()) {
    if (blocks.count(bb) == 0)
      topologicalSortImpl(blocks, bb);
  }
}

// Sort function blocks topologically.
static llvm::SetVector<Block *> topologicalSort(FuncOp f) {
  // For each blocks that has not been visited yet (i.e. that has no
  // predecessors), add it to the list and traverse its successors in DFS
  // preorder.
  llvm::SetVector<Block *> blocks;
  for (Block &b : f.getBlocks()) {
    if (blocks.count(&b) == 0)
      topologicalSortImpl(blocks, &b);
  }
  assert(blocks.size() == f.getBlocks().size() && "some blocks are not sorted");

  return blocks;
}

LogicalResult ModuleTranslation::convertOneFunction(FuncOp func) {
  // Clear the block and value mappings, they are only relevant within one
  // function.
  blockMapping.clear();
  valueMapping.clear();
  llvm::Function *llvmFunc = functionMapping.lookup(func.getName());
  // Add function arguments to the value remapping table.
  // If there was noalias info then we decorate each argument accordingly.
  unsigned int argIdx = 0;
  for (const auto &kvp : llvm::zip(func.getArguments(), llvmFunc->args())) {
    llvm::Argument &llvmArg = std::get<1>(kvp);
    BlockArgument *mlirArg = std::get<0>(kvp);

    if (auto attr = func.getArgAttrOfType<BoolAttr>(argIdx, "llvm.noalias")) {
      // NB: Attribute already verified to be boolean, so check if we can indeed
      // attach the attribute to this argument, based on its type.
      auto argTy = mlirArg->getType().dyn_cast<LLVM::LLVMType>();
      if (!argTy.getUnderlyingType()->isPointerTy())
        return func.emitError(
            "llvm.noalias attribute attached to LLVM non-pointer argument");
      if (attr.getValue())
        llvmArg.addAttr(llvm::Attribute::AttrKind::NoAlias);
    }
    valueMapping[mlirArg] = &llvmArg;
    argIdx++;
  }

  // First, create all blocks so we can jump to them.
  llvm::LLVMContext &llvmContext = llvmFunc->getContext();
  for (auto &bb : func) {
    auto *llvmBB = llvm::BasicBlock::Create(llvmContext);
    llvmBB->insertInto(llvmFunc);
    blockMapping[&bb] = llvmBB;
  }

  // Then, convert blocks one by one in topological order to ensure defs are
  // converted before uses.
  auto blocks = topologicalSort(func);
  for (auto indexedBB : llvm::enumerate(blocks)) {
    auto *bb = indexedBB.value();
    if (failed(convertBlock(*bb, /*ignoreArguments=*/indexedBB.index() == 0)))
      return failure();
  }

  // Finally, after all blocks have been traversed and values mapped, connect
  // the PHI nodes to the results of preceding blocks.
  connectPHINodes(func);
  return success();
}

LogicalResult ModuleTranslation::convertFunctions() {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles.
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    mlir::BoolAttr isVarArgsAttr =
        function.getAttrOfType<BoolAttr>("std.varargs");
    bool isVarArgs = isVarArgsAttr && isVarArgsAttr.getValue();
    llvm::FunctionType *functionType =
        convertFunctionType(llvmModule->getContext(), function.getType(),
                            function.getLoc(), isVarArgs);
    if (!functionType)
      return failure();
    llvm::FunctionCallee llvmFuncCst =
        llvmModule->getOrInsertFunction(function.getName(), functionType);
    assert(isa<llvm::Function>(llvmFuncCst.getCallee()));
    functionMapping[function.getName()] =
        cast<llvm::Function>(llvmFuncCst.getCallee());
  }

  // Convert functions.
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    // Ignore external functions.
    if (function.isExternal())
      continue;

    if (failed(convertOneFunction(function)))
      return failure();
  }

  return success();
}

std::unique_ptr<llvm::Module> ModuleTranslation::prepareLLVMModule(ModuleOp m) {
  auto *dialect = m.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  assert(dialect && "LLVM dialect must be registered");

  auto llvmModule = llvm::CloneModule(dialect->getLLVMModule());
  if (!llvmModule)
    return nullptr;

  llvm::LLVMContext &llvmContext = llvmModule->getContext();
  llvm::IRBuilder<> builder(llvmContext);

  // Inject declarations for `malloc` and `free` functions that can be used in
  // memref allocation/deallocation coming from standard ops lowering.
  llvmModule->getOrInsertFunction("malloc", builder.getInt8PtrTy(),
                                  builder.getInt64Ty());
  llvmModule->getOrInsertFunction("free", builder.getVoidTy(),
                                  builder.getInt8PtrTy());

  return llvmModule;
}

} // namespace LLVM
} // namespace mlir
