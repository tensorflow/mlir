//===- Pattern.cpp - Pattern wrapper class --------------------------------===//
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
// Pattern wrapper class to simplify using TableGen Record defining a MLIR
// Pattern.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Pattern.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

using llvm::formatv;
using mlir::tblgen::Operator;

//===----------------------------------------------------------------------===//
// DagLeaf
//===----------------------------------------------------------------------===//

bool tblgen::DagLeaf::isUnspecified() const {
  return dyn_cast_or_null<llvm::UnsetInit>(def);
}

bool tblgen::DagLeaf::isOperandMatcher() const {
  // Operand matchers specify a type constraint.
  return isSubClassOf("TypeConstraint");
}

bool tblgen::DagLeaf::isAttrMatcher() const {
  // Attribute matchers specify an attribute constraint.
  return isSubClassOf("AttrConstraint");
}

bool tblgen::DagLeaf::isNativeCodeCall() const {
  return isSubClassOf("NativeCodeCall");
}

bool tblgen::DagLeaf::isConstantAttr() const {
  return isSubClassOf("ConstantAttr");
}

bool tblgen::DagLeaf::isEnumAttrCase() const {
  return isSubClassOf("EnumAttrCaseInfo");
}

tblgen::Constraint tblgen::DagLeaf::getAsConstraint() const {
  assert((isOperandMatcher() || isAttrMatcher()) &&
         "the DAG leaf must be operand or attribute");
  return Constraint(cast<llvm::DefInit>(def)->getDef());
}

tblgen::ConstantAttr tblgen::DagLeaf::getAsConstantAttr() const {
  assert(isConstantAttr() && "the DAG leaf must be constant attribute");
  return ConstantAttr(cast<llvm::DefInit>(def));
}

tblgen::EnumAttrCase tblgen::DagLeaf::getAsEnumAttrCase() const {
  assert(isEnumAttrCase() && "the DAG leaf must be an enum attribute case");
  return EnumAttrCase(cast<llvm::DefInit>(def));
}

std::string tblgen::DagLeaf::getConditionTemplate() const {
  return getAsConstraint().getConditionTemplate();
}

llvm::StringRef tblgen::DagLeaf::getNativeCodeTemplate() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(def)->getDef()->getValueAsString("expression");
}

bool tblgen::DagLeaf::isSubClassOf(StringRef superclass) const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(def))
    return defInit->getDef()->isSubClassOf(superclass);
  return false;
}

//===----------------------------------------------------------------------===//
// DagNode
//===----------------------------------------------------------------------===//

bool tblgen::DagNode::isNativeCodeCall() const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(node->getOperator()))
    return defInit->getDef()->isSubClassOf("NativeCodeCall");
  return false;
}

bool tblgen::DagNode::isOperation() const {
  return !(isNativeCodeCall() || isReplaceWithValue());
}

llvm::StringRef tblgen::DagNode::getNativeCodeTemplate() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(node->getOperator())
      ->getDef()
      ->getValueAsString("expression");
}

llvm::StringRef tblgen::DagNode::getSymbol() const {
  return node->getNameStr();
}

Operator &tblgen::DagNode::getDialectOp(RecordOperatorMap *mapper) const {
  llvm::Record *opDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  auto it = mapper->find(opDef);
  if (it != mapper->end())
    return *it->second;
  return *mapper->try_emplace(opDef, llvm::make_unique<Operator>(opDef))
              .first->second;
}

int tblgen::DagNode::getNumOps() const {
  int count = isReplaceWithValue() ? 0 : 1;
  for (int i = 0, e = getNumArgs(); i != e; ++i) {
    if (auto child = getArgAsNestedDag(i))
      count += child.getNumOps();
  }
  return count;
}

int tblgen::DagNode::getNumArgs() const { return node->getNumArgs(); }

bool tblgen::DagNode::isNestedDagArg(unsigned index) const {
  return isa<llvm::DagInit>(node->getArg(index));
}

tblgen::DagNode tblgen::DagNode::getArgAsNestedDag(unsigned index) const {
  return DagNode(dyn_cast_or_null<llvm::DagInit>(node->getArg(index)));
}

tblgen::DagLeaf tblgen::DagNode::getArgAsLeaf(unsigned index) const {
  assert(!isNestedDagArg(index));
  return DagLeaf(node->getArg(index));
}

StringRef tblgen::DagNode::getArgName(unsigned index) const {
  return node->getArgNameStr(index);
}

bool tblgen::DagNode::isReplaceWithValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "replaceWithValue";
}

//===----------------------------------------------------------------------===//
// SymbolInfoMap
//===----------------------------------------------------------------------===//

StringRef tblgen::SymbolInfoMap::getValuePackName(StringRef symbol,
                                                  int *index) {
  StringRef name, indexStr;
  int idx = -1;
  std::tie(name, indexStr) = symbol.rsplit("__");

  if (indexStr.consumeInteger(10, idx)) {
    // The second part is not an index; we return the whole symbol as-is.
    return symbol;
  }
  if (index) {
    *index = idx;
  }
  return name;
}

tblgen::SymbolInfoMap::SymbolInfo::SymbolInfo(const Operator *op,
                                              SymbolInfo::Kind kind,
                                              Optional<int> index)
    : op(op), kind(kind), argIndex(index) {}

int tblgen::SymbolInfoMap::SymbolInfo::getStaticValueCount() const {
  switch (kind) {
  case Kind::Attr:
  case Kind::Operand:
  case Kind::Value:
    return 1;
  case Kind::Result:
    return op->getNumResults();
  }
  llvm_unreachable("unknown kind");
}

std::string
tblgen::SymbolInfoMap::SymbolInfo::getVarDecl(StringRef name) const {
  switch (kind) {
  case Kind::Attr: {
    auto type =
        op->getArg(*argIndex).get<NamedAttribute *>()->attr.getStorageType();
    return formatv("{0} {1};\n", type, name);
  }
  case Kind::Operand:
  case Kind::Value: {
    return formatv("Value *{0};\n", name);
  }
  case Kind::Result: {
    // Use the op itself for the results.
    return formatv("{0} {1};\n", op->getQualCppClassName(), name);
  }
  }
  llvm_unreachable("unknown kind");
}

std::string
tblgen::SymbolInfoMap::SymbolInfo::getValueAndRangeUse(StringRef name,
                                                       int index) const {
  switch (kind) {
  case Kind::Attr:
  case Kind::Operand: {
    assert(index < 0 && "only allowed for symbol bound to result");
    return name;
  }
  case Kind::Result: {
    // TODO(b/133341698): The following is incorrect for variadic results. We
    // should use getODSResults().
    if (index >= 0) {
      return formatv("{0}.getOperation()->getResult({1})", name, index);
    }

    // If referencing multiple results, compose a comma-separated list.
    SmallVector<std::string, 4> values;
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      values.push_back(formatv("{0}.getOperation()->getResult({1})", name, i));
    }
    return llvm::join(values, ", ");
  }
  case Kind::Value: {
    assert(index < 0 && "only allowed for symbol bound to result");
    assert(op == nullptr);
    return name;
  }
  }
  llvm_unreachable("unknown kind");
}

bool tblgen::SymbolInfoMap::bindOpArgument(StringRef symbol, const Operator &op,
                                           int argIndex) {
  StringRef name = getValuePackName(symbol);
  if (name != symbol) {
    auto error = formatv(
        "symbol '{0}' with trailing index cannot bind to op argument", symbol);
    PrintFatalError(loc, error);
  }

  auto symInfo = op.getArg(argIndex).is<NamedAttribute *>()
                     ? SymbolInfo::getAttr(&op, argIndex)
                     : SymbolInfo::getOperand(&op, argIndex);

  return symbolInfoMap.insert({symbol, symInfo}).second;
}

bool tblgen::SymbolInfoMap::bindOpResult(StringRef symbol, const Operator &op) {
  StringRef name = getValuePackName(symbol);
  return symbolInfoMap.insert({name, SymbolInfo::getResult(&op)}).second;
}

bool tblgen::SymbolInfoMap::bindValue(StringRef symbol) {
  return symbolInfoMap.insert({symbol, SymbolInfo::getValue()}).second;
}

bool tblgen::SymbolInfoMap::contains(StringRef symbol) const {
  return find(symbol) != symbolInfoMap.end();
}

tblgen::SymbolInfoMap::const_iterator
tblgen::SymbolInfoMap::find(StringRef key) const {
  StringRef name = getValuePackName(key);
  return symbolInfoMap.find(name);
}

int tblgen::SymbolInfoMap::getStaticValueCount(StringRef symbol) const {
  StringRef name = getValuePackName(symbol);
  if (name != symbol) {
    // If there is a trailing index inside symbol, it references just one
    // static value.
    return 1;
  }
  // Otherwise, find how many it represents by querying the symbol's info.
  return find(name)->getValue().getStaticValueCount();
}

std::string tblgen::SymbolInfoMap::getValueAndRangeUse(StringRef symbol) const {
  int index = -1;
  StringRef name = getValuePackName(symbol, &index);

  auto it = symbolInfoMap.find(name);
  if (it == symbolInfoMap.end()) {
    auto error = formatv("referencing unbound symbol '{0}'", symbol);
    PrintFatalError(loc, error);
  }

  return it->getValue().getValueAndRangeUse(name, index);
}

//===----------------------------------------------------------------------===//
// Pattern
//==----------------------------------------------------------------------===//

tblgen::Pattern::Pattern(const llvm::Record *def, RecordOperatorMap *mapper)
    : def(*def), recordOpMap(mapper) {}

tblgen::DagNode tblgen::Pattern::getSourcePattern() const {
  return tblgen::DagNode(def.getValueAsDag("sourcePattern"));
}

int tblgen::Pattern::getNumResultPatterns() const {
  auto *results = def.getValueAsListInit("resultPatterns");
  return results->size();
}

tblgen::DagNode tblgen::Pattern::getResultPattern(unsigned index) const {
  auto *results = def.getValueAsListInit("resultPatterns");
  return tblgen::DagNode(cast<llvm::DagInit>(results->getElement(index)));
}

void tblgen::Pattern::collectSourcePatternBoundSymbols(
    tblgen::SymbolInfoMap &infoMap) {
  collectBoundSymbols(getSourcePattern(), infoMap, /*isSrcPattern=*/true);
}

void tblgen::Pattern::collectResultPatternBoundSymbols(
    tblgen::SymbolInfoMap &infoMap) {
  for (int i = 0, e = getNumResultPatterns(); i < e; ++i) {
    auto pattern = getResultPattern(i);
    collectBoundSymbols(pattern, infoMap, /*isSrcPattern=*/false);
  }
}

const tblgen::Operator &tblgen::Pattern::getSourceRootOp() {
  return getSourcePattern().getDialectOp(recordOpMap);
}

tblgen::Operator &tblgen::Pattern::getDialectOp(DagNode node) {
  return node.getDialectOp(recordOpMap);
}

std::vector<tblgen::AppliedConstraint> tblgen::Pattern::getConstraints() const {
  auto *listInit = def.getValueAsListInit("constraints");
  std::vector<tblgen::AppliedConstraint> ret;
  ret.reserve(listInit->size());

  for (auto it : *listInit) {
    auto *dagInit = dyn_cast<llvm::DagInit>(it);
    if (!dagInit)
      PrintFatalError(def.getLoc(), "all elemements in Pattern multi-entity "
                                    "constraints should be DAG nodes");

    std::vector<std::string> entities;
    entities.reserve(dagInit->arg_size());
    for (auto *argName : dagInit->getArgNames())
      entities.push_back(argName->getValue());

    ret.emplace_back(cast<llvm::DefInit>(dagInit->getOperator())->getDef(),
                     dagInit->getNameStr(), std::move(entities));
  }
  return ret;
}

int tblgen::Pattern::getBenefit() const {
  // The initial benefit value is a heuristic with number of ops in the source
  // pattern.
  int initBenefit = getSourcePattern().getNumOps();
  llvm::DagInit *delta = def.getValueAsDag("benefitDelta");
  if (delta->getNumArgs() != 1 || !isa<llvm::IntInit>(delta->getArg(0))) {
    PrintFatalError(def.getLoc(),
                    "The 'addBenefit' takes and only takes one integer value");
  }
  return initBenefit + dyn_cast<llvm::IntInit>(delta->getArg(0))->getValue();
}

std::vector<tblgen::Pattern::IdentifierLine>
tblgen::Pattern::getLocation() const {
  std::vector<std::pair<StringRef, unsigned>> result;
  result.reserve(def.getLoc().size());
  for (auto loc : def.getLoc()) {
    unsigned buf = llvm::SrcMgr.FindBufferContainingLoc(loc);
    assert(buf && "invalid source location");
    result.emplace_back(
        llvm::SrcMgr.getBufferInfo(buf).Buffer->getBufferIdentifier(),
        llvm::SrcMgr.getLineAndColumn(loc, buf).first);
  }
  return result;
}

void tblgen::Pattern::collectBoundSymbols(DagNode tree, SymbolInfoMap &infoMap,
                                          bool isSrcPattern) {
  auto treeName = tree.getSymbol();
  if (!tree.isOperation()) {
    if (!treeName.empty()) {
      PrintFatalError(
          def.getLoc(),
          formatv("binding symbol '{0}' to non-operation unsupported right now",
                  treeName));
    }
    return;
  }

  auto &op = getDialectOp(tree);
  auto numOpArgs = op.getNumArgs();
  auto numTreeArgs = tree.getNumArgs();

  if (numOpArgs != numTreeArgs) {
    auto err = formatv("op '{0}' argument number mismatch: "
                       "{1} in pattern vs. {2} in definition",
                       op.getOperationName(), numTreeArgs, numOpArgs);
    PrintFatalError(def.getLoc(), err);
  }

  // The name attached to the DAG node's operator is for representing the
  // results generated from this op. It should be remembered as bound results.
  if (!treeName.empty()) {
    if (!infoMap.bindOpResult(treeName, op))
      PrintFatalError(def.getLoc(),
                      formatv("symbol '{0}' bound more than once", treeName));
  }

  for (int i = 0; i != numTreeArgs; ++i) {
    if (auto treeArg = tree.getArgAsNestedDag(i)) {
      // This DAG node argument is a DAG node itself. Go inside recursively.
      collectBoundSymbols(treeArg, infoMap, isSrcPattern);
    } else if (isSrcPattern) {
      // We can only bind symbols to op arguments in source pattern. Those
      // symbols are referenced in result patterns.
      auto treeArgName = tree.getArgName(i);
      if (!treeArgName.empty()) {
        if (!infoMap.bindOpArgument(treeArgName, op, i)) {
          auto err = formatv("symbol '{0}' bound more than once", treeArgName);
          PrintFatalError(def.getLoc(), err);
        }
      }
    }
  }
}
