//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
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
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/MLIRGen.h"
#include "toy/AST.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace toy;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
///
/// At this point we take advantage of the "raw" MLIR APIs to create operations
/// that haven't been registered in any way with MLIR. These operations are
/// unknown to MLIR, custom passes could operate by string-matching the name of
/// these operations, but no other type checking or semantics are associated
/// with them natively by MLIR.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context)
      : context(context), builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    for (FunctionAST &F : moduleAST) {
      auto func = mlirGen(F);
      if (!func)
        return nullptr;
      theModule.push_back(func);
    }

    // FIXME: (in the next chapter...) without registering a dialect in MLIR,
    // this won't do much, but it should at least check some structural
    // properties of the generated MLIR module.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("Module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// ownership of many internal structures of the IR and provides a level of
  /// "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value *> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::FuncOp mlirGen(PrototypeAST &proto) {
    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> ret_types;
    // Arguments type is uniformly a generic array.
    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                               getType(VarType{}));
    auto func_type = builder.getFunctionType(arg_types, ret_types);
    auto function = mlir::FuncOp::create(loc(proto.loc()), proto.getName(),
                                         func_type, /* attrs = */ {});

    // Mark the function as generic: it'll require type specialization for every
    // call site.
    if (function.getNumArguments())
      function.setAttr("toy.generic", builder.getUnitAttr());
    return function;
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value *> var_scope(symbolTable);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(mlirGen(*funcAST.getProto()));
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
    auto &protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto &name_value :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(name_value)->getName(),
                         std::get<1>(name_value))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    if (function.getBody().back().back().getName().getStringRef() !=
        "toy.return") {
      ReturnExprAST fakeRet(funcAST.getProto()->loc(), llvm::None);
      mlirGen(fakeRet);
    }

    return function;
  }

  /// Emit a binary operation
  mlir::Value *mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value *L = mlirGen(*binop.getLHS());
    if (!L)
      return nullptr;
    mlir::Value *R = mlirGen(*binop.getRHS());
    if (!R)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    const char *op_name = nullptr;
    switch (binop.getOp()) {
    case '+':
      op_name = "toy.add";
      break;
    case '*':
      op_name = "toy.mul";
      break;
    default:
      emitError(location, "Error: invalid binary operator '")
          << binop.getOp() << "'";
      return nullptr;
    }

    // Build the MLIR operation from the name and the two operands. The return
    // type is always a generic array for binary operators.
    mlir::OperationState result(location, op_name);
    result.addTypes(getType(VarType{}));
    result.addOperands({L, R});
    return builder.createOperation(result)->getResult(0);
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value *mlirGen(VariableExprAST &expr) {
    if (auto *variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "Error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    mlir::OperationState result(loc(ret.loc()), "toy.return");

    // `return` takes an optional expression, we need to account for it here.
    if (ret.getExpr().hasValue()) {
      auto *expr = mlirGen(*ret.getExpr().getValue());
      if (!expr)
        return mlir::failure();
      result.addOperands(expr);
    }

    builder.createOperation(result);
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value *mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = builder.getTensorType(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

    // Build the MLIR op `toy.constant`, only boilerplate below.
    mlir::OperationState result(loc(lit.loc()), "toy.constant");
    result.addTypes(type);
    result.addAttribute("value", dataAttribute);
    return builder.createOperation(result)->getResult(0);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value *mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();

    // Codegen the operands first.
    SmallVector<mlir::Value *, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto *arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Builting calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      mlir::OperationState result(loc(call.loc()), "toy.transpose");
      result.addTypes(getType(VarType{}));
      result.operands = std::move(operands);
      return builder.createOperation(result)->getResult(0);
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    mlir::OperationState result(loc(call.loc()), "toy.generic_call");
    result.addTypes(getType(VarType{}));
    result.operands = std::move(operands);
    result.addAttribute("callee", builder.getSymbolRefAttr(callee));
    return builder.createOperation(result)->getResult(0);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto *arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    mlir::OperationState result(loc(call.loc()), "toy.print");
    result.addOperands(arg);
    builder.createOperation(result);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value *mlirGen(NumberExprAST &num) {
    mlir::OperationState result(loc(num.loc()), "toy.constant");
    mlir::Type elementType = builder.getF64Type();
    result.addTypes(builder.getTensorType({}, elementType));
    result.addAttribute("value", builder.getF64FloatAttr(num.getValue()));
    return builder.createOperation(result)->getResult(0);
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value *mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value *mlirGen(VarDeclExprAST &vardecl) {
    auto init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "Missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value *value = mlirGen(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      mlir::OperationState result(loc(vardecl.loc()), "toy.reshape");
      result.addTypes(getType(vardecl.getType()));
      result.addOperands(value);
      value = builder.createOperation(result)->getResult(0);
    }

    // Register the value in the symbol table
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<llvm::StringRef, mlir::Value *> var_scope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return builder.getTensorType(builder.getF64Type());

    // Otherwise, we use the given shape.
    return builder.getTensorType(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type
  /// (forward to the generic getType(T) above).
  mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace

namespace toy {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace toy
