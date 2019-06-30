//===- pybind.cpp - MLIR Python bindings ----------------------------------===//
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <unordered_map>

#include "mlir-c/Core.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

static bool inited = [] {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  return true;
}();

namespace mlir {
namespace edsc {
namespace python {

namespace py = pybind11;

struct PythonAttribute;
struct PythonAttributedType;
struct PythonBindable;
struct PythonExpr;
struct PythonFunctionContext;
struct PythonStmt;
struct PythonBlock;

struct PythonType {
  PythonType() : type{nullptr} {}
  PythonType(mlir_type_t t) : type{t} {}

  operator mlir_type_t() const { return type; }

  PythonAttributedType attachAttributeDict(
      const std::unordered_map<std::string, PythonAttribute> &attrs) const;

  std::string str() {
    mlir::Type f = mlir::Type::getFromOpaquePointer(type);
    std::string res;
    llvm::raw_string_ostream os(res);
    f.print(os);
    return res;
  }

  mlir_type_t type;
};

struct PythonValueHandle {
  PythonValueHandle(PythonType type)
      : value(mlir::Type::getFromOpaquePointer(type.type)) {}
  PythonValueHandle(const PythonValueHandle &other) = default;
  PythonValueHandle(const mlir::edsc::ValueHandle &other) : value(other) {}
  operator ValueHandle() const { return value; }
  operator ValueHandle &() { return value; }

  std::string str() const {
    return std::to_string(reinterpret_cast<intptr_t>(value.getValue()));
  }

  PythonValueHandle call(const std::vector<PythonValueHandle> &args) {
    assert(value.hasType() && value.getType().isa<FunctionType>() &&
           "can only call function-typed values");

    std::vector<Value *> argValues;
    argValues.reserve(args.size());
    for (auto arg : args)
      argValues.push_back(arg.value.getValue());
    return ValueHandle::create<CallIndirectOp>(value, argValues);
  }

  mlir::edsc::ValueHandle value;
};

struct PythonFunction {
  PythonFunction() : function{nullptr} {}
  PythonFunction(mlir_func_t f) : function{f} {}
  PythonFunction(mlir::Function *f) : function{f} {}
  operator mlir_func_t() { return function; }
  std::string str() {
    mlir::Function *f = reinterpret_cast<mlir::Function *>(function);
    std::string res;
    llvm::raw_string_ostream os(res);
    f->print(os);
    return res;
  }

  // If the function does not yet have an entry block, i.e. if it is a function
  // declaration, add the entry block, transforming the declaration into a
  // definition.  Return true if the block was added, false otherwise.
  bool define() {
    auto *f = reinterpret_cast<mlir::Function *>(function);
    if (!f->getBlocks().empty())
      return false;

    f->addEntryBlock();
    return true;
  }

  PythonValueHandle arg(unsigned index) {
    Function *f = static_cast<Function *>(function);
    assert(index < f->getNumArguments() && "argument index out of bounds");
    return PythonValueHandle(ValueHandle(f->getArgument(index)));
  }

  mlir_func_t function;
};

/// Trivial C++ wrappers make use of the EDSC C API.
struct PythonMLIRModule {
  PythonMLIRModule()
      : mlirContext(), module(new mlir::Module(&mlirContext)),
        moduleManager(module.get()) {}

  PythonType makeScalarType(const std::string &mlirElemType,
                            unsigned bitwidth) {
    return ::makeScalarType(mlir_context_t{&mlirContext}, mlirElemType.c_str(),
                            bitwidth);
  }
  PythonType makeMemRefType(PythonType elemType, std::vector<int64_t> sizes) {
    return ::makeMemRefType(mlir_context_t{&mlirContext}, elemType,
                            int64_list_t{sizes.data(), sizes.size()});
  }
  PythonType makeIndexType() {
    return ::makeIndexType(mlir_context_t{&mlirContext});
  }

  // Declare a function with the given name, input types and their attributes,
  // output types, and function attributes, but do not define it.
  PythonFunction declareFunction(const std::string &name,
                                 const py::list &inputs,
                                 const std::vector<PythonType> &outputTypes,
                                 const py::kwargs &funcAttributes);

  // Declare a function with the given name, input types and their attributes,
  // output types, and function attributes.
  PythonFunction makeFunction(const std::string &name, const py::list &inputs,
                              const std::vector<PythonType> &outputTypes,
                              const py::kwargs &funcAttributes) {
    auto declaration =
        declareFunction(name, inputs, outputTypes, funcAttributes);
    declaration.define();
    return declaration;
  }

  // Create a custom op given its name and arguments.
  PythonExpr op(const std::string &name, PythonType type,
                const py::list &arguments, const py::list &successors,
                py::kwargs attributes);

  // Create an integer attribute.
  PythonAttribute integerAttr(PythonType type, int64_t value);

  // Create a boolean attribute.
  PythonAttribute boolAttr(bool value);

  void compile() {
    PassManager manager;
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createCSEPass());
    manager.addPass(mlir::createLowerAffinePass());
    manager.addPass(mlir::createConvertToLLVMIRPass());
    if (failed(manager.run(module.get()))) {
      llvm::errs() << "conversion to the LLVM IR dialect failed\n";
      return;
    }

    auto created = mlir::ExecutionEngine::create(module.get());
    llvm::handleAllErrors(created.takeError(),
                          [](const llvm::ErrorInfoBase &b) {
                            b.log(llvm::errs());
                            assert(false);
                          });
    engine = std::move(*created);
  }

  std::string getIR() {
    std::string res;
    llvm::raw_string_ostream os(res);
    module->print(os);
    return res;
  }

  uint64_t getEngineAddress() {
    assert(engine && "module must be compiled into engine first");
    return reinterpret_cast<uint64_t>(reinterpret_cast<void *>(engine.get()));
  }

  PythonFunction getNamedFunction(const std::string &name) {
    return moduleManager.getNamedFunction(name);
  }

  PythonFunctionContext
  makeFunctionContext(const std::string &name, const py::list &inputs,
                      const std::vector<PythonType> &outputs,
                      const py::kwargs &attributes);

private:
  mlir::MLIRContext mlirContext;
  // One single module in a python-exposed MLIRContext for now.
  std::unique_ptr<mlir::Module> module;
  mlir::ModuleManager moduleManager;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

struct PythonFunctionContext {
  PythonFunctionContext(PythonFunction f) : function(f) {}
  PythonFunctionContext(PythonMLIRModule &module, const std::string &name,
                        const py::list &inputs,
                        const std::vector<PythonType> &outputs,
                        const py::kwargs &attributes) {
    auto function = module.declareFunction(name, inputs, outputs, attributes);
    function.define();
  }

  PythonFunction enter() {
    assert(function.function && "function is not set up");
    auto *mlirFunc = static_cast<mlir::Function *>(function.function);
    contextBuilder.emplace(mlirFunc->getBody());
    context =
        new mlir::edsc::ScopedContext(*contextBuilder, mlirFunc->getLoc());
    return function;
  }

  void exit(py::object, py::object, py::object) {
    delete context;
    context = nullptr;
    contextBuilder.reset();
  }

  PythonFunction function;
  mlir::edsc::ScopedContext *context;
  llvm::Optional<OpBuilder> contextBuilder;
};

PythonFunctionContext PythonMLIRModule::makeFunctionContext(
    const std::string &name, const py::list &inputs,
    const std::vector<PythonType> &outputs, const py::kwargs &attributes) {
  auto func = declareFunction(name, inputs, outputs, attributes);
  func.define();
  return PythonFunctionContext(func);
}

struct PythonBlockHandle {
  PythonBlockHandle() : value(nullptr) {}
  PythonBlockHandle(const PythonBlockHandle &other) = default;
  PythonBlockHandle(const mlir::edsc::BlockHandle &other) : value(other) {}
  operator mlir::edsc::BlockHandle() const { return value; }

  PythonValueHandle arg(int index) { return arguments[index]; }

  std::string str() {
    std::string s;
    llvm::raw_string_ostream os(s);
    value.getBlock()->print(os);
    return os.str();
  }

  mlir::edsc::BlockHandle value;
  std::vector<mlir::edsc::ValueHandle> arguments;
};

struct PythonLoopContext {
  PythonLoopContext(PythonValueHandle lb, PythonValueHandle ub, int64_t step)
      : lb(lb), ub(ub), step(step) {}
  PythonLoopContext(const PythonLoopContext &) = delete;
  PythonLoopContext(PythonLoopContext &&) = default;
  PythonLoopContext &operator=(const PythonLoopContext &) = delete;
  PythonLoopContext &operator=(PythonLoopContext &&) = default;
  ~PythonLoopContext() { assert(!builder && "did not exit from the context"); }

  PythonValueHandle enter() {
    ValueHandle iv(lb.value.getType());
    builder = new LoopBuilder(&iv, lb.value, ub.value, step);
    return iv;
  }

  void exit(py::object, py::object, py::object) {
    (*builder)({}); // exit from the builder's scope.
    delete builder;
    builder = nullptr;
  }

  PythonValueHandle lb, ub;
  int64_t step;
  LoopBuilder *builder = nullptr;
};

struct PythonLoopNestContext {
  PythonLoopNestContext(const std::vector<PythonValueHandle> &lbs,
                        const std::vector<PythonValueHandle> &ubs,
                        const std::vector<int64_t> steps)
      : lbs(lbs), ubs(ubs), steps(steps) {
    assert(lbs.size() == ubs.size() && lbs.size() == steps.size() &&
           "expected the same number of lower, upper bounds, and steps");
  }
  PythonLoopNestContext(const PythonLoopNestContext &) = delete;
  PythonLoopNestContext(PythonLoopNestContext &&) = default;
  PythonLoopNestContext &operator=(const PythonLoopNestContext &) = delete;
  PythonLoopNestContext &operator=(PythonLoopNestContext &&) = default;
  ~PythonLoopNestContext() {
    assert(!builder && "did not exit from the context");
  }

  std::vector<PythonValueHandle> enter() {
    if (steps.empty())
      return {};

    auto type = mlir_type_t(lbs.front().value.getType().getAsOpaquePointer());
    std::vector<PythonValueHandle> handles(steps.size(),
                                           PythonValueHandle(type));
    std::vector<ValueHandle *> handlePtrs;
    handlePtrs.reserve(steps.size());
    for (auto &h : handles)
      handlePtrs.push_back(&h.value);
    builder = new LoopNestBuilder(
        handlePtrs, std::vector<ValueHandle>(lbs.begin(), lbs.end()),
        std::vector<ValueHandle>(ubs.begin(), ubs.end()), steps);
    return handles;
  }

  void exit(py::object, py::object, py::object) {
    (*builder)({}); // exit from the builder's scope.
    delete builder;
    builder = nullptr;
  }

  std::vector<PythonValueHandle> lbs;
  std::vector<PythonValueHandle> ubs;
  std::vector<int64_t> steps;
  LoopNestBuilder *builder = nullptr;
};

struct PythonBlockAppender {
  PythonBlockAppender(const PythonBlockHandle &handle) : handle(handle) {}
  PythonBlockHandle handle;
};

struct PythonBlockContext {
public:
  PythonBlockContext() {
    createBlockBuilder();
    clearBuilder();
  }
  PythonBlockContext(const std::vector<PythonType> &argTypes) {
    handle.arguments.reserve(argTypes.size());
    for (const auto &t : argTypes) {
      auto type =
          Type::getFromOpaquePointer(reinterpret_cast<const void *>(t.type));
      handle.arguments.emplace_back(type);
    }
    createBlockBuilder();
    clearBuilder();
  }
  PythonBlockContext(const PythonBlockAppender &a) : handle(a.handle) {}
  PythonBlockContext(const PythonBlockContext &) = delete;
  PythonBlockContext(PythonBlockContext &&) = default;
  PythonBlockContext &operator=(const PythonBlockContext &) = delete;
  PythonBlockContext &operator=(PythonBlockContext &&) = default;
  ~PythonBlockContext() {
    assert(!builder && "did not exit from the block context");
  }

  // EDSC maintain an implicit stack of builders (mostly for keeping track of
  // insretion points); every operation gets inserted using the top-of-the-stack
  // builder.  Creating a new EDSC Builder automatically puts it on the stack,
  // effectively entering the block for it.
  void createBlockBuilder() {
    if (handle.value.getBlock()) {
      builder = new BlockBuilder(handle.value, mlir::edsc::Append());
    } else {
      std::vector<ValueHandle *> args;
      args.reserve(handle.arguments.size());
      for (auto &a : handle.arguments)
        args.push_back(&a);
      builder = new BlockBuilder(&handle.value, args);
    }
  }

  PythonBlockHandle enter() {
    createBlockBuilder();
    return handle;
  }

  void exit(py::object, py::object, py::object) { clearBuilder(); }

  PythonBlockHandle getHandle() { return handle; }

  // EDSC maintain an implicit stack of builders (mostly for keeping track of
  // insretion points); every operation gets inserted using the top-of-the-stack
  // builder.  Calling operator() on a builder pops the builder from the stack,
  // effectively resetting the insertion point to its position before we entered
  // the block.
  void clearBuilder() {
    (*builder)({}); // exit from the builder's scope.
    delete builder;
    builder = nullptr;
  }

  PythonBlockHandle handle;
  BlockBuilder *builder = nullptr;
};

struct PythonAttribute {
  PythonAttribute() : attr(nullptr) {}
  PythonAttribute(const mlir_attr_t &a) : attr(a) {}
  PythonAttribute(const PythonAttribute &other) = default;
  operator mlir_attr_t() { return attr; }

  std::string str() const {
    if (!attr)
      return "##null attr##";

    std::string res;
    llvm::raw_string_ostream os(res);
    Attribute::getFromOpaquePointer(reinterpret_cast<const void *>(attr))
        .print(os);
    return res;
  }

  mlir_attr_t attr;
};

struct PythonAttributedType {
  PythonAttributedType() : type(nullptr) {}
  PythonAttributedType(mlir_type_t t) : type(t) {}
  PythonAttributedType(
      PythonType t,
      const std::unordered_map<std::string, PythonAttribute> &attributes =
          std::unordered_map<std::string, PythonAttribute>())
      : type(t), attrs(attributes) {}

  operator mlir_type_t() const { return type.type; }
  operator PythonType() const { return type; }

  // Return a vector of named attribute descriptors.  The vector owns the
  // mlir_named_attr_t objects it contains, but not the names and attributes
  // those objects point to (names and opaque pointers to attributes are owned
  // by `this`).
  std::vector<mlir_named_attr_t> getNamedAttrs() const {
    std::vector<mlir_named_attr_t> result;
    result.reserve(attrs.size());
    for (const auto &namedAttr : attrs)
      result.push_back({namedAttr.first.c_str(), namedAttr.second.attr});
    return result;
  }

  std::string str() {
    mlir::Type t = mlir::Type::getFromOpaquePointer(type);
    std::string res;
    llvm::raw_string_ostream os(res);
    t.print(os);
    if (attrs.empty())
      return os.str();

    os << '{';
    bool first = true;
    for (const auto &namedAttr : attrs) {
      if (first)
        first = false;
      else
        os << ", ";
      os << namedAttr.first << ": " << namedAttr.second.str();
    }
    os << '}';

    return os.str();
  }

private:
  PythonType type;
  std::unordered_map<std::string, PythonAttribute> attrs;
};

struct PythonIndexedValue {
  explicit PythonIndexedValue(PythonType type)
      : indexed(Type::getFromOpaquePointer(type.type)) {}
  explicit PythonIndexedValue(const IndexedValue &other) : indexed(other) {}
  PythonIndexedValue(PythonValueHandle handle) : indexed(handle.value) {}
  PythonIndexedValue(const PythonIndexedValue &other) = default;

  // Create a new indexed value with the same base as this one but with indices
  // provided as arguments.
  PythonIndexedValue index(const std::vector<PythonValueHandle> &indices) {
    std::vector<ValueHandle> handles(indices.begin(), indices.end());
    return PythonIndexedValue(IndexedValue(indexed(handles)));
  }

  void store(const std::vector<PythonValueHandle> &indices,
             PythonValueHandle value) {
    // Uses the overloaded `opreator=` to emit a store.
    index(indices).indexed = value.value;
  }

  PythonValueHandle load(const std::vector<PythonValueHandle> &indices) {
    // Uses the overloaded cast to `ValueHandle` to emit a load.
    return static_cast<ValueHandle>(index(indices).indexed);
  }

  IndexedValue indexed;
};

template <typename ListTy, typename PythonTy, typename Ty>
ListTy makeCList(SmallVectorImpl<Ty> &owning, const py::list &list) {
  for (auto &inp : list) {
    owning.push_back(Ty{inp.cast<PythonTy>()});
  }
  return ListTy{owning.data(), owning.size()};
}

static mlir_type_list_t makeCTypes(llvm::SmallVectorImpl<mlir_type_t> &owning,
                                   const py::list &types) {
  return makeCList<mlir_type_list_t, PythonType>(owning, types);
}

PythonFunction
PythonMLIRModule::declareFunction(const std::string &name,
                                  const py::list &inputs,
                                  const std::vector<PythonType> &outputTypes,
                                  const py::kwargs &funcAttributes) {

  std::vector<PythonAttributedType> attributedInputs;
  attributedInputs.reserve(inputs.size());
  for (const auto &in : inputs) {
    std::string className = in.get_type().str();
    if (className.find(".Type'") != std::string::npos)
      attributedInputs.emplace_back(in.cast<PythonType>());
    else
      attributedInputs.push_back(in.cast<PythonAttributedType>());
  }

  // Create the function type.
  std::vector<mlir_type_t> ins(attributedInputs.begin(),
                               attributedInputs.end());
  std::vector<mlir_type_t> outs(outputTypes.begin(), outputTypes.end());
  auto funcType = ::makeFunctionType(
      mlir_context_t{&mlirContext}, mlir_type_list_t{ins.data(), ins.size()},
      mlir_type_list_t{outs.data(), outs.size()});

  // Build the list of function attributes.
  std::vector<mlir::NamedAttribute> attrs;
  attrs.reserve(funcAttributes.size());
  for (const auto &named : funcAttributes)
    attrs.emplace_back(
        Identifier::get(std::string(named.first.str()), &mlirContext),
        mlir::Attribute::getFromOpaquePointer(reinterpret_cast<const void *>(
            named.second.cast<PythonAttribute>().attr)));

  // Build the list of lists of function argument attributes.
  std::vector<mlir::NamedAttributeList> inputAttrs;
  inputAttrs.reserve(attributedInputs.size());
  for (const auto &in : attributedInputs) {
    std::vector<mlir::NamedAttribute> inAttrs;
    for (const auto &named : in.getNamedAttrs())
      inAttrs.emplace_back(Identifier::get(named.name, &mlirContext),
                           mlir::Attribute::getFromOpaquePointer(
                               reinterpret_cast<const void *>(named.value)));
    inputAttrs.emplace_back(inAttrs);
  }

  // Create the function itself.
  auto *func = new mlir::Function(
      UnknownLoc::get(&mlirContext), name,
      mlir::Type::getFromOpaquePointer(funcType).cast<FunctionType>(), attrs,
      inputAttrs);
  moduleManager.insert(func);
  return func;
}

PythonAttributedType PythonType::attachAttributeDict(
    const std::unordered_map<std::string, PythonAttribute> &attrs) const {
  return PythonAttributedType(*this, attrs);
}

PythonAttribute PythonMLIRModule::integerAttr(PythonType type, int64_t value) {
  return PythonAttribute(::makeIntegerAttr(type, value));
}

PythonAttribute PythonMLIRModule::boolAttr(bool value) {
  return PythonAttribute(::makeBoolAttr(&mlirContext, value));
}

PYBIND11_MODULE(pybind, m) {
  m.doc() =
      "Python bindings for MLIR Embedded Domain-Specific Components (EDSCs)";
  m.def("version", []() { return "EDSC Python extensions v1.0"; });

  py::class_<PythonLoopContext>(
      m, "LoopContext", "A context for building the body of a 'for' loop")
      .def(py::init<PythonValueHandle, PythonValueHandle, int64_t>())
      .def("__enter__", &PythonLoopContext::enter)
      .def("__exit__", &PythonLoopContext::exit);

  py::class_<PythonLoopNestContext>(m, "LoopNestContext",
                                    "A context for building the body of a the "
                                    "innermost loop in a nest of 'for' loops")
      .def(py::init<const std::vector<PythonValueHandle> &,
                    const std::vector<PythonValueHandle> &,
                    const std::vector<int64_t> &>())
      .def("__enter__", &PythonLoopNestContext::enter)
      .def("__exit__", &PythonLoopNestContext::exit);

  m.def("constant_index", [](int64_t val) -> PythonValueHandle {
    return ValueHandle(index_t(val));
  });
  m.def("constant_int", [](int64_t val, int width) -> PythonValueHandle {
    return ValueHandle::create<ConstantIntOp>(val, width);
  });
  m.def("constant_float", [](double val, PythonType type) -> PythonValueHandle {
    FloatType floatType =
        Type::getFromOpaquePointer(type.type).cast<FloatType>();
    assert(floatType);
    auto value = APFloat(val);
    bool lostPrecision;
    value.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &lostPrecision);
    return ValueHandle::create<ConstantFloatOp>(value, floatType);
  });
  m.def("constant_function", [](PythonFunction func) -> PythonValueHandle {
    auto *function = reinterpret_cast<Function *>(func.function);
    auto attr = FunctionAttr::get(function);
    return ValueHandle::create<ConstantOp>(function->getType(), attr);
  });
  m.def("appendTo", [](const PythonBlockHandle &handle) {
    return PythonBlockAppender(handle);
  });
  m.def(
      "ret",
      [](const std::vector<PythonValueHandle> &args) {
        std::vector<ValueHandle> values(args.begin(), args.end());
        (intrinsics::ret(ArrayRef<ValueHandle>{values})); // vexing parse
        return PythonValueHandle(nullptr);
      },
      py::arg("args") = std::vector<PythonValueHandle>());
  m.def(
      "br",
      [](const PythonBlockHandle &dest,
         const std::vector<PythonValueHandle> &args) {
        std::vector<ValueHandle> values(args.begin(), args.end());
        intrinsics::br(dest, values);
        return PythonValueHandle(nullptr);
      },
      py::arg("dest"), py::arg("args") = std::vector<PythonValueHandle>());
  m.def(
      "cond_br",
      [](PythonValueHandle condition, const PythonBlockHandle &trueDest,
         const std::vector<PythonValueHandle> &trueArgs,
         const PythonBlockHandle &falseDest,
         const std::vector<PythonValueHandle> &falseArgs) -> PythonValueHandle {
        std::vector<ValueHandle> trueArguments(trueArgs.begin(),
                                               trueArgs.end());
        std::vector<ValueHandle> falseArguments(falseArgs.begin(),
                                                falseArgs.end());
        intrinsics::cond_br(condition, trueDest, trueArguments, falseDest,
                            falseArguments);
        return PythonValueHandle(nullptr);
      });
  m.def("select",
        [](PythonValueHandle condition, PythonValueHandle trueValue,
           PythonValueHandle falseValue) -> PythonValueHandle {
          return ValueHandle::create<SelectOp>(condition.value, trueValue.value,
                                               falseValue.value);
        });
  m.def("op",
        [](const std::string &name,
           const std::vector<PythonValueHandle> &operands,
           const std::vector<PythonType> &resultTypes,
           const py::kwargs &attributes) -> PythonValueHandle {
          std::vector<ValueHandle> operandHandles(operands.begin(),
                                                  operands.end());
          std::vector<Type> types;
          types.reserve(resultTypes.size());
          for (auto t : resultTypes)
            types.push_back(Type::getFromOpaquePointer(t.type));

          std::vector<NamedAttribute> attrs;
          attrs.reserve(attributes.size());
          for (const auto &a : attributes) {
            std::string name = a.first.str();
            auto pyAttr = a.second.cast<PythonAttribute>();
            auto cppAttr = Attribute::getFromOpaquePointer(pyAttr.attr);
            auto identifier =
                Identifier::get(name, ScopedContext::getContext());
            attrs.emplace_back(identifier, cppAttr);
          }

          return ValueHandle::create(name, operandHandles, types, attrs);
        });

  py::class_<PythonFunction>(m, "Function",
                             "Wrapping class for mlir::Function.")
      .def(py::init<PythonFunction>())
      .def("__str__", &PythonFunction::str)
      .def("define", &PythonFunction::define,
           "Adds a body to the function if it does not already have one.  "
           "Returns true if the body was added")
      .def("arg", &PythonFunction::arg,
           "Get the ValueHandle to the indexed argument of the function");

  py::class_<PythonAttribute>(m, "Attribute",
                              "Wrapping class for mlir::Attribute")
      .def(py::init<PythonAttribute>())
      .def("__str__", &PythonAttribute::str);

  py::class_<PythonType>(m, "Type", "Wrapping class for mlir::Type.")
      .def(py::init<PythonType>())
      .def("__call__", &PythonType::attachAttributeDict,
           "Attach the attributes to these type, making it suitable for "
           "constructing functions with argument attributes")
      .def("__str__", &PythonType::str);

  py::class_<PythonAttributedType>(
      m, "AttributedType",
      "A class containing a wrapped mlir::Type and a wrapped "
      "mlir::NamedAttributeList that are used together, e.g. in function "
      "argument declaration")
      .def(py::init<PythonAttributedType>())
      .def("__str__", &PythonAttributedType::str);

  py::class_<PythonMLIRModule>(
      m, "MLIRModule",
      "An MLIRModule is the abstraction that owns the allocations to support "
      "compilation of a single mlir::Module into an ExecutionEngine backed by "
      "the LLVM ORC JIT. A typical flow consists in creating an MLIRModule, "
      "adding functions, compiling the module to obtain an ExecutionEngine on "
      "which named functions may be called. For now the only means to retrieve "
      "the ExecutionEngine is by calling `get_engine_address`. This mode of "
      "execution is limited to passing the pointer to C++ where the function "
      "is called. Extending the API to allow calling JIT compiled functions "
      "directly require integration with a tensor library (e.g. numpy). This "
      "is left as the prerogative of libraries and frameworks for now.")
      .def(py::init<>())
      .def("boolAttr", &PythonMLIRModule::boolAttr,
           "Creates an mlir::BoolAttr with the given value")
      .def(
          "integerAttr", &PythonMLIRModule::integerAttr,
          "Creates an mlir::IntegerAttr of the given type with the given value "
          "in the context associated with this MLIR module.")
      .def("declare_function", &PythonMLIRModule::declareFunction,
           "Declares a new mlir::Function in the current mlir::Module.  The "
           "function arguments can have attributes.  The function has no "
           "definition and can be linked to an external library.")
      .def("make_function", &PythonMLIRModule::makeFunction,
           "Defines a new mlir::Function in the current mlir::Module.")
      .def("function_context", &PythonMLIRModule::makeFunctionContext,
           "Defines a new mlir::Function in the mlir::Module and creates the "
           "function context for building the body of the function.")
      .def("get_function", &PythonMLIRModule::getNamedFunction,
           "Looks up the function with the given name in the module.")
      .def(
          "make_scalar_type",
          [](PythonMLIRModule &instance, const std::string &type,
             unsigned bitwidth) {
            return instance.makeScalarType(type, bitwidth);
          },
          py::arg("type"), py::arg("bitwidth") = 0,
          "Returns a scalar mlir::Type using the following convention:\n"
          "  - makeScalarType(c, \"bf16\") return an "
          "`mlir::FloatType::getBF16`\n"
          "  - makeScalarType(c, \"f16\") return an `mlir::FloatType::getF16`\n"
          "  - makeScalarType(c, \"f32\") return an `mlir::FloatType::getF32`\n"
          "  - makeScalarType(c, \"f64\") return an `mlir::FloatType::getF64`\n"
          "  - makeScalarType(c, \"index\") return an `mlir::IndexType::get`\n"
          "  - makeScalarType(c, \"i\", bitwidth) return an "
          "`mlir::IntegerType::get(bitwidth)`\n\n"
          " No other combinations are currently supported.")
      .def("make_memref_type", &PythonMLIRModule::makeMemRefType,
           "Returns an mlir::MemRefType of an elemental scalar. -1 is used to "
           "denote symbolic dimensions in the resulting memref shape.")
      .def("make_index_type", &PythonMLIRModule::makeIndexType,
           "Returns an mlir::IndexType")
      .def("compile", &PythonMLIRModule::compile,
           "Compiles the mlir::Module to LLVMIR a creates new opaque "
           "ExecutionEngine backed by the ORC JIT.")
      .def("get_ir", &PythonMLIRModule::getIR,
           "Returns a dump of the MLIR representation of the module. This is "
           "used for serde to support out-of-process execution as well as "
           "debugging purposes.")
      .def("get_engine_address", &PythonMLIRModule::getEngineAddress,
           "Returns the address of the compiled ExecutionEngine. This is used "
           "for in-process execution.")
      .def("__str__", &PythonMLIRModule::getIR,
           "Get the string representation of the module");

  py::class_<PythonFunctionContext>(
      m, "FunctionContext", "A wrapper around mlir::edsc::ScopedContext")
      .def(py::init<PythonFunction>())
      .def("__enter__", &PythonFunctionContext::enter)
      .def("__exit__", &PythonFunctionContext::exit);

  {
    using namespace mlir::edsc::op;
    py::class_<PythonValueHandle>(m, "ValueHandle",
                                  "A wrapper around mlir::edsc::ValueHandle")
        .def(py::init<PythonType>())
        .def(py::init<PythonValueHandle>())
        .def("__add__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value + rhs.value; })
        .def("__sub__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value - rhs.value; })
        .def("__mul__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value * rhs.value; })
        .def("__div__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value / rhs.value; })
        .def("__truediv__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value / rhs.value; })
        .def("__floordiv__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return floorDiv(lhs, rhs); })
        .def("__mod__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value % rhs.value; })
        .def("__lt__",
             [](PythonValueHandle lhs,
                PythonValueHandle rhs) -> PythonValueHandle {
               return ValueHandle::create<CmpIOp>(CmpIPredicate::SLT, lhs.value,
                                                  rhs.value);
             })
        .def("__le__",
             [](PythonValueHandle lhs,
                PythonValueHandle rhs) -> PythonValueHandle {
               return ValueHandle::create<CmpIOp>(CmpIPredicate::SLE, lhs.value,
                                                  rhs.value);
             })
        .def("__gt__",
             [](PythonValueHandle lhs,
                PythonValueHandle rhs) -> PythonValueHandle {
               return ValueHandle::create<CmpIOp>(CmpIPredicate::SGT, lhs.value,
                                                  rhs.value);
             })
        .def("__ge__",
             [](PythonValueHandle lhs,
                PythonValueHandle rhs) -> PythonValueHandle {
               return ValueHandle::create<CmpIOp>(CmpIPredicate::SGE, lhs.value,
                                                  rhs.value);
             })
        .def("__eq__",
             [](PythonValueHandle lhs,
                PythonValueHandle rhs) -> PythonValueHandle {
               return ValueHandle::create<CmpIOp>(CmpIPredicate::EQ, lhs.value,
                                                  rhs.value);
             })
        .def("__ne__",
             [](PythonValueHandle lhs,
                PythonValueHandle rhs) -> PythonValueHandle {
               return ValueHandle::create<CmpIOp>(CmpIPredicate::NE, lhs.value,
                                                  rhs.value);
             })
        .def("__invert__",
             [](PythonValueHandle handle) -> PythonValueHandle {
               return !handle.value;
             })
        .def("__and__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value && rhs.value; })
        .def("__or__",
             [](PythonValueHandle lhs, PythonValueHandle rhs)
                 -> PythonValueHandle { return lhs.value || rhs.value; })
        .def("__call__", &PythonValueHandle::call);
  }

  py::class_<PythonBlockAppender>(
      m, "BlockAppender",
      "A dummy class signaling BlockContext to append IR to the given block "
      "instead of creating a new block")
      .def(py::init<const PythonBlockHandle &>());
  py::class_<PythonBlockHandle>(m, "BlockHandle",
                                "A wrapper around mlir::edsc::BlockHandle")
      .def(py::init<PythonBlockHandle>())
      .def("arg", &PythonBlockHandle::arg);

  py::class_<PythonBlockContext>(m, "BlockContext",
                                 "A wrapper around mlir::edsc::BlockBuilder")
      .def(py::init<>())
      .def(py::init<const std::vector<PythonType> &>())
      .def(py::init<const PythonBlockAppender &>())
      .def("__enter__", &PythonBlockContext::enter)
      .def("__exit__", &PythonBlockContext::exit)
      .def("handle", &PythonBlockContext::getHandle);

  py::class_<PythonIndexedValue>(m, "IndexedValue",
                                 "A wrapper around mlir::edsc::IndexedValue")
      .def(py::init<PythonValueHandle>())
      .def("load", &PythonIndexedValue::load)
      .def("store", &PythonIndexedValue::store);
}

} // namespace python
} // namespace edsc
} // namespace mlir
