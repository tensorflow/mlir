//===- Serializer.cpp - MLIR SPIR-V Serialization -------------------------===//
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
// This file defines the MLIR SPIR-V module to SPIR-V binary seralization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Serialization.h"

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Returns the word-count-prefixed opcode for an SPIR-V instruction.
static inline uint32_t getPrefixedOpcode(uint32_t wordCount,
                                         spirv::Opcode opcode) {
  assert(((wordCount >> 16) == 0) && "word count out of range!");
  return (wordCount << 16) | static_cast<uint32_t>(opcode);
}

/// Encodes an SPIR-V instruction with the given `opcode` and `operands` into
/// the given `binary` vector.
static LogicalResult encodeInstructionInto(SmallVectorImpl<uint32_t> &binary,
                                           spirv::Opcode op,
                                           ArrayRef<uint32_t> operands) {
  uint32_t wordCount = 1 + operands.size();
  binary.push_back(getPrefixedOpcode(wordCount, op));
  if (!operands.empty()) {
    binary.append(operands.begin(), operands.end());
  }
  return success();
}

/// Encodes an SPIR-V `literal` string into the given `binary` vector.
static LogicalResult encodeStringLiteralInto(SmallVectorImpl<uint32_t> &binary,
                                             StringRef literal) {
  // We need to encode the literal and the null termination.
  auto encodingSize = literal.size() / 4 + 1;
  auto bufferStartSize = binary.size();
  binary.resize(bufferStartSize + encodingSize, 0);
  std::memcpy(binary.data() + bufferStartSize, literal.data(), literal.size());
  return success();
}

namespace {

/// A SPIR-V module serializer.
///
/// A SPIR-V binary module is a single linear stream of instructions; each
/// instruction is composed of 32-bit words with the layout:
///
///   | <word-count>|<opcode> |  <operand>   |  <operand>   | ... |
///   | <------ word -------> | <-- word --> | <-- word --> | ... |
///
/// For the first word, the 16 high-order bits are the word count of the
/// instruction, the 16 low-order bits are the opcode enumerant. The
/// instructions then belong to different sections, which must be laid out in
/// the particular order as specified in "2.4 Logical Layout of a Module" of
/// the SPIR-V spec.
class Serializer {
public:
  /// Creates a serializer for the given SPIR-V `module`.
  explicit Serializer(spirv::ModuleOp module);

  /// Serializes the remembered SPIR-V module.
  LogicalResult serialize();

  /// Collects the final SPIR-V `binary`.
  void collect(SmallVectorImpl<uint32_t> &binary);

private:
  // Note that there are two main categories of methods in this class:
  // * process*() methods are meant to fully serialize a SPIR-V module entity
  //   (header, type, op, etc.). They update internal vectors containing
  //   different binary sections. They are not meant to be called except the
  //   top-level serialization loop.
  // * prepare*() methods are meant to be helpers that prepare for serializing
  //   certain entity. They may or may not update internal vectors containing
  //   different binary sections. They are meant to be called among themselves
  //   or by other process*() methods for subtasks.

  //===--------------------------------------------------------------------===//
  // <id>
  //===--------------------------------------------------------------------===//

  // Note that it is illegal to use id <0> in SPIR-V binary module. Various
  // methods in this class, if using SPIR-V word (uint32_t) as interface,
  // check or return id <0> to indicate error in processing.

  /// Consumes the next unused <id>. This method will never return 0.
  uint32_t getNextID() { return nextID++; }

  //===--------------------------------------------------------------------===//
  // Module structure
  //===--------------------------------------------------------------------===//

  uint32_t findSpecConstID(StringRef constName) const {
    return specConstIDMap.lookup(constName);
  }

  uint32_t findVariableID(StringRef varName) const {
    return globalVarIDMap.lookup(varName);
  }

  uint32_t findFunctionID(StringRef fnName) const {
    return funcIDMap.lookup(fnName);
  }

  void processCapability();

  void processMemoryModel();

  LogicalResult processConstantOp(spirv::ConstantOp op);

  LogicalResult processSpecConstantOp(spirv::SpecConstantOp op);

  /// Emit OpName for the given `resultID`.
  LogicalResult processName(uint32_t resultID, StringRef name);

  /// Processes a SPIR-V function op.
  LogicalResult processFuncOp(FuncOp op);

  /// Process a SPIR-V GlobalVariableOp
  LogicalResult processGlobalVariableOp(spirv::GlobalVariableOp varOp);

  /// Process attributes that translate to decorations on the result <id>
  LogicalResult processDecoration(Location loc, uint32_t resultID,
                                  NamedAttribute attr);

  template <typename DType>
  LogicalResult processTypeDecoration(Location loc, DType type,
                                      uint32_t resultId) {
    return emitError(loc, "unhandled decoraion for type:") << type;
  }

  /// Process member decoration
  LogicalResult processMemberDecoration(uint32_t structID, uint32_t memberNum,
                                        spirv::Decoration decorationType,
                                        uint32_t value);

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  uint32_t findTypeID(Type type) const { return typeIDMap.lookup(type); }

  Type getVoidType() { return mlirBuilder.getNoneType(); }

  bool isVoidType(Type type) const { return type.isa<NoneType>(); }

  /// Main dispatch method for serializing a type. The result <id> of the
  /// serialized type will be returned as `typeID`.
  LogicalResult processType(Location loc, Type type, uint32_t &typeID);

  /// Method for preparing basic SPIR-V type serialization. Returns the type's
  /// opcode and operands for the instruction via `typeEnum` and `operands`.
  LogicalResult prepareBasicType(Location loc, Type type, uint32_t resultID,
                                 spirv::Opcode &typeEnum,
                                 SmallVectorImpl<uint32_t> &operands);

  LogicalResult prepareFunctionType(Location loc, FunctionType type,
                                    spirv::Opcode &typeEnum,
                                    SmallVectorImpl<uint32_t> &operands);

  //===--------------------------------------------------------------------===//
  // Constant
  //===--------------------------------------------------------------------===//

  uint32_t findConstantID(Attribute value) const {
    return constIDMap.lookup(value);
  }

  /// Main dispatch method for processing a constant with the given `constType`
  /// and `valueAttr`. `constType` is needed here because we can interpret the
  /// `valueAttr` as a different type than the type of `valueAttr` itself; for
  /// example, ArrayAttr, whose type is NoneType, is used for spirv::ArrayType
  /// constants.
  uint32_t prepareConstant(Location loc, Type constType, Attribute valueAttr);

  /// Prepares bool ElementsAttr serialization. This method updates `opcode`
  /// with a proper OpConstant* instruction and pushes literal values for the
  /// constant to `operands`.
  LogicalResult prepareBoolVectorConstant(Location loc,
                                          DenseIntElementsAttr elementsAttr,
                                          spirv::Opcode &opcode,
                                          SmallVectorImpl<uint32_t> &operands);

  /// Prepares int ElementsAttr serialization. This method updates `opcode` with
  /// a proper OpConstant* instruction and pushes literal values for the
  /// constant to `operands`.
  LogicalResult prepareIntVectorConstant(Location loc,
                                         DenseIntElementsAttr elementsAttr,
                                         spirv::Opcode &opcode,
                                         SmallVectorImpl<uint32_t> &operands);

  /// Prepares float ElementsAttr serialization. This method updates `opcode`
  /// with a proper OpConstant* instruction and pushes literal values for the
  /// constant to `operands`.
  LogicalResult prepareFloatVectorConstant(Location loc,
                                           DenseFPElementsAttr elementsAttr,
                                           spirv::Opcode &opcode,
                                           SmallVectorImpl<uint32_t> &operands);

  /// Prepares scalar attribute serialization. This method emits corresponding
  /// OpConstant* and returns the result <id> associated with it. Returns 0 if
  /// the attribute is not for a scalar bool/integer/float value. If `isSpec` is
  /// true, then the constant will be serialized as a specialization constant.
  uint32_t prepareConstantScalar(Location loc, Attribute valueAttr,
                                 bool isSpec = false);

  uint32_t prepareConstantBool(Location loc, BoolAttr boolAttr,
                               bool isSpec = false);

  uint32_t prepareConstantInt(Location loc, IntegerAttr intAttr,
                              bool isSpec = false);

  uint32_t prepareConstantFp(Location loc, FloatAttr floatAttr,
                             bool isSpec = false);

  //===--------------------------------------------------------------------===//
  // Operations
  //===--------------------------------------------------------------------===//

  uint32_t findValueID(Value *val) const { return valueIDMap.lookup(val); }

  LogicalResult processAddressOfOp(spirv::AddressOfOp addressOfOp);

  LogicalResult processReferenceOfOp(spirv::ReferenceOfOp referenceOfOp);

  /// Main dispatch method for serializing an operation.
  LogicalResult processOperation(Operation *op);

  /// Method to dispatch to the serialization function for an operation in
  /// SPIR-V dialect that is a mirror of an instruction in the SPIR-V spec.
  /// This is auto-generated from ODS. Dispatch is handled for all operations
  /// in SPIR-V dialect that have hasOpcode == 1.
  LogicalResult dispatchToAutogenSerialization(Operation *op);

  /// Method to serialize an operation in the SPIR-V dialect that is a mirror of
  /// an instruction in the SPIR-V spec. This is auto generated if hasOpcode ==
  /// 1 and autogenSerialization == 1 in ODS.
  template <typename OpTy> LogicalResult processOp(OpTy op) {
    return op.emitError("unsupported op serialization");
  }

private:
  /// The SPIR-V module to be serialized.
  spirv::ModuleOp module;

  /// An MLIR builder for getting MLIR constructs.
  mlir::Builder mlirBuilder;

  /// The next available result <id>.
  uint32_t nextID = 1;

  // The following are for different SPIR-V instruction sections. They follow
  // the logical layout of a SPIR-V module.

  SmallVector<uint32_t, 4> capabilities;
  SmallVector<uint32_t, 0> extensions;
  SmallVector<uint32_t, 0> extendedSets;
  SmallVector<uint32_t, 3> memoryModel;
  SmallVector<uint32_t, 0> entryPoints;
  SmallVector<uint32_t, 4> executionModes;
  // TODO(antiagainst): debug instructions
  SmallVector<uint32_t, 0> names;
  SmallVector<uint32_t, 0> decorations;
  SmallVector<uint32_t, 0> typesGlobalValues;
  SmallVector<uint32_t, 0> functions;

  /// Map from type used in SPIR-V module to their <id>s.
  DenseMap<Type, uint32_t> typeIDMap;

  /// Map from constant values to their <id>s.
  DenseMap<Attribute, uint32_t> constIDMap;

  /// Map from specialization constant names to their <id>s.
  llvm::StringMap<uint32_t> specConstIDMap;

  /// Map from GlobalVariableOps name to <id>s.
  llvm::StringMap<uint32_t> globalVarIDMap;

  /// Map from FuncOps name to <id>s.
  llvm::StringMap<uint32_t> funcIDMap;

  /// Map from results of normal operations to their <id>s.
  DenseMap<Value *, uint32_t> valueIDMap;
};
} // namespace

Serializer::Serializer(spirv::ModuleOp module)
    : module(module), mlirBuilder(module.getContext()) {}

LogicalResult Serializer::serialize() {
  if (failed(module.verify()))
    return failure();

  // TODO(antiagainst): handle the other sections
  processCapability();
  processMemoryModel();

  // Iterate over the module body to serialze it. Assumptions are that there is
  // only one basic block in the moduleOp
  for (auto &op : module.getBlock()) {
    if (failed(processOperation(&op))) {
      return failure();
    }
  }
  return success();
}

void Serializer::collect(SmallVectorImpl<uint32_t> &binary) {
  auto moduleSize = spirv::kHeaderWordCount + capabilities.size() +
                    extensions.size() + extendedSets.size() +
                    memoryModel.size() + entryPoints.size() +
                    executionModes.size() + decorations.size() +
                    typesGlobalValues.size() + functions.size();

  binary.clear();
  binary.reserve(moduleSize);

  spirv::appendModuleHeader(binary, nextID);
  binary.append(capabilities.begin(), capabilities.end());
  binary.append(extensions.begin(), extensions.end());
  binary.append(extendedSets.begin(), extendedSets.end());
  binary.append(memoryModel.begin(), memoryModel.end());
  binary.append(entryPoints.begin(), entryPoints.end());
  binary.append(executionModes.begin(), executionModes.end());
  binary.append(names.begin(), names.end());
  binary.append(decorations.begin(), decorations.end());
  binary.append(typesGlobalValues.begin(), typesGlobalValues.end());
  binary.append(functions.begin(), functions.end());
}
//===----------------------------------------------------------------------===//
// Module structure
//===----------------------------------------------------------------------===//

void Serializer::processCapability() {
  auto caps = module.getAttrOfType<ArrayAttr>("capabilities");
  if (!caps)
    return;

  for (auto cap : caps.getValue()) {
    auto capStr = cap.cast<StringAttr>().getValue();
    auto capVal = spirv::symbolizeCapability(capStr);
    encodeInstructionInto(capabilities, spirv::Opcode::OpCapability,
                          {static_cast<uint32_t>(*capVal)});
  }
}

void Serializer::processMemoryModel() {
  uint32_t mm = module.getAttrOfType<IntegerAttr>("memory_model").getInt();
  uint32_t am = module.getAttrOfType<IntegerAttr>("addressing_model").getInt();

  encodeInstructionInto(memoryModel, spirv::Opcode::OpMemoryModel, {am, mm});
}

LogicalResult Serializer::processConstantOp(spirv::ConstantOp op) {
  if (auto resultID = prepareConstant(op.getLoc(), op.getType(), op.value())) {
    valueIDMap[op.getResult()] = resultID;
    return success();
  }
  return failure();
}

LogicalResult Serializer::processSpecConstantOp(spirv::SpecConstantOp op) {
  if (auto resultID = prepareConstantScalar(op.getLoc(), op.default_value(),
                                            /*isSpec=*/true)) {
    specConstIDMap[op.sym_name()] = resultID;
    return processName(resultID, op.sym_name());
  }
  return failure();
}

LogicalResult Serializer::processDecoration(Location loc, uint32_t resultID,
                                            NamedAttribute attr) {
  auto attrName = attr.first.strref();
  auto decorationName = mlir::convertToCamelCase(attrName, true);
  auto decoration = spirv::symbolizeDecoration(decorationName);
  if (!decoration) {
    return emitError(
               loc, "non-argument attributes expected to have snake-case-ified "
                    "decoration name, unhandled attribute with name : ")
           << attrName;
  }
  SmallVector<uint32_t, 1> args;
  args.push_back(resultID);
  args.push_back(static_cast<uint32_t>(decoration.getValue()));
  switch (decoration.getValue()) {
  case spirv::Decoration::DescriptorSet:
  case spirv::Decoration::Binding:
    if (auto intAttr = attr.second.dyn_cast<IntegerAttr>()) {
      args.push_back(intAttr.getValue().getZExtValue());
      break;
    }
    return emitError(loc, "expected integer attribute for ") << attrName;
  case spirv::Decoration::BuiltIn:
    if (auto strAttr = attr.second.dyn_cast<StringAttr>()) {
      auto enumVal = spirv::symbolizeBuiltIn(strAttr.getValue());
      if (enumVal) {
        args.push_back(static_cast<uint32_t>(enumVal.getValue()));
        break;
      }
      return emitError(loc, "invalid ")
             << attrName << " attribute " << strAttr.getValue();
    }
    return emitError(loc, "expected string attribute for ") << attrName;
  default:
    return emitError(loc, "unhandled decoration ") << decorationName;
  }
  return encodeInstructionInto(decorations, spirv::Opcode::OpDecorate, args);
}

LogicalResult Serializer::processName(uint32_t resultID, StringRef name) {
  assert(!name.empty() && "unexpected empty string for OpName");

  SmallVector<uint32_t, 4> nameOperands;
  nameOperands.push_back(resultID);
  if (failed(encodeStringLiteralInto(nameOperands, name))) {
    return failure();
  }
  return encodeInstructionInto(names, spirv::Opcode::OpName, nameOperands);
}

namespace {
template <>
LogicalResult Serializer::processTypeDecoration<spirv::ArrayType>(
    Location loc, spirv::ArrayType type, uint32_t resultID) {
  if (type.hasLayout()) {
    // OpDecorate %arrayTypeSSA ArrayStride strideLiteral
    SmallVector<uint32_t, 3> args;
    args.push_back(resultID);
    args.push_back(static_cast<uint32_t>(spirv::Decoration::ArrayStride));
    args.push_back(type.getArrayStride());
    return encodeInstructionInto(decorations, spirv::Opcode::OpDecorate, args);
  }
  return success();
}

LogicalResult
Serializer::processMemberDecoration(uint32_t structID, uint32_t memberIndex,
                                    spirv::Decoration decorationType,
                                    uint32_t value) {
  SmallVector<uint32_t, 4> args(
      {structID, memberIndex, static_cast<uint32_t>(decorationType), value});
  return encodeInstructionInto(decorations, spirv::Opcode::OpMemberDecorate,
                               args);
}
} // namespace

LogicalResult Serializer::processFuncOp(FuncOp op) {
  uint32_t fnTypeID = 0;
  // Generate type of the function.
  processType(op.getLoc(), op.getType(), fnTypeID);

  // Add the function definition.
  SmallVector<uint32_t, 4> operands;
  uint32_t resTypeID = 0;
  auto resultTypes = op.getType().getResults();
  if (resultTypes.size() > 1) {
    return emitError(op.getLoc(),
                     "cannot serialize function with multiple return types");
  }
  if (failed(processType(op.getLoc(),
                         (resultTypes.empty() ? getVoidType() : resultTypes[0]),
                         resTypeID))) {
    return failure();
  }
  operands.push_back(resTypeID);
  auto funcID = getNextID();
  funcIDMap[op.getName()] = funcID;
  operands.push_back(funcID);
  // TODO : Support other function control options.
  operands.push_back(static_cast<uint32_t>(spirv::FunctionControl::None));
  operands.push_back(fnTypeID);
  encodeInstructionInto(functions, spirv::Opcode::OpFunction, operands);

  // Add function name.
  if (failed(processName(funcID, op.getName()))) {
    return failure();
  }

  // Declare the parameters.
  for (auto arg : op.getArguments()) {
    uint32_t argTypeID = 0;
    if (failed(processType(op.getLoc(), arg->getType(), argTypeID))) {
      return failure();
    }
    auto argValueID = getNextID();
    valueIDMap[arg] = argValueID;
    encodeInstructionInto(functions, spirv::Opcode::OpFunctionParameter,
                          {argTypeID, argValueID});
  }

  // Process the body.
  if (op.isExternal()) {
    return emitError(op.getLoc(), "external function is unhandled");
  }

  for (auto &b : op) {
    for (auto &op : b) {
      if (failed(processOperation(&op))) {
        return failure();
      }
    }
  }

  // Insert Function End.
  return encodeInstructionInto(functions, spirv::Opcode::OpFunctionEnd, {});
}

LogicalResult
Serializer::processGlobalVariableOp(spirv::GlobalVariableOp varOp) {
  // Get TypeID.
  uint32_t resultTypeID = 0;
  SmallVector<StringRef, 4> elidedAttrs;
  if (failed(processType(varOp.getLoc(), varOp.type(), resultTypeID))) {
    return failure();
  }
  elidedAttrs.push_back("type");
  SmallVector<uint32_t, 4> operands;
  operands.push_back(resultTypeID);
  auto resultID = getNextID();

  // Encode the name.
  auto varName = varOp.sym_name();
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());
  if (failed(processName(resultID, varName))) {
    return failure();
  }
  globalVarIDMap[varName] = resultID;
  operands.push_back(resultID);

  // Encode StorageClass.
  operands.push_back(static_cast<uint32_t>(varOp.storageClass()));

  // Encode initialization.
  if (auto initializer = varOp.initializer()) {
    auto initializerID = findVariableID(initializer.getValue());
    if (!initializerID) {
      return emitError(varOp.getLoc(),
                       "invalid usage of undefined variable as initializer");
    }
    operands.push_back(initializerID);
    elidedAttrs.push_back("initializer");
  }

  if (failed(encodeInstructionInto(functions, spirv::Opcode::OpVariable,
                                   operands))) {
    elidedAttrs.push_back("initializer");
    return failure();
  }

  // Encode decorations.
  for (auto attr : varOp.getAttrs()) {
    if (llvm::any_of(elidedAttrs,
                     [&](StringRef elided) { return attr.first.is(elided); })) {
      continue;
    }
    if (failed(processDecoration(varOp.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

LogicalResult Serializer::processType(Location loc, Type type,
                                      uint32_t &typeID) {
  typeID = findTypeID(type);
  if (typeID) {
    return success();
  }
  typeID = getNextID();
  SmallVector<uint32_t, 4> operands;
  operands.push_back(typeID);
  auto typeEnum = spirv::Opcode::OpTypeVoid;
  if ((type.isa<FunctionType>() &&
       succeeded(prepareFunctionType(loc, type.cast<FunctionType>(), typeEnum,
                                     operands))) ||
      succeeded(prepareBasicType(loc, type, typeID, typeEnum, operands))) {
    typeIDMap[type] = typeID;
    return encodeInstructionInto(typesGlobalValues, typeEnum, operands);
  }
  return failure();
}

LogicalResult
Serializer::prepareBasicType(Location loc, Type type, uint32_t resultID,
                             spirv::Opcode &typeEnum,
                             SmallVectorImpl<uint32_t> &operands) {
  if (isVoidType(type)) {
    typeEnum = spirv::Opcode::OpTypeVoid;
    return success();
  }

  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      typeEnum = spirv::Opcode::OpTypeBool;
      return success();
    }

    typeEnum = spirv::Opcode::OpTypeInt;
    operands.push_back(intType.getWidth());
    // TODO(antiagainst): support unsigned integers
    operands.push_back(1);
    return success();
  }

  if (auto floatType = type.dyn_cast<FloatType>()) {
    typeEnum = spirv::Opcode::OpTypeFloat;
    operands.push_back(floatType.getWidth());
    return success();
  }

  if (auto vectorType = type.dyn_cast<VectorType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processType(loc, vectorType.getElementType(), elementTypeID))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeVector;
    operands.push_back(elementTypeID);
    operands.push_back(vectorType.getNumElements());
    return success();
  }

  if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    typeEnum = spirv::Opcode::OpTypeArray;
    uint32_t elementTypeID = 0;
    if (failed(processType(loc, arrayType.getElementType(), elementTypeID))) {
      return failure();
    }
    operands.push_back(elementTypeID);
    if (auto elementCountID = prepareConstantInt(
            loc, mlirBuilder.getI32IntegerAttr(arrayType.getNumElements()))) {
      operands.push_back(elementCountID);
    }
    return processTypeDecoration(loc, arrayType, resultID);
  }

  if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    uint32_t pointeeTypeID = 0;
    if (failed(processType(loc, ptrType.getPointeeType(), pointeeTypeID))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypePointer;
    operands.push_back(static_cast<uint32_t>(ptrType.getStorageClass()));
    operands.push_back(pointeeTypeID);
    return success();
  }

  if (auto structType = type.dyn_cast<spirv::StructType>()) {
    bool hasLayout = structType.hasLayout();
    for (auto elementIndex :
         llvm::seq<uint32_t>(0, structType.getNumElements())) {
      uint32_t elementTypeID = 0;
      if (failed(processType(loc, structType.getElementType(elementIndex),
                             elementTypeID))) {
        return failure();
      }
      operands.push_back(elementTypeID);
      if (hasLayout) {
        // Decorate each struct member with an offset
        if (failed(processMemberDecoration(
                resultID, elementIndex, spirv::Decoration::Offset,
                static_cast<uint32_t>(structType.getOffset(elementIndex))))) {
          return emitError(loc, "cannot decorate ")
                 << elementIndex << "-th member of : " << structType
                 << "with its offset";
        }
      }
    }
    typeEnum = spirv::Opcode::OpTypeStruct;
    return success();
  }

  // TODO(ravishankarm) : Handle other types.
  return emitError(loc, "unhandled type in serialization: ") << type;
}

LogicalResult
Serializer::prepareFunctionType(Location loc, FunctionType type,
                                spirv::Opcode &typeEnum,
                                SmallVectorImpl<uint32_t> &operands) {
  typeEnum = spirv::Opcode::OpTypeFunction;
  assert(type.getNumResults() <= 1 &&
         "Serialization supports only a single return value");
  uint32_t resultID = 0;
  if (failed(processType(
          loc, type.getNumResults() == 1 ? type.getResult(0) : getVoidType(),
          resultID))) {
    return failure();
  }
  operands.push_back(resultID);
  for (auto &res : type.getInputs()) {
    uint32_t argTypeID = 0;
    if (failed(processType(loc, res, argTypeID))) {
      return failure();
    }
    operands.push_back(argTypeID);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

uint32_t Serializer::prepareConstant(Location loc, Type constType,
                                     Attribute valueAttr) {
  if (auto id = prepareConstantScalar(loc, valueAttr)) {
    return id;
  }
  // This is a composite literal. We need to handle each component separately
  // and then emit an OpConstantComposite for the whole.

  if (auto id = findConstantID(valueAttr)) {
    return id;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }
  auto resultID = getNextID();

  spirv::Opcode opcode = spirv::Opcode::OpNop;
  SmallVector<uint32_t, 4> operands;
  operands.push_back(typeID);
  operands.push_back(resultID);

  if (auto vectorAttr = valueAttr.dyn_cast<DenseIntElementsAttr>()) {
    if (vectorAttr.getType().getElementType().isInteger(1)) {
      if (failed(prepareBoolVectorConstant(loc, vectorAttr, opcode, operands)))
        return 0;
    } else if (failed(
                   prepareIntVectorConstant(loc, vectorAttr, opcode, operands)))
      return 0;
  } else if (auto vectorAttr = valueAttr.dyn_cast<DenseFPElementsAttr>()) {
    if (failed(prepareFloatVectorConstant(loc, vectorAttr, opcode, operands)))
      return 0;
  } else if (auto arrayAttr = valueAttr.dyn_cast<ArrayAttr>()) {
    opcode = spirv::Opcode::OpConstantComposite;
    operands.reserve(arrayAttr.size() + 2);

    auto elementType = constType.cast<spirv::ArrayType>().getElementType();
    for (Attribute elementAttr : arrayAttr)
      if (auto elementID = prepareConstant(loc, elementType, elementAttr)) {
        operands.push_back(elementID);
      } else {
        return 0;
      }
  } else {
    emitError(loc, "cannot serialize attribute: ") << valueAttr;
    return 0;
  }

  encodeInstructionInto(typesGlobalValues, opcode, operands);
  constIDMap[valueAttr] = resultID;
  return resultID;
}

LogicalResult Serializer::prepareBoolVectorConstant(
    Location loc, DenseIntElementsAttr elementsAttr, spirv::Opcode &opcode,
    SmallVectorImpl<uint32_t> &operands) {
  auto type = elementsAttr.getType();
  assert(type.hasRank() && type.getRank() == 1 &&
         "spv.constant should have verified only vector literal uses "
         "ElementsAttr");
  assert(type.getElementType().isInteger(1) && "must be bool ElementsAttr");
  auto count = type.getNumElements();

  // Operands for constructing the SPIR-V OpConstant* instruction
  operands.reserve(count + 2);

  // For splat cases, we don't need to loop over all elements, especially when
  // the splat value is zero.
  if (elementsAttr.isSplat()) {
    // We can use OpConstantNull if this bool ElementsAttr is splatting false.
    if (!elementsAttr.getSplatValue<bool>()) {
      opcode = spirv::Opcode::OpConstantNull;
      return success();
    }

    if (auto id =
            prepareConstantBool(loc, elementsAttr.getSplatValue<BoolAttr>())) {
      opcode = spirv::Opcode::OpConstantComposite;
      operands.append(count, id);
      return success();
    }

    return failure();
  }

  // Otherwise, we need to process each element and compose them with
  // OpConstantComposite.
  opcode = spirv::Opcode::OpConstantComposite;
  for (auto boolAttr : elementsAttr.getValues<BoolAttr>()) {
    // We are constructing an BoolAttr for each value here. But given that
    // we only use ElementsAttr for vectors with no more than 4 elements, it
    // should be fine here.
    if (auto elementID = prepareConstantBool(loc, boolAttr)) {
      operands.push_back(elementID);
    } else {
      return failure();
    }
  }
  return success();
}

LogicalResult Serializer::prepareIntVectorConstant(
    Location loc, DenseIntElementsAttr elementsAttr, spirv::Opcode &opcode,
    SmallVectorImpl<uint32_t> &operands) {
  auto type = elementsAttr.getType();
  assert(type.hasRank() && type.getRank() == 1 &&
         "spv.constant should have verified only vector literal uses "
         "ElementsAttr");
  assert(!type.getElementType().isInteger(1) &&
         "must be non-bool ElementsAttr");
  auto count = type.getNumElements();

  // Operands for constructing the SPIR-V OpConstant* instruction
  operands.reserve(count + 2);

  // For splat cases, we don't need to loop over all elements, especially when
  // the splat value is zero.
  if (elementsAttr.isSplat()) {
    auto splatAttr = elementsAttr.getSplatValue<IntegerAttr>();

    // We can use OpConstantNull if this int ElementsAttr is splatting 0.
    if (splatAttr.getValue().isNullValue()) {
      opcode = spirv::Opcode::OpConstantNull;
      return success();
    }

    if (auto id = prepareConstantInt(loc, splatAttr)) {
      opcode = spirv::Opcode::OpConstantComposite;
      operands.append(count, id);
      return success();
    }
    return failure();
  }

  // Otherwise, we need to process each element and compose them with
  // OpConstantComposite.
  opcode = spirv::Opcode::OpConstantComposite;
  for (auto intAttr : elementsAttr.getValues<IntegerAttr>()) {
    // We are constructing an IntegerAttr for each value here. But given that
    // we only use ElementsAttr for vectors with no more than 4 elements, it
    // should be fine here.
    // TODO(antiagainst): revisit this if special extensions enabling large
    // vectors are supported.
    if (auto elementID = prepareConstantInt(loc, intAttr)) {
      operands.push_back(elementID);
    } else {
      return failure();
    }
  }
  return success();
}

LogicalResult Serializer::prepareFloatVectorConstant(
    Location loc, DenseFPElementsAttr elementsAttr, spirv::Opcode &opcode,
    SmallVectorImpl<uint32_t> &operands) {
  auto type = elementsAttr.getType();
  assert(type.hasRank() && type.getRank() == 1 &&
         "spv.constant should have verified only vector literal uses "
         "ElementsAttr");
  auto count = type.getNumElements();

  operands.reserve(count + 2);

  if (elementsAttr.isSplat()) {
    FloatAttr splatAttr = elementsAttr.getSplatValue<FloatAttr>();
    if (splatAttr.getValue().isZero()) {
      opcode = spirv::Opcode::OpConstantNull;
      return success();
    }

    if (auto id = prepareConstantFp(loc, splatAttr)) {
      opcode = spirv::Opcode::OpConstantComposite;
      operands.append(count, id);
      return success();
    }

    return failure();
  }

  opcode = spirv::Opcode::OpConstantComposite;
  for (auto fpAttr : elementsAttr.getValues<FloatAttr>()) {
    if (auto elementID = prepareConstantFp(loc, fpAttr)) {
      operands.push_back(elementID);
    } else {
      return failure();
    }
  }
  return success();
}

uint32_t Serializer::prepareConstantScalar(Location loc, Attribute valueAttr,
                                           bool isSpec) {
  if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>()) {
    return prepareConstantFp(loc, floatAttr, isSpec);
  }
  if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
    return prepareConstantInt(loc, intAttr, isSpec);
  }
  if (auto boolAttr = valueAttr.dyn_cast<BoolAttr>()) {
    return prepareConstantBool(loc, boolAttr, isSpec);
  }

  return 0;
}

uint32_t Serializer::prepareConstantBool(Location loc, BoolAttr boolAttr,
                                         bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate nomral contants, but not specialization constants.
    if (auto id = findConstantID(boolAttr)) {
      return id;
    }
  }

  // Process the type for this bool literal
  uint32_t typeID = 0;
  if (failed(processType(loc, boolAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  auto opcode = boolAttr.getValue()
                    ? (isSpec ? spirv::Opcode::OpSpecConstantTrue
                              : spirv::Opcode::OpConstantTrue)
                    : (isSpec ? spirv::Opcode::OpSpecConstantFalse
                              : spirv::Opcode::OpConstantFalse);
  encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID});

  if (!isSpec) {
    constIDMap[boolAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareConstantInt(Location loc, IntegerAttr intAttr,
                                        bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate nomral contants, but not specialization constants.
    if (auto id = findConstantID(intAttr)) {
      return id;
    }
  }

  // Process the type for this integer literal
  uint32_t typeID = 0;
  if (failed(processType(loc, intAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  APInt value = intAttr.getValue();
  unsigned bitwidth = value.getBitWidth();
  bool isSigned = value.isSignedIntN(bitwidth);

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  // According to SPIR-V spec, "When the type's bit width is less than 32-bits,
  // the literal's value appears in the low-order bits of the word, and the
  // high-order bits must be 0 for a floating-point type, or 0 for an integer
  // type with Signedness of 0, or sign extended when Signedness is 1."
  if (bitwidth == 32 || bitwidth == 16) {
    uint32_t word = 0;
    if (isSigned) {
      word = static_cast<int32_t>(value.getSExtValue());
    } else {
      word = static_cast<uint32_t>(value.getZExtValue());
    }
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  }
  // According to SPIR-V spec: "When the type's bit width is larger than one
  // word, the literal’s low-order words appear first."
  else if (bitwidth == 64) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words;
    if (isSigned) {
      words = llvm::bit_cast<DoubleWord>(value.getSExtValue());
    } else {
      words = llvm::bit_cast<DoubleWord>(value.getZExtValue());
    }
    encodeInstructionInto(typesGlobalValues, opcode,
                          {typeID, resultID, words.word1, words.word2});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss, /*isSigned=*/false);

    emitError(loc, "cannot serialize ")
        << bitwidth << "-bit integer literal: " << rss.str();
    return 0;
  }

  if (!isSpec) {
    constIDMap[intAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareConstantFp(Location loc, FloatAttr floatAttr,
                                       bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate nomral contants, but not specialization constants.
    if (auto id = findConstantID(floatAttr)) {
      return id;
    }
  }

  // Process the type for this float literal
  uint32_t typeID = 0;
  if (failed(processType(loc, floatAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  APFloat value = floatAttr.getValue();
  APInt intValue = value.bitcastToAPInt();

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  if (&value.getSemantics() == &APFloat::IEEEsingle()) {
    uint32_t word = llvm::bit_cast<uint32_t>(value.convertToFloat());
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } else if (&value.getSemantics() == &APFloat::IEEEdouble()) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words = llvm::bit_cast<DoubleWord>(value.convertToDouble());
    encodeInstructionInto(typesGlobalValues, opcode,
                          {typeID, resultID, words.word1, words.word2});
  } else if (&value.getSemantics() == &APFloat::IEEEhalf()) {
    uint32_t word =
        static_cast<uint32_t>(value.bitcastToAPInt().getZExtValue());
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss);

    emitError(loc, "cannot serialize ")
        << floatAttr.getType() << "-typed float literal: " << rss.str();
    return 0;
  }

  if (!isSpec) {
    constIDMap[floatAttr] = resultID;
  }
  return resultID;
}

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

LogicalResult Serializer::processAddressOfOp(spirv::AddressOfOp addressOfOp) {
  auto varName = addressOfOp.variable();
  auto variableID = findVariableID(varName);
  if (!variableID) {
    return addressOfOp.emitError("unknown result <id> for variable ")
           << varName;
  }
  valueIDMap[addressOfOp.pointer()] = variableID;
  return success();
}

LogicalResult
Serializer::processReferenceOfOp(spirv::ReferenceOfOp referenceOfOp) {
  auto constName = referenceOfOp.spec_const();
  auto constID = findSpecConstID(constName);
  if (!constID) {
    return referenceOfOp.emitError(
               "unknown result <id> for specialization constant ")
           << constName;
  }
  valueIDMap[referenceOfOp.reference()] = constID;
  return success();
}

LogicalResult Serializer::processOperation(Operation *op) {
  // First dispatch the methods that do not directly mirror an operation from
  // the SPIR-V spec
  if (auto constOp = dyn_cast<spirv::ConstantOp>(op)) {
    return processConstantOp(constOp);
  }
  if (auto specConstOp = dyn_cast<spirv::SpecConstantOp>(op)) {
    return processSpecConstantOp(specConstOp);
  }
  if (auto refOpOp = dyn_cast<spirv::ReferenceOfOp>(op)) {
    return processReferenceOfOp(refOpOp);
  }
  if (auto fnOp = dyn_cast<FuncOp>(op)) {
    return processFuncOp(fnOp);
  }
  if (isa<spirv::ModuleEndOp>(op)) {
    return success();
  }
  if (auto varOp = dyn_cast<spirv::GlobalVariableOp>(op)) {
    return processGlobalVariableOp(varOp);
  }
  if (auto addressOfOp = dyn_cast<spirv::AddressOfOp>(op)) {
    return processAddressOfOp(addressOfOp);
  }
  return dispatchToAutogenSerialization(op);
}

namespace {
template <>
LogicalResult
Serializer::processOp<spirv::EntryPointOp>(spirv::EntryPointOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the ExectionModel.
  operands.push_back(static_cast<uint32_t>(op.execution_model()));
  // Add the function <id>.
  auto funcID = findFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be defined before spv.EntryPoint is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the name of the function.
  encodeStringLiteralInto(operands, op.fn());

  // Add the interface values.
  if (auto interface = op.interface()) {
    for (auto var : interface.getValue()) {
      auto id = findVariableID(var.cast<SymbolRefAttr>().getValue());
      if (!id) {
        return op.emitError("referencing undefined global variable."
                            "spv.EntryPoint is at the end of spv.module. All "
                            "referenced variables should already be defined");
      }
      operands.push_back(id);
    }
  }
  return encodeInstructionInto(entryPoints, spirv::Opcode::OpEntryPoint,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::ExecutionModeOp>(spirv::ExecutionModeOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the function <id>.
  auto funcID = findFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be serialized before ExecutionModeOp is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the ExecutionMode.
  operands.push_back(static_cast<uint32_t>(op.execution_mode()));

  // Serialize values if any.
  auto values = op.values();
  if (values) {
    for (auto &intVal : values.getValue()) {
      operands.push_back(static_cast<uint32_t>(
          intVal.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  return encodeInstructionInto(executionModes, spirv::Opcode::OpExecutionMode,
                               operands);
}

// Pull in auto-generated Serializer::dispatchToAutogenSerialization() and
// various Serializer::processOp<...>() specializations.
#define GET_SERIALIZATION_FNS
#include "mlir/Dialect/SPIRV/SPIRVSerialization.inc"
} // namespace

LogicalResult spirv::serialize(spirv::ModuleOp module,
                               SmallVectorImpl<uint32_t> &binary) {
  Serializer serializer(module);

  if (failed(serializer.serialize()))
    return failure();

  serializer.collect(binary);
  return success();
}
