//===- TestCustomMemRefLLVMLowering.cpp - Pass to test strides
// computation--===//
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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
/// Test pass that lowers MemRef type to LLVM using a custom descriptor.
struct TestCustomMemRefLLVMLowering
    : public ModulePass<struct TestCustomMemRefLLVMLowering> {
  void runOnModule() override;
};

/// Custom MemRef descriptor that lowers MemRef types to LLVM plain pointers.
/// Alignment and dynamic shapes are currently not supported.
class CustomMemRefDescriptor : public MemRefDescriptor {
public:
  /// Construct a helper for the given descriptor value.
  explicit CustomMemRefDescriptor(Value *descriptor) : value(descriptor){};

  Value *getValue() override { return value; }

  /// Builds IR extracting the allocated pointer from the descriptor.
  Value *allocatedPtr(OpBuilder &builder, Location loc) override {
    return value;
  };
  /// Builds IR inserting the allocated pointer into the descriptor.
  void setAllocatedPtr(OpBuilder &builder, Location loc, Value *ptr) override {
    value = ptr;
  };

  /// Builds IR extracting the aligned pointer from the descriptor.
  Value *alignedPtr(OpBuilder &builder, Location loc) override {
    return allocatedPtr(builder, loc);
  };

  /// Builds IR inserting the aligned pointer into the descriptor.
  void setAlignedPtr(OpBuilder &builder, Location loc, Value *ptr) override{
      // Alignment is not supported by this memref descriptor.
      // 'alignedPtr' returns allocatedPtr instead.
  };

  /// Builds IR extracting the offset from the descriptor.
  Value *offset(OpBuilder &builder, Location loc) override {
    llvm_unreachable("'offset' is not implemented in CustomMemRefDescriptor");
  };

  /// Builds IR inserting the offset into the descriptor.
  void setOffset(OpBuilder &builder, Location loc, Value *offset) override{};

  void setConstantOffset(OpBuilder &builder, Location loc,
                         uint64_t offset) override{};

  /// Builds IR extracting the pos-th size from the descriptor.
  Value *size(OpBuilder &builder, Location loc, unsigned pos) override {
    llvm_unreachable("'size' is not implemented in CustomMemRefDescriptor");
  };

  /// Builds IR inserting the pos-th size into the descriptor
  void setSize(OpBuilder &builder, Location loc, unsigned pos,
               Value *size) override{};
  void setConstantSize(OpBuilder &builder, Location loc, unsigned pos,
                       uint64_t size) override{};

  /// Builds IR extracting the pos-th size from the descriptor.
  Value *stride(OpBuilder &builder, Location loc, unsigned pos) override {
    llvm_unreachable("'stride' is not implemented in CustomMemRefDescriptor");
  };

  /// Builds IR inserting the pos-th stride into the descriptor
  void setStride(OpBuilder &builder, Location loc, unsigned pos,
                 Value *stride) override{};
  void setConstantStride(OpBuilder &builder, Location loc, unsigned pos,
                         uint64_t stride) override{};

  /// Returns the (LLVM) type this descriptor points to.
  LLVM::LLVMType getElementType() override {
    return value->getType().cast<LLVM::LLVMType>();
  }

private:
  Value *value;
};

/// Provides Std-to-LLVM type conversion by using CustomMemRefDescriptor to
/// lower MemRef types. Falls back to base LLVMTypeConverter for the remaining
/// types.
class CustomLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  using LLVMTypeConverter::LLVMTypeConverter;

  Type convertType(Type type) override {
    if (auto memrefTy = type.dyn_cast<MemRefType>()) {
      return convertMemRefType(memrefTy);
    }

    // Fall back to base class converter.
    return LLVMTypeConverter::convertType(type);
  }

  /// Creates a CustomMemRefDescriptor object for 'value'.
  std::unique_ptr<MemRefDescriptor>
  createMemRefDescriptor(Value *value) override {
    return std::make_unique<CustomMemRefDescriptor>(value);
  }

  /// Creates a CustomMemRefDescriptor object for an uninitialized descriptor
  /// (nullptr value). No new IR is needed for such initialization.
  std::unique_ptr<MemRefDescriptor>
  buildMemRefDescriptor(OpBuilder &builder, Location loc,
                        Type descriptorType) override {
    return createMemRefDescriptor(nullptr);
  }

  /// Builds IR creating a MemRef descriptor that represents `type` and
  /// populates it with static shape and stride information extracted from the
  /// type.
  std::unique_ptr<MemRefDescriptor>
  buildStaticMemRefDescriptor(OpBuilder &builder, Location loc, MemRefType type,
                              Value *memory) override {
    assert(type.hasStaticShape() && "unexpected dynamic shape");
    assert(type.getAffineMaps().empty() && "unexpected layout map");

    auto convertedType = convertType(type);
    assert(convertedType && "unexpected failure in memref type conversion");

    auto descr = buildMemRefDescriptor(builder, loc, convertedType);
    descr->setAllocatedPtr(builder, loc, memory);
    return descr;
  }

private:
  /// Converts MemRef type to plain LLVM pointer to element type.
  Type convertMemRefType(MemRefType type) {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    bool strideSuccess = succeeded(getStridesAndOffset(type, strides, offset));
    assert(strideSuccess &&
           "Non-strided layout maps must have been normalized away");
    (void)strideSuccess;

    LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
    if (!elementType)
      return {};
    auto ptrTy = elementType.getPointerTo(type.getMemorySpace());
    return ptrTy;
  }
};

} // end anonymous namespace

void TestCustomMemRefLLVMLowering::runOnModule() {
  // Populate Std-to-LLVM conversion patterns using the custom type converter.
  CustomLLVMTypeConverter typeConverter(&getContext());
  OwningRewritePatternList patterns;
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  if (failed(applyPartialConversion(getModule(), target, patterns,
                                    &typeConverter)))
    signalPassFailure();
}

static PassRegistration<TestCustomMemRefLLVMLowering>
    pass("test-custom-memref-llvm-lowering",
         "Test custom LLVM lowering of memrefs");

