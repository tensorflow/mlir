//===- ShapeInference.h - Class definition for Shape Inference Interface ----------------------===//
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
// This file implements the interface for Shape Inference.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_SHAPE_INFERENCE_H_
#define MLIR_TUTORIAL_SHAPE_INFERENCE_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Dialect.h"
#include "include/ShapeInferencePass.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class DialectShapeInferenceInterface :
    public DialectInterface::Base<DialectShapeInferenceInterface> {
public:
  DialectShapeInferenceInterface(Dialect *dialect) : Base(dialect) {}
  virtual bool returnsGenericArray(Operation *op) const {
    return false;
  }
  virtual bool requiresShapeInference() const {
    return false;
  }
  virtual void inferShape(Operation *op) const {}
};

/// This interface provides the hooks into the shape inference interface.
class ShapeInferenceInterface
    : public DialectInterfaceCollection<DialectShapeInferenceInterface> {
public:
  using Base::Base;
  virtual ~ShapeInferenceInterface();

  /// These hooks mirror the hooks for the DialectShapeInferenceInterface, with default
  /// implementations that call the hook on the handler for the dialect 'op' is
  /// registered to.

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  virtual bool returnsGenericArray(Operation *op);
  virtual bool requiresShapeInference(Operation *op);

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  virtual void inferShape(Operation *op);
};
} // end mlir

#endif // MLIR_TUTORIAL_SHAPE_INFERENCE_H_
