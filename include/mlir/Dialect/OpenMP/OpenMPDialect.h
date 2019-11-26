#ifndef MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_
#define MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionSupport.h"

namespace mlir {
namespace openmp {

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.h.inc"

class OpenMPDialect : public Dialect {
public:
  explicit OpenMPDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "omp"; }
};

} // namespace openmp
} // namespace mlir

#endif /* MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_ */
