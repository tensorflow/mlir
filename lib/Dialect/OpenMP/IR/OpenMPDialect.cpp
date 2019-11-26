#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace openmp {

OpenMPDialect::OpenMPDialect(MLIRContext *context) : Dialect("omp", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.cpp.inc"

static DialectRegistration<OpenMPDialect> openmpDialect;

} // namespace openmp
} // namespace mlir
