#ifndef OPENMP_OPS
#define OPENMP_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def OpenMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "openmp";
}

class OpenMP_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<OpenMP_Dialect, mnemonic, traits>;

def BarrierOp : OpenMP_Op<"barrier"> {
  let summary = "barrier construct";
  let description = [{
    The barrier construct specifies an explicit barrier at the point at which
    the construct appears
  }];

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

#endif //OPENMP_OPS
