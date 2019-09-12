
#ifdef TOY_COMBINE
#else
#define TOY_COMBINE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
include "include/toy/Ops.td"
#endif // OP_BASE

def TransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)), (replaceWithValue $arg)>;

#endif // TOY_COMBINE
