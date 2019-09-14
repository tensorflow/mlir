
#ifdef TOY_COMBINE
#else
#define TOY_COMBINE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
include "include/toy/Ops.td"
#endif // OP_BASE

def TransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)), (replaceWithValue $arg)>;

def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

def ReshapeConstant : NativeCodeCall<"reshapeConstant($_builder, $0)">;
def FoldConstantReshapeOptPattern : Pat<(ReshapeOp:$res (ConstantOp $arg)), (ReshapeConstant $res)>;

def TypesAreIdentical : Constraint<CPred<"$0->getType() == $1->getType()">>;
def RedundantReshapeOptPattern : Pat<(ReshapeOp:$res $arg), (replaceWithValue $arg), [(TypesAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE
