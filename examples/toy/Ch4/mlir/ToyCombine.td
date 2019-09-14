
#ifdef TOY_COMBINE
#else
#define TOY_COMBINE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
include "include/toy/Ops.td"
#endif // OP_BASE

// Transpose(Transpose(x)) = x
def TransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)), (replaceWithValue $arg)>;

// Reshape(Reshape(x)) = x
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

// Reshape(Constant(x)) = x'
def ReshapeConstant : NativeCodeCall<"reshapeConstant($_builder, $0)">;
def FoldConstantReshapeOptPattern : Pat<(ReshapeOp:$res (ConstantOp $arg)), (ReshapeConstant $res)>;

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0->getType() == $1->getType()">>;
def RedundantReshapeOptPattern : Pat<(ReshapeOp:$res $arg), (replaceWithValue $arg), [(TypesAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE
