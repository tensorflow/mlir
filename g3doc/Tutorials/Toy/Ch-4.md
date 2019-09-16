# Chapter 4: High-level Language-Specific Analysis and Transformation

Creating a dialect that closely represents the semantics of an input language
enables analyses, transformations and optimizations in MLIR that require high level language 
information and are generally performed on the language AST. For example, `clang` has a fairly
[heavy mechanism](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html)
for performing template instantiation in C++.

We divide compiler transformations into two: local and global. In this chapter, we 
focus on how to leverage the Toy Dialect and its high-level semantics to perform 
local pattern-match transformations that would be difficult in LLVM. For this, we use 
MLIR's [Generic DAG Rewriter](https://github.com/tensorflow/mlir/blob/master/g3doc/GenericDAGRewriter.md).

There are two methods that can be used to implement pattern-match transformations:
1. Declarative, rule-based pattern-match and rewrite using ODS
2. Imperative, C++ pattern-match and rewrite

# Eliminate Redundant Transpose

Let's start with a simple pattern and try to eliminate a sequence of two
transpose that cancel out: `transpose(transpose(X)) -> X`. Here is the
corresponding Toy example:

```Toy(.toy)
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Which corresponds to the following IR:

```MLIR(.mlir)
func @transpose_transpose(%arg0: !toy<"array">)
  attributes  {toy.generic: true} {
  %0 = "toy.transpose"(%arg0) : (!toy<"array">) -> !toy<"array">
  %1 = "toy.transpose"(%0) : (!toy<"array">) -> !toy<"array">
  "toy.return"(%1) : (!toy<"array">) -> ()
}
```

This is a good example of a transformation that is trivial to match on the Toy
IR but that would be quite hard for LLVM to figure. For example today clang
can't optimize away the temporary array and the computation with the naive
transpose expressed with these loops:

```c++
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

For a simple C++ approach to rewrite involving matching a tree-like pattern in the IR and
replacing it with a different set of operations, we can plug into the MLIR
`Canonicalizer` pass by implementing a `RewritePattern`:

```c++
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::RewritePattern {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : RewritePattern(TransposeOp::getOperationName(), /* benefit = */ 1, context) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::PatternMatchResult matchAndRewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    // We can directly cast the current operation as this will only get invoked
    // on TransposeOp.
    TransposeOp transpose = op->cast<TransposeOp>();
    // look through the input to the current transpose
    mlir::Value *transposeInput = transpose.getOperand();
    // If the input is defined by another Transpose, bingo!
    if (!matchPattern(transposeInput, mlir::m_Op<TransposeOp>()))
      return matchFailure();

    auto transposeInputOp =
        transposeInput->getDefiningOp()->cast<TransposeOp>();
    // Use the rewriter to perform the replacement
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return matchSuccess();
  }
};
```

This match and rewrite can be expressed more simply using TableGen:

```TableGen(.td):
def TransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)), (replaceWithValue $arg)>;
```

The implementation of this rewriter is in `ToyCombine.cpp`. We also need to
update our main file, `toyc.cpp`, to add an optimization pipeline. In MLIR, the
optimizations are ran through a `PassManager` in a similar way to LLVM:

```c++
mlir::PassManager pm(ctx);
pm.addPass(createTransposeOptPass());
pm.run(&module);
```

Finally, we can try to run `toyc test/transpose_transpose.toy -emit=mlir -opt`
and observe our pattern in action:

```MLIR(.mlir)
func @transpose_transpose(%arg0: !toy<"array">)
  attributes  {toy.generic: true} {
  %0 = "toy.transpose"(%arg0) : (!toy<"array">) -> !toy<"array">
  "toy.return"(%arg0) : (!toy<"array">) -> ()
}
```

As expected we now directly return the function argument, bypassing any
transpose operation. However one of the transpose hasn't been eliminated. That
is not ideal! What happened is that our pattern replaced the last transform with
the function input and left behind the now dead transpose input. The
Canonicalizer knows to cleanup dead operations, however MLIR conservatively
assumes that operations may have side-effects. We can fix it by adding a new
trait, `HasNoSideEffect`, to our `TransposeOp`:

```c++
class TransposeOp : public mlir::Op<TransposeOp, mlir::OpTrait::OneOperand,
                                    mlir::OpTrait::OneResult,
                                    mlir::OpTrait::HasNoSideEffect> {
```

Let's retry now `toyc test/transpose_transpose.toy -emit=mlir -opt`:

```MLIR(.mlir)
func @transpose_transpose(%arg0: !toy<"array">)
  attributes  {toy.generic: true} {
  "toy.return"(%arg0) : (!toy<"array">) -> ()
}
```

Perfect! No `transpose` operation is left, the code is optimal.


# Optimize Reshapes

TableGen also provides a method for adding argument constraints when the transformation 
is conditional on some properties of the arguments and results. An example is a transformation 
that eliminates reshapes when they are redundant, i.e. when the input and output shapes are identical.

```TableGen(.td):
def TypesAreIdentical : Constraint<CPred<"$0->getType() == $1->getType()">>;
def RedundantReshapeOptPattern : Pat<(ReshapeOp:$res $arg), (replaceWithValue $arg), [(TypesAreIdentical $res, $arg)]>;
```

Finally, some optimizations may require additional transformations on instruction 
arguments. This is achieved using NativeCodeCall, which allows for more 
complex transformations by calling into a C++ helper function. An example of 
such an optimization is FoldConstantReshape, where we optimize Reshape of a constant value 
by reshaping the constant in place and eliminating the reshape operation.

```TableGen(.td):
def ReshapeConstant : NativeCodeCall<"reshapeConstant($_builder, $0)">;
def FoldConstantReshapeOptPattern : Pat<(ReshapeOp:$res (ConstantOp $arg)), (ReshapeConstant $res)>;
```
A helper C++ function "reshapeConstant" performs the actual in place transformation of the constant:

```c++
// Helper function to fold reshape(constant) in place
Value *reshapeConstant(Builder &builder, Value* arg) {
    ReshapeOp reshape = llvm::dyn_cast_or_null<ReshapeOp>(arg->getDefiningOp());
    mlir::OpBuilder builder2(reshape.getOperation());
    ConstantOp constantOp = llvm::dyn_cast_or_null<ConstantOp>(
        reshape.getOperand()->getDefiningOp());
    auto reshapeType = reshape.getType().cast<TensorType>();
    if (auto valueAttr =
            constantOp.getAttrOfType<mlir::DenseElementsAttr>("value")) {
      // FIXME Check matching of element count!
      //      auto oldType = constantOp.getType();
      auto newType = builder.getTensorType(
          reshapeType.getShape(), valueAttr.getType().getElementType());
      auto newAttr = valueAttr.reshape(newType);
      return builder2.create<ConstantOp>(reshape.getLoc(), newType, newAttr);
    } else if (auto valueAttr =
                   constantOp.getAttrOfType<mlir::FloatAttr>("value")) {
      // Broadcast
      auto dataSize = std::accumulate(reshapeType.getShape().begin(),
                                      reshapeType.getShape().end(), 1,
                                      std::multiplies<int>());
      std::vector<mlir::Attribute> data(dataSize, valueAttr);
      auto tensorTy = builder.getTensorType(reshapeType.getShape(),
                                             reshapeType.getElementType());
      auto newAttr = mlir::DenseElementsAttr::get(tensorTy, data);
      return builder2.create<ConstantOp>(reshape.getLoc(), tensorTy, newAttr);
    } else {
      llvm_unreachable("Unsupported Constant format");
    }
    return reshape;
}
```
Further details on the declarative rewrite method can be found at [Table-driven Declarative Rewrite Rule (DRR)](https://github.com/tensorflow/mlir/blob/master/g3doc/DeclarativeRewrites.md).

