# GPU Dialect

Note: this dialect is more likely to change than others in the near future; use
with caution.

This dialect provides middle-level abstractions for launching GPU kernels
following a programming model similar to that of CUDA or OpenCL. It provides
abstractions for kernel invocations (and may eventually provide those for device
management) that are not present at the lower level (e.g., as LLVM IR intrinsics
for GPUs). Its goal is to abstract away device- and driver-specific
manipulations to launch a GPU kernel and provide a simple path towards GPU
execution from MLIR. It may be targeted, for example, by DSLs using MLIR. The
dialect uses `gpu` as its canonical prefix.

## Operations

### `gpu.block_dim`

Returns the number of threads in the thread block (aka the block size) along the
x, y, or z `dimension`.

Example:

```mlir {.mlir}
  %bDimX = "gpu.block_dim"() {dimension: "x"} : () -> (index)
```

### `gpu.block_id`

Returns the block id, i.e. the index of the current block within the grid along
the x, y, or z `dimension`.

Example:

```mlir {.mlir}
  %bIdY = "gpu.block_id"() {dimension: "y"} : () -> (index)
```

### `gpu.grid_dim`

Returns the number of thread blocks in the grid along the x, y, or z
`dimension`.

Example:

```mlir {.mlir}
  %gDimZ = "gpu.grid_dim"() {dimension: "z"} : () -> (index)
```

### `gpu.launch`

Launch a kernel on the specified grid of thread blocks. The body of the kernel
is defined by the single region that this operation contains. The operation
takes at least six operands, with first three operands being grid sizes along
x,y,z dimensions, the following three arguments being block sizes along x,y,z
dimension, and the remaining operands are arguments of the kernel. When a
lower-dimensional kernel is required, unused sizes must be explicitly set to
`1`.

The body region has at least _twelve_ arguments, grouped as follows:

-   three arguments that contain block identifiers along x,y,z dimensions;
-   three arguments that contain thread identifiers along x,y,z dimensions;
-   operands of the `gpu.launch` operation as is, including six leading operands
    for grid and block sizes.

Operations inside the body region, and any operations in the nested regions, are
_not_ allowed to use values defined outside the _body_ region, as if this region
was a function. If necessary, values must be passed as kernel arguments into the
body region. Nested regions inside the kernel body are allowed to use values
defined in their ancestor regions as long as they don't cross the kernel body
region boundary.

Syntax:

``` {.ebnf}
operation ::= `gpu.launch` `block` `(` ssa-id-list `)` `in` ssa-reassignment
                         `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                           (`args` ssa-reassignment `:` type-list)?
                           region attr-dict?
ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
```

Example:

```mlir {.mlir}
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
           args(%arg0 = %6, %arg1 = 7) : f32, memref<?xf32, 1> {
  // Block and thread identifiers, as well as block/grid sizes are
  // immediately usable inside body region.
  "some_op"(%bx, %tx) : (index, index) -> ()
  %42 = load %arg1[%bx] : memref<?xf32, 1>
}

// Generic syntax explains how the pretty syntax maps to the IR structure.
"gpu.launch"(%cst, %cst, %c1,  // Grid sizes.
                    %cst, %c1, %c1,   // Block sizes.
                    %arg0, %arg1)     // Actual arguments.
    {/*attributes*/}
    // All sizes and identifiers have "index" size.
    : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> () {
// The operation passes block and thread identifiers, followed by grid and block
// sizes, followed by actual arguments to the entry block of the region.
^bb0(%bx : index, %by : index, %bz : index,
     %tx : index, %ty : index, %tz : index,
     %num_bx : index, %num_by : index, %num_bz : index,
     %num_tx : index, %num_ty : index, %num_tz : index,
     %arg0 : f32, %arg1 : memref<?xf32, 1>):
  "some_op"(%bx, %tx) : (index, index) -> ()
  %3 = "std.load"(%arg1, %bx) : (memref<?xf32, 1>, index) -> f32
}
```

Rationale: using operation/block arguments gives analyses a clear way of
understanding that a value has additional semantics (e.g., we will need to know
what value corresponds to threadIdx.x for coalescing). We can recover these
properties by analyzing the operations producing values, but it is easier just
to have that information by construction.

### `gpu.launch_func`

Launch a kernel given as a function on the specified grid of thread blocks.
`gpu.launch` operations are lowered to `gpu.launch_func` operations by outlining
the kernel body into a function, which is closer to the NVVM model. The
`gpu.launch_func` operation has a function attribute named `kernel` to specify
the kernel function to launch. The kernel function itself has a `nvvm.kernel`
attribute.

The operation takes at least six operands, with the first three operands being
grid sizes along x,y,z dimensions and the following three being block sizes
along x,y,z dimensions. When a lower-dimensional kernel is required, unused
sizes must be explicitly set to `1`. The remaining operands are passed as
arguments to the kernel function.

A custom syntax for this operation is currently not available.

Example:

```mlir {.mlir}
func @kernel_1(%arg0 : f32, %arg1 : !llvm<"float*">)
    attributes { nvvm.kernel: true } {

  // Operations that produce block/thread IDs and dimensions are injected when
  // outlining the `gpu.launch` body to a function called by `gpu.launch_func`.
  %tIdX = "gpu.thread_id"() {dimension: "x"} : () -> (index)
  %tIdY = "gpu.thread_id"() {dimension: "y"} : () -> (index)
  %tIdZ = "gpu.thread_id"() {dimension: "z"} : () -> (index)

  %bDimX = "gpu.block_dim"() {dimension: "x"} : () -> (index)
  %bDimY = "gpu.block_dim"() {dimension: "y"} : () -> (index)
  %bDimZ = "gpu.block_dim"() {dimension: "z"} : () -> (index)

  %bIdX = "gpu.block_id"() {dimension: "x"} : () -> (index)
  %bIdY = "gpu.block_id"() {dimension: "y"} : () -> (index)
  %bIdZ = "gpu.block_id"() {dimension: "z"} : () -> (index)

  %gDimX = "gpu.grid_dim"() {dimension: "x"} : () -> (index)
  %gDimY = "gpu.grid_dim"() {dimension: "y"} : () -> (index)
  %gDimZ = "gpu.grid_dim"() {dimension: "z"} : () -> (index)

  "some_op"(%bx, %tx) : (index, index) -> ()
  %42 = load %arg1[%bx] : memref<?xf32, 1>
}

"gpu.launch_func"(%cst, %cst, %cst,  // Grid sizes.
                  %cst, %cst, %cst,  // Block sizes.
                  %arg0, %arg1)      // Arguments passed to the kernel function.
      {kernel: @kernel_1 : (f32, !llvm<"float*">) -> ()}  // Kernel function.
      : (index, index, index, index, index, index, f32, !llvm<"float*">) -> ()
```

### `gpu.thread_id`

Returns the thread id, i.e. the index of the current thread within the block
along the x, y, or z `dimension`.

Example:

```mlir {.mlir}
  %tIdX = "gpu.thread_id"() {dimension: "x"} : () -> (index)
```
