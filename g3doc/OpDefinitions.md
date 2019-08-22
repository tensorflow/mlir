# Table-driven Operation Definition Specification (ODS)

In addition to specializing the `mlir::Op` C++ template, MLIR also supports
defining operations in a table-driven manner. This is achieved via
[TableGen][TableGen], which is both a generic language and its tooling to
maintain records of domain-specific information. Facts regarding an operation
are specified concisely into a TableGen record, which will be expanded into an
equivalent `mlir::Op` C++ template specialization at compiler build time.

This manual explains in detail all the available mechansims for defining
operations in such a table-driven manner. It aims to be a specification instead
of a tutorial. Please refer to [Quickstart tutorial to adding MLIR graph
rewrite](QuickstartRewrites.md) for the latter.

In addition to detailing each mechanism, this manual also tries to capture
best practices. They are rendered as quoted bullet points.

## Motivation

MLIR allows pluggable dialects, and dialects contain, among others, a list of
operations. This open and extensible ecosystem leads to the "stringly" type IR
problem, e.g., repetitive string comparisons during optimization and analysis
passes, unintuitive accessor methods (e.g., generic/error prone `getOperand(3)`
vs self-documenting `getStride()`) with more generic return types, verbose and
generic constructors without default arguments, verbose textual IR dump, and
so on. Furthermore, operation verification is:

1. best case: a central string-to-verification-function map,
1. middle case: duplication of verification across the code base, or
1. worst case: no verification functions.

The fix is to support defining ops in a table-driven manner. Then for each
dialect, we can have a central place that contains everything you need to know
about each op, including its constraints, custom assembly form, etc. This
description is also used to generate helper functions and classes to allow
building, verification, parsing, printing, analysis, and many more.

## Benefits

Compared to the C++ template, this table-driven approach has several benefits
including but not limited to:

* **Single source of truth**: We strive to encode all facts regarding an
  operation into the record, so that readers don't need to jump among code
  snippets to fully understand an operation.
* **Removing boilerplate**: We can automatically generate
  operand/attribute/result getter methods, operation build methods, operation
  verify methods, and many more utilities from the record. This greatly reduces
  the boilerplate needed for defining a new op.
* **Facilitating auto-generation**: The usage of these operation information
  records are by no means limited to op definition itself. We can use them to
  drive the auto-generation of many other components, like computation graph
  serialization.

## TableGen Syntax

We use TableGen as the language for specifying operation information. TableGen
itself just provides syntax for writing records; the syntax and constructs
allowed in a TableGen file (typically with filename suffix `.td`) can be found
[here][TableGenIntro]. The formal language specification can be found
[here][TableGenRef]. _Roughly_ speaking,

* TableGen `class` is similar to C++ class; it can be templated and subclassed.
* TableGen `def` is similar to C++ object; it can be declared by specializing
  a TableGen `class` (e.g., `def MyDef : MyClass<...>;`) or completely
  independently (e.g., `def MyDef;`). It cannot be further templated or
  subclassed.
* TableGen `dag` is a dedicated type for directed graph of elements. A `dag`
  has one operator and zero or more arguments. Its syntax is `(operator arg0,
  arg1, argN)`. The operator can be any TableGen `def`; an argument can be
  anything, including `dag` itself. We can have names attached to both the
  operator and the arguments like `(MyOp:$op_name MyArg:$arg_name)`.

Please see the [language introduction][TableGenIntro] to learn about all the
types and expressions supported by TableGen.

## Operation Definition

MLIR defines several common constructs to help operation definition and provide
their semantics via a special [TableGen backend][TableGenBackend]:
[`OpDefinitionsGen`][OpDefinitionsGen]. These constructs are defined in
[`OpBase.td`][OpBase]. The main ones are

* The `Op` class: It is the main construct for defining operations. All facts
  regarding the operation is specified when specializing this class, with the
  help of the following constructs.
* The `Dialect` class: Operations belonging to one logical group are placed in
  the same dialect. The `Dialect` class contains dialect-level information.
* The `OpTrait` class hierarchy: They are used to specify special properties and
  constraints of the operation, including whether the operation has side effect
  or whether its output has the same shape as the input.
* The `ins`/`outs` marker: These are two special makers builtin to the
  `OpDefinitionsGen` backend. They lead the definitions of operands/attributes
  and results respectively.
* The `TypeConstraint` class hierarchy: They are used to specify the constraints
  over operands or results. A notable subclass hierarchy is `Type`, which
  stands for constraints for common C++ types.
* The `AttrConstraint` class hierarchy: They are used to specify the constraints
  over attributes. A notable subclass hierarchy is `Attr`, which stands for
  constraints for attributes whose values are of common types.

An operation is defined by specializing the `Op` class with concrete contents
for all the fields it requires. For example, `tf.AvgPool` is defined as

```tablegen
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoSideEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
```

In the following we describe all the fields needed. Please see the definition
of the `Op` class for the complete list of fields supported.

### Operation name

The operation name is a unique identifier of the operation within MLIR, e.g.,
`tf.Add` for addition operation in the TensorFlow dialect. This is the
equivalent of the mnemonic in assembly language. It is used for parsing and
printing in the textual format. It is also used for pattern matching in graph
rewrites.

The full operation name is composed of the dialect name and the op name, with
the former provided via the dialect and the latter provided as the second
template parameter to the `Op` class.

### Operation documentation

This includes both an one-line `summary` and a longer human-readable
`description`. They will be used to drive automatic generation of dialect
documentation. They need to be provided in the operation's definition body:

```tablegen
let summary = "...";

let description = [{
...
}];
```

`description` should be written in Markdown syntax.

Placing the documentation at the beginning is recommended since
it helps in understanding the operation.

> * Place documentation at the beginning of the operation definition
> * The summary should be short and concise. It should be a one-liner without
>   trailing punctuation. Put expanded explanation in description.

### Operation arguments

There are two kinds of arguments: operands and attributes. Operands are runtime
values produced by other ops; while attributes are compile-time known constant
values, including two categories:

1. Natural attributes: these attributes affect the behavior of the operations
   (e.g., padding for convolution);
1. Derived attributes: these attributes are not needed to define the operation
   but are instead derived from information of the operation. E.g., the output
   shape of type. This is mostly used for convenience interface generation or
   interaction with other frameworks/translation.

Both operands and attributes are specified inside the `dag`-typed `arguments`,
led by `ins`:

```tablegen
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...

  <attr-constraint>:$<attr-name>,
  ...
);
```

Here `<type-constraint>` is a TableGen `def` from the `TypeConstraint` class
hierarchy. Similarly, `<attr-constraint>` is a TableGen `def` from the
`AttrConstraint` class hierarchy. See [Constraints](#constraints) for more
information.

There is no requirements on the relative order of operands and attributes; they
can mix freely. But it is recommended to put all operands ahead of attributes,
and use an empty line to separate them to make it more visually distinguishable
if possible. The relative order of operands themselves matters.

All the arguments should be named to 1) provide documentation, 2) drive
auto-generation of getter methods, 3) provide a handle to reference for other
places like constraints.

> * Place attributes after operands if possible
> * Give operands and attribute proper names

#### Variadic operands

To declare a variadic operand, wrap the `TypeConstraint` for the operand with
`Variadic<...>`.

Normally operations have no variadic operands or just one variadic operand.
For the latter case, it is easily deduce which dynamic operands are for the
static variadic operand definition. But if an operation has more than one
variadic operands, it would be impossible to attribute dynamic operands to the
corresponding static variadic operand definitions without further information
from the operation. Therefore, the `SameVariadicOperandSize` trait is needed
to indicate that all variadic operands have the same number of dynamic values.

#### Optional attributes

To declare an optional attribute, wrap the `AttrConstraint` for the attribute
with `OptionalAttr<...>`.

#### Attributes with default values

To declare an attribute with a default value, wrap the `AttrConstraint` for the
attribute with `DefaultValuedAttr<..., "...">`.

The second parameter to `DefaultValuedAttr` should be a string containing the
C++ default value. For example, a float default value should be specified as
like `"0.5f"`, and an integer array default value should be specified as like
`"{1, 2, 3}"`.

#### Confining attributes

`Confined` is provided as a general mechanism to help modelling further
constraints on attributes beyond the ones brought by value types. You can use
`Confined` to compose complex constraints out of more primitive ones. For
example, an 32-bit integer attribute whose minimal value must be 10 can be
expressed as `Confined<I32Attr, [IntMinValue<10>]>`.

Right now, the following primitive constraints are supported:

* `IntMinValue<N>`: Specifying an integer attribute to be greater than or equal
  to `N`
* `ArrayMinCount<N>`: Specifying an array attribute to have at least `N`
  elements
* `IntArrayNthElemEq<I, N>`: Specifying an integer array attribute's `I`-th
  element to be equal to `N`
* `IntArrayNthElemMinValue<I, N>`: Specifying an integer array attribute's
  `I`-th element to be greater than or equal to `N`

TODO: Design and implement more primitive constraints

### Operation results

Similar to operands, results are specified inside the `dag`-typed `results`, led
by `outs`:

```tablgen
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

#### Variadic results

Similar to variadic operands, `Variadic<...>` can also be used for results.
And similarly, `SameVariadicResultSize` for multiple variadic results in the
same operation.

### Operation traits and constraints

Traits are operation properties that affect syntax or semantics. MLIR C++
models various traits in the `mlir::OpTrait` namespace.

Both operation traits, [interfaces](#operation-interfaces), and constraints
involving multiple operands/attributes/results are provided as the second
template parameter to the `Op` class. They should be deriving from the `OpTrait`
class. See [Constraints](#constraints) for more information.

### Operation interfaces

[Operation interfaces](Interfaces.md#operation-interfaces) are a mechanism by
which to opaquely call methods and access information on an *Op instance,
without knowing the exact operation type. Operation interfaces defined in C++
can be accessed in the ODS framework via the `OpInterfaceTrait` class. Aside
from using pre-existing interfaces in the C++ API, the ODS framework also
provides a simplified mechanism for defining such interfaces; that removes much
of the boilerplate necessary.

Providing a definition of the `OpInterface` class will auto-generate the C++
classes for the interface. An `OpInterface` includes a name, for the C++ class,
along with a list of interface methods.

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let methods = [...];
}
```

There are two types of methods that can be used with an interface,
`InterfaceMethod` and `StaticInterfaceMethod`. They are both comprised of the
same core components, with the distinction that `StaticInterfaceMethod` models a
static method on the derived operation.

An `InterfaceMethod` is comprised of the following components:

*   ReturnType
    -   A string corresponding to the C++ return type of the method.
*   MethodName
    -   A string corresponding to the desired name of the method.
*   Arguments (Optional)
    -   A dag of strings that correspond to a C++ type and variable name
        respectively.
*   MethodBody (Optional)
    -   An optional explicit implementation of the interface method.
    -   `ConcreteOp` is an implicitly defined typename that can be used to refer
        to the type of the derived operation currently being operated on.
    -   In non-static methods, a variable 'ConcreteOp op' is defined and may be
        used to refer to an instance of the derived operation.

Examples:

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let methods = [
    // A simple non-static method with no inputs.
    InterfaceMethod<"unsigned", "foo">,

    // A new non-static method accepting an input argument.
    InterfaceMethod<"Value *", "bar", (ins "unsigned":$i)>,

    // Query a static property of the derived operation.
    StaticInterfaceMethod<"unsigned", "fooStatic">,

    // Provide the definition of a static interface method.
    // Note: `ConcreteOp` corresponds to the derived operation typename.
    StaticInterfaceMethod<"Operation *", "create",
      (ins "OpBuilder &":$builder, "Location":$loc), [{
        return builder.create<ConcreteOp>(loc);
    }]>,

    // Provide a definition of the non-static method.
    // Note: `op` corresponds to the derived operation variable.
    InterfaceMethod<"unsigned", "getNumInputsAndOutputs", (ins), [{
      return op.getNumInputs() + op.getNumOutputs();
    }]>,
  ];
}
```

### Custom builder methods

For each operation, there are two builder automatically generated based on the
arguments and returns types:

```c++
static void build(Builder *, OperationState *tblgen_state,
                  Type <result0-name>, Type <result1-name>, ...,
                  Value <arg0-name>, Value <arg1-name>, ...,
                  Attribute <attr0-name>, Attribute <attr1-name>, ...);

static void build(Builder *, OperationState *tblgen_state,
                  ArrayRef<Type> resultTypes,
                  ArrayRef<Value> operands,
                  ArrayRef<NamedAttribute> attributes);
```

The above cases makes sure basic uniformity so that we can create ops using the
same form regardless of the exact op. This is particularly useful for
implementing declarative pattern rewrites.

However, if the above cases cannot satisfy all needs, you can define additional
convenience build methods with `OpBuilder`.

`OpBuilder` is a class that takes the parameter list and the optional `build()`
method body. They are separated because we need to generate op declaration and
definition into separate files. The parameter list should _include_
`Builder *builder, OperationState *state`. If the `body` is not provided, _only_
the builder declaration will be generated; this provides a way to define
complicated builders entirely in C++ files.

For example, for the following op:

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let results = (outs);
}
```

If we want to define a builder with a default value for the only attribute, we
can add into `MyOp`:

```tablegen
def MyOp : ... {
  ...

  let builders = [
    OpBuilder<"Builder *builder, OperationState *state, float val = 0.5f", [{
      state->addAttribute("attr", builder->getF32FloatAttr(val));
    ]}>
  ]
}
```

The generated builder will look like:

```c++
static void build(Builder *builder, OperationState *state, float val = 0.5f) {
  state->addAttribute("attr", builder->getF32FloatAttr(val));
}
```

### Custom parser and printer methods

Functions to parse and print the operation's custom assembly form.

### Custom verifier code

Verification code will be automatically generated for
[constraints](#constraints) specified on various entities of the op. To
perform _additional_ verification, you can use

```tablegen
let verifier = [{
  ...
}];
```

Code placed in `verifier` will be called after the auto-generated verification
code.

### `hasCanonicalizer`

This boolean field indicate whether canonicalization patterns have been defined
for this operation. If it is `1`, then `::getCanonicalizationPatterns()` should
be defined.

### `hasFolder`

This boolean field indicate whether general folding rules have been defined
for this operation. If it is `1`, then `::fold()` should be defined.

### Extra declarations

One of the goals of table-driven op definition is to auto-generate as much logic
and methods needed for each op as possible. With that said,there will always be
long-tail cases that won't be covered. For such cases, you can use
`extraClassDeclaration`. Code in `extraClassDeclaration` will be copied
literally to the generated C++ op class.

Note that `extraClassDeclaration` is a mechanism intended for long-tail cases
by power users; for not-yet-implemented widely-applicable cases, improving the
infrastructure is preferable.

### Generated C++ code

[OpDefinitionsGen][OpDefinitionsGen] processes the op definition spec file and
generates two files containing the corresponding C++ code: one for declarations,
the other for definitions. The former is generated via the `-gen-op-decls`
command-line option, while the latter is via the `-gen-op-defs` option.

The definition file contains all the op method definitions, which can be
included and enabled by defining `GET_OP_CLASSES`. For each operation,
OpDefinitionsGen generates an operation class and an
[operand adaptor](#operand-adaptors) class. Besides, it also contains a
comma-separated list of all defined ops, which can be included and enabled by
defining `GET_OP_LIST`.

#### Class name and namespaces

For each operation, its generated C++ class name is the symbol `def`ed with
TableGen with dialect prefix removed. The first `_` serves as the delimiter.
For example, for `def TF_AddOp`, the C++ class name would be `AddOp`.
We remove the `TF` prefix because it is for scoping ops; other dialects
may as well define their own `AddOp`s.

The namespaces of the generated C++ class will come from the dialect's
`cppNamespace` field. For example, if a dialect's `cppNamespace` is `A::B`,
then an op of that dialect will be placed in
`namespace A { namespace B { ... } }`. If a dialect does not specify a
`cppNamespace`, we then use the dialect's name as the namespace.

This means the qualified name of the generated C++ class does not necessarily
match exactly with the operation name as explained in
[Operation name](#operation-name). This is to allow flexible naming to satisfy
coding style requirements.

#### Operand adaptors

For each operation, we automatically generate an _operand adaptor_. This class
solves the problem of accessing operands provided as a list of `Value`s without
using "magic" constants. The operand adaptor takes a reference to an array of
`Value *` and provides methods with the same names as those in the operation
class to access them. For example, for a binary arithmethic operation, it may
provide `.lhs()` to access the first operand and `.rhs()` to access the second
operand.

The operand adaptor class lives in the same namespace as the operation class,
and has the name of the operation followed by `OperandAdaptor`. A template
declaration `OperandAdaptor<>` is provided to look up the operand adaptor for
the given operation.

Operand adaptors can be used in function templates that also process operations:

```c++
template <typename BinaryOpTy>
std::pair<Value *, Value *> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());;
}

void process(AddOp op, ArrayRef<Value *> newOperands) {
  zip(op);
  zip(OperandAdaptor<AddOp>(newOperands));
  /*...*/
}
```

## Constraints

Constraint is a core concept in table-driven operation definition: operation
verification and graph operation matching are all based on satisfying
constraints. So both the operation definition and rewrite rules specification
significantly involve writing constraints. We have the `Constraint` class in
[`OpBase.td`][OpBase] has the common base class for all constraints.

An operation's constraint can cover different range; it may

* Only concern a single attribute (e.g. being an 32-bit integer greater than 5),
* Multiple operands and results (e.g., the 1st result's shape must be the same
  as the 1st operand), or
* Intrinsic to the operation itself (e.g., having no side effect).

We call them as single-entity constraint, multi-entity constraint, and traits,
respectively.

### Single-entity constraint

Constraints scoped to a single operand, attribute, or result are specified at
the entity's declaration place as described in
[Operation arguments](#operation-arguments) and
[Operation results](#operation-results).

To help modelling constraints of common types, a set of `TypeConstraint`s are
created; they are the `Type` subclass hierarchy. It includes `F32` for the
constraints of being a float, `TensorOf<[F32]>` for the constraints of being
a float tensor, and so on.

Similarly, a set of `AttrConstraint`s are created for helping modelling
constraints of common attribute kinds. They are the `Attr` subclass hierarchy.
It includes `F32Attr` for the constraints of being an float attribute,
`F32ArrayAttr` for the constraints of being a float array attribute, and so on.

### Multi-entity constraint

Constraints involving more than one operand/attribute/result are quite common
on operations, like the element type and shape relation between operands and
results. These constraints should be specified as the `Op` class template
parameter as described in
[Operation traits and constraints](#operation-traits-and-constraints).

Multi-entity constraints are modeled as `PredOpTrait` (a subclass of `OpTrait`)
in [`OpBase.td`][OpBase].A bunch of constraint primitives are provided to help
specification. See [`OpBase.td`][OpBase] for the complete list.

### Trait

Traits are intrinsic properties of the operation like having side effect or not,
commutative or not, whether is a terminator, etc. These constraints should be
specified as the `Op` class template parameter as described in
[Operation traits and constraints](#operation-traits-and-constraints).

Traits are modeled as `NativeOpTrait` (a subclass of `OpTrait`) in
[`OpBase.td`][OpBase]. They are backed and will be translated into the
corresponding C++ `mlir::OpTrait` classes.

### How to specify new constraint

To write a constraint, you need to provide its predicates and give it a
descriptive name. Predicates, modeled with the `Pred` class, are the workhorse
for composing constraints. The predicate for a constraint is typically built up
in a nested manner, using the two categories of predicates:

1.  `CPred`: the primitive leaf predicate.
2.  Compound predicate: a predicate composed from child predicates using
    predicate combiners (conjunction: `And`, disjunction: `Or`, negation: `Neg`,
    substitution: `SubstLeaves`, concatenation: `Concat`).

`CPred` is the basis for composing more complex predicates. It is the "atom"
predicate from the perspective of TableGen and the "interface" between
TableGen and C++. What is inside is already C++ code, which will be treated
as opaque strings with special placeholders to be substituted.

You can put any C++ code that returns a boolean value inside a `CPred`,
including evaluating expressions, calling functions, calling class methods,
and so on.

To help interaction with the C++ environment, there are a few special
placeholders provided to refer to entities in the context where this predicate
is used. They serve as "hooks" to the enclosing environment.  This includes
`$_builder`, `$_op`, and `$_self`:

* `$_builder` will be replaced by a `mlir::Builder` instance so that you can
  access common build methods.
* `$_op` will be replaced by the current operation so that you can access
  information of the current operation.
* `$_self` will be replaced with the entity this predicate is attached to.
  E.g., `BoolAttr` is an attribute constraint that wraps a
  `CPred<"$_self.isa<BoolAttr>()">`. Then for `F32:$attr`,`$_self` will be
  replaced by `$attr`. For type constraints, it's a little bit special since
  we want the constraints on each type definition reads naturally and we want
  to attach type constraints directly to an operand/result, `$_self` will be
  replaced by the operand/result's type. E.g., for `F32` in `F32:$operand`, its
  `$_self` will be expanded as `getOperand(...)->getType()`.

TODO(b/130663252): Reconsider the leading symbol for special placeholders.
Eventually we want to allow referencing operand/result $-names; such $-names
can start with underscore.

For example, to write an attribute `attr` is an `IntegerAttr`, in C++ you can
just call `attr.isa<IntegerAttr>()`. The code can be wrapped in a `CPred` as
`$_self.isa<IntegerAttr>()`, with `$_self` as the special placeholder to be
replaced by the current attribute `attr` at expansion time.

For more complicated predicates, you can wrap it in a single `CPred`, or you
can use predicate combiners to combine them. For example, to write the
constraint that an attribute `attr` is an 32-bit or 64-bit integer, you can
write it as

```tablegen
And<[
  CPred<"$_self.isa<IntegerAttr>()">,
  Or<[
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(32)">,
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(64)">
  ]>
]>
```

(Note that the above is just to show with a familiar example how you can use
`CPred` and predicate combiners to write complicated predicates. For integer
attributes specifically, [`OpBase.td`][OpBase] already defines `I32Attr` and
`I64Attr`. So you can actually reuse them to write it as `Or<[I32Attr.predicate,
I64Attr.predicate]>`.)

TODO: Build up a library of reusable primitive constraints

If the predicate is very complex to write with `CPred` together with predicate
combiners, you can also write it as a normal C++ function and use the `CPred`
as a way to "invoke" the function. For example, to verify an attribute `attr`
has some property, you can write a C++ function like

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

and then define the op as:

```tablegen
def HasSomeProperty : AttrConstraint<CPred<"HasSomeProperty($_self)">,
                                     "has some property">;

def MyOp : Op<...> {
  let arguments = (ins
    ...
    HasSomeProperty:$attr
  );
}
```

As to whether we should define the predicate using a single `CPred` wrapping
the whole expression, multiple `CPred`s with predicate combiners, or a single
`CPred` "invoking" a function, there are no clear-cut criteria. Defining using
`CPred` and predicate combiners is preferrable since it exposes more information
(instead hiding all the logic behind a C++ function) into the op definition spec
so that it can pontentially drive more auto-generation cases. But it will
require a nice library of common predicates as the building blocks to avoid the
duplication, which is being worked on right now.

## Attribute Definition

### Enum attributes

Enum attributes can be defined using `EnumAttr`, which requires all its cases to
be defined with `EnumAttrCase`. To facilitate the interaction between
`EnumAttr`s and their C++ consumers, the [`EnumsGen`][EnumsGen] TableGen backend
can generate a few common utilities, including an enum class,
`llvm::DenseMapInfo` for the enum class, conversion functions from/to strings.
This is controlled via the `-gen-enum-decls` and `-gen-enum-defs` command-line
options of `mlir-tblgen`.

For example, given the following `EnumAttr`:

```tablegen
def CaseA: EnumAttrCase<"caseA", 0>;
def CaseB: EnumAttrCase<"caseB", 10>;

def MyEnum: EnumAttr<"MyEnum", "An example enum", [CaseA, CaseB]> {
  let cppNamespace = "Outer::Inner";
  let underlyingType = "uint64_t";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```

The following will be generated via `mlir-tblgen -gen-enum-decls`:

```c++
namespace Outer {
namespace Inner {
// An example enum
enum class MyEnum : uint64_t {
  caseA = 0,
  caseB = 10,
};

llvm::StringRef ConvertToString(MyEnum);
llvm::Optional<MyEnum> ConvertToEnum(llvm::StringRef);
} // namespace Inner
} // namespace Outer

namespace llvm {
template<> struct DenseMapInfo<Outer::Inner::MyEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint64_t>;

  static inline Outer::Inner::MyEnum getEmptyKey() {
    return static_cast<Outer::Inner::MyEnum>(StorageInfo::getEmptyKey());
  }

  static inline Outer::Inner::MyEnum getTombstoneKey() {
    return static_cast<Outer::Inner::MyEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Outer::Inner::MyEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const Outer::Inner::MyEnum &lhs,
                      const Outer::Inner::MyEnum &rhs) {
    return lhs == rhs;
  }
};
}
```

The following will be generated via `mlir-tblgen -gen-enum-defs`:

```c++
namespace Outer {
namespace Inner {
llvm::StringRef ConvertToString(MyEnum val) {
  switch (val) {
    case MyEnum::caseA: return "caseA";
    case MyEnum::caseB: return "caseB";
    default: return "";
  }
}

llvm::Optional<MyEnum> ConvertToEnum(llvm::StringRef str) {
  return llvm::StringSwitch<llvm::Optional<MyEnum>>(str)
      .Case("caseA", MyEnum::caseA)
      .Case("caseB", MyEnum::caseB)
      .Default(llvm::None);
}
} // namespace Inner
} // namespace Outer
```

TODO(b/132506080): This following is outdated. Update it.

An attribute is a compile time known constant of an operation. Attributes are
required to be known to construct an operation (e.g., the padding behavior is
required to fully define the `conv2d` op).

Attributes are defined as having a storage type (corresponding to a derived
class of `mlir::Attribute`), a return type (that corresponds to the C++ type to
use in the generation of the helper accessors) as well as method to convert
between the internal storage and the helper method. Derived attributes are a
special class of attributes that do not have storage but are instead calculated
based on the operation and its attributes.

## Appendix

### Requirements and existing mechanisms analysis

The op description should as declarative as possible to allow a wide range of
tools to work with them and query methods generated from them. In particular
this means specifying traits, constraints and shape inference information in
a way that is easily analyzable (e.g., avoid opaque calls to C++ functions where
possible).

We considered the approaches of several contemporary systems and focused on
requirements that were desirable:

* Ops registered using a registry separate from C++ code.
  * Unknown ops are allowed in MLIR, so ops need not be registered. The
    ability of the compiler to optimize those ops or graphs containing those
    ops is constrained but correct.
  * The current proposal does not include a runtime op description, but it
    does not preclude such description, it can be added later.
  * The op registry is essential for generating C++ classes that make
    manipulating ops, verifying correct construction etc. in C++ easier by
    providing a typed representation and accessors.
* The op registry will be defined in
  [TableGen](https://llvm.org/docs/TableGen/index.html) and be used to
  generate C++ classes and utility functions
  (builder/verifier/parser/printer).
  * TableGen is a modelling specification language used by LLVM's backends
    and fits in well with trait based modelling. This is an implementation
    decision and there are alternative ways of doing this. But the
    specification language is good for the requirements of modelling the
    traits (as seen from usage in LLVM processor backend modelling) and easy
    to extend, so a practical choice. If another good option comes up, we
    will consider it.
* MLIR allows both defined and undefined ops.
  * Defined ops should have fixed semantics and could have a corresponding
    reference implementation defined using, for example, EDSC.
  * Dialects are under full control of the dialect owner and normally live
    with the framework of the dialect.
* The op's traits (e.g., commutative) are modelled along with the op in
  the registry.
* The op's operand/return type constraints are modelled along with the op in
  the registry (see [Type constraints](#type-constraints) discussion below),
  this allows (e.g.) optimized concise syntax in textual dumps.
* Behavior of the op is documented along with the op with a summary and a
  description. The description is written in markdown and extracted for
  inclusion in the generated LangRef section of the dialect.
* The generic assembly form of printing and parsing is available as normal,
  but a custom parser and printer can either be specified or automatically
  generated from an optional string representation showing the mapping of the
  "assembly" string to operands/type.
  * Parser-level remappings (e.g., `eq` to enum) will be supported as part
    of the parser generation.
* Matching patterns are specified separately from the op description.
  * Contrasted with LLVM there is no "base" set of ops that every backend
    needs to be aware of. Instead there are many different dialects and the
    transformations/legalizations between these dialects form a graph of
    transformations.
* Reference implementation may be provided along with the op definition.
  * The reference implementation may be in terms of either standard ops or
    other reference implementations.

    TODO: document expectation if the dependent op's definition changes.

### A proposal for auto-generating printer and parser methods

NOTE: Auto-generating printing/parsing (as explained in the below) has _not_
been prototyped, and potentially just being able to specify custom printer/
parser methods are sufficient. This should presumably be influenced by the
design of the assembler/disassembler logic that LLVM backends get for free
for machine instructions.

The custom assembly form of the operation is specified using a string with
matching operation name, operands and attributes. With the ability
to express additional information that needs to be parsed to build the
operation:

```tablegen
tfl.add $lhs, $rhs {fused_activation_function: $fused_activation_function}: ${type(self)}
```

1. The output is never shown in the "mnemonics" string as that is fixed form
   and cannot be altered.
1. Custom parsing of ops may include some punctuation (e.g., parenthesis).
1. The operands/results are added to the created operation in the order that
   they are shown in the input and output dags.
1. The `${type(self)}` operator is used to represent the type of the operator.
   The type of operands can also be queried.
1. Attributes names are matched to the placeholders in the mnemonic strings.
   E.g., attribute axis is matched with `$axis`. Custom parsing for attribute
   type can be defined along with the attribute definition.
1. The information in the custom assembly form should be sufficient to invoke
   the builder generated. That may require being able to propagate information
   (e.g., the `$lhs` has the same type as the result).

Printing is effectively the inverse of the parsing function generated with the
mnemonic string serving as a template.

### Shape inference

Type constraints are along (at least) three axis: 1) elemental type, 2) rank
(including static or dynamic), 3) dimensions. While some ops have no compile
time fixed shape (e.g., output shape is dictated by data) we could still have
some knowledge of constraints/bounds in the system for that op (e.g., the output
of a `tf.where` is at most the size of the input data). And so there are
additional valuable constraints that could be captured even without full
knowledge.

Initially the shape inference will be declaratively specified using:

*   Constraint on the operands of an operation directly. For example
    constraining the input type to be tensor/vector elements or that the
    elemental type be of a specific type (e.g., output of sign is of elemental
    type `i1`) or class (e.g., float like).
*   Constraints across operands and results of an operation. For example,
    enabling specifying equality constraints on type/constituents of a type
    (shape and elemental type) between operands and results (e.g., the output
    type of an add is the same as those of the input operands).

In general there is an input/output transfer function which maps the inputs to
the outputs (e.g., given input X and Y [or slices thereof] with these sizes, the
output is Z [or this slice thereof]). Such a function could be used to determine
the output type (shape) for given input type (shape).

But shape functions are determined by attributes and could be arbitrarily
complicated with a wide-range of specification possibilities. Equality
relationship are common (e.g., the elemental type of the output matches the
primitive type of the inputs, both inputs have exactly the same type [primitive
type and shape]) and so these should be easy to specify. Algebraic relationships
would also be common (e.g., a concat of `[n,m]` and `[n,m]` matrix along axis 0
is `[n+n, m]` matrix), while some ops only have defined shapes under certain
cases (e.g., matrix multiplication of `[a,b]` and `[c,d]` is only defined if
`b == c`). As ops are also verified, the shape inference need only specify rules
for the allowed cases (e.g., shape inference for matmul can ignore the case
where `b != c`), which would simplify type constraint specification.

Instead of specifying an additional mechanism to specify a shape transfer
function, the reference implementation of the operation will be used to derive
the shape function. The reference implementation is general and can support the
arbitrary computations needed to specify output shapes.



# Rewrite pattern description

TODO: Move this section to a dedicated doc for graph rewrites

MLIR aims to support many graph transformations across multiple levels of
representation using declarative patterns. These patterns can be expressed using
TableGen as well as dynamically (TBD).

## Op DAG pattern rewrites

The most direct pattern supported in MLIR is rewrites of the form `(dag of
operations) -> (dag of operations)` along with constraints (on operands and
operations). The matchers require both dialects being matched between to be
included in the same TableGen file. Hence pattern matching is normally defined
in either a separate file that imports both. Matchers are defined in terms of
the TableGen instances rather than mnemonics to allow for better error checking
and verification generation.

```tablegen
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>;
def : Pat<(TF_ReluOp (TF_AddOp $lhs, $rhs)), (TFL_AddOp $lhs, $rhs, TFL_AF_Relu)>;
def : Pat<(TF_BiasAddOp F32Tensor:$l, F32Tensor:$r),
          (TFL_AddOp $l, $r, TFL_AF_None)>;
```

In the above examples it was shown how to construct matching rules between two
dialects (TensorFlow and TensorFlowLite), showing matching arguments (attributes
and operands) as well as matching a DAG pattern of multiple input operations to
single output.

1.  Matchers can be partially specified on the input (e.g., not all arguments
    constrained) and so multiple matchers can match the same set of nodes. The
    most discriminative matcher (as determined by the number of
    constrained/matching terms) will be selected, if two patterns are equally
    discriminative then an error will be reported.

1.  Matchers between dialects have to be completely specified on the output
    (i.e., there can be no unspecified attributes of the op generated).

1.  Operands and attributes can be further constrained from the op definition
    (e.g., bias add rule only matches the case where both Tensors have F32
    elements).

    1.  Attributes can be transformed by transform rules to produce an attribute
        of a type different than the type matched.

TODO: Add constraints on the matching rules.

TODO: Describe the generation of benefit metric given pattern.

[TableGen]: https://llvm.org/docs/TableGen/index.html
[TableGenIntro]: https://llvm.org/docs/TableGen/LangIntro.html
[TableGenRef]: https://llvm.org/docs/TableGen/LangRef.html
[TableGenBackend]: https://llvm.org/docs/TableGen/BackEnds.html#introduction
[OpBase]: https://github.com/tensorflow/mlir/blob/master/include/mlir/IR/OpBase.td
[OpDefinitionsGen]: https://github.com/tensorflow/mlir/blob/master/tools/mlir-tblgen/OpDefinitionsGen.cpp
[EnumsGen]: https://github.com/tensorflow/mlir/blob/master/tools/mlir-tblgen/EnumsGen.cpp
