```@meta
CurrentModule = FerriteOperators
```

# Provided integrators and caches

The integrators and element caches the package ships: composition over one
domain, multi-domain routing, the automatic-differentiation decorator, the
transfer prolongators, and the element-assembly decorator that gives any
bilinear cache the ELEMENT storage level. Element authors implement the
contracts on the [Element API reference](element-api.md) page; the types below
are ready-made implementations of them.

```@autodocs
Modules = [FerriteOperators]
Pages = [
    "elements/composite_elements.jl",
    "elements/domain_elements.jl",
    "elements/ad_element.jl",
    "elements/prolongators.jl",
    "elements/element_assembly.jl",
]
```

## The tensor-product sum-factorization core

The sum-factorization core a matrix-free tensor-product element is written
against — the 1D reference operators, the lattice permutations, the
contraction pipeline, and the two mapping pipelines — plus its two reference
consumers, ships as its own lib subpackage,
[FerriteOperatorsTensorProduct](https://github.com/termi-official/FerriteOperators.jl/tree/main/lib/FerriteOperatorsTensorProduct).

```@meta
CurrentModule = FerriteOperatorsTensorProduct
```

```@autodocs
Modules = [FerriteOperatorsTensorProduct]
```
