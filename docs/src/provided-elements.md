```@meta
CurrentModule = FerriteOperators
```

# Provided integrators and caches

The integrators and element caches the package ships: composition over one
domain, multi-domain routing, the automatic-differentiation decorator, the
transfer prolongators, the sum-factorization core a matrix-free
tensor-product element is written against, and the element-assembly decorator
that gives any bilinear cache the ELEMENT storage level. Element authors implement the
contracts on the [Element API reference](element-api.md) page; the types below
are ready-made implementations of them.

```@autodocs
Modules = [FerriteOperators]
Pages = [
    "elements/composite_elements.jl",
    "elements/domain_elements.jl",
    "elements/ad_element.jl",
    "elements/prolongators.jl",
    "elements/tensor_product.jl",
    "elements/element_assembly.jl",
]
```
