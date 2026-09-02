```@meta
CurrentModule = FerriteOperators
```

# Provided integrators and caches

The integrators and element caches the package ships: composition over one
domain, multi-domain routing, the automatic-differentiation decorator, and the
transfer prolongators. Element authors implement the contracts on the
[Element API reference](element-api.md) page; the types below are ready-made
implementations of them.

```@autodocs
Modules = [FerriteOperators]
Pages = [
    "elements/composite_elements.jl",
    "elements/domain_elements.jl",
    "elements/ad_element.jl",
    "elements/prolongators.jl",
]
```
