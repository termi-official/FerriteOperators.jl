```@meta
CurrentModule = FerriteOperators
```

# FerriteOperators

*A SciML compatible high performance parallel assembly system for [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)*.

!!! note
    For an assembly framework in Ferrite.jl style we refer users for now to [FerriteAssembly.jl](https://github.com/KnutAM/FerriteAssembly.jl).

!!! warning
    This package is under heavy development. Expect regular breaking changes
    for now. If you are interested in joining development, then either comment
    an issue or reach out via julialang.zulipchat.com, via mail or via 
    julialang.slack.com. Alternatively open a discussion if you have something 
    specific in mind.

!!! note
    If you are interested in using this package, then I am also happy to
    to get some constructive feedback, especially if things don't work out
    in the current design. This can be done via julialang.slack.com,
    julialang.zulipchat.com or via mail.

## The Element Interface

For users the most important piece is the element interface.
Users need to provide some structs and corresponding dispatches to work with FerriteOperators.jl.

Essentially there are three super-types for elements

```@docs
FerriteOperators.AbstractVolumetricElement
FerriteOperators.AbstractSurfaceElementCache
FerriteOperators.AbstractInterfaceElementCache
assemble_element!
assemble_face!
assemble_interface!
FerriteOperators.setup_element_cache
FerriteOperators.load_element_unknowns!

```

Only `FerriteOperators.AbstractVolumetricElement` is implemented for now and it covers already all typical use-cases.

Furthermore, each element formulation is derived from an integrator. Integrators are the bridge between elements and materials.
Right now, these types of integrators are provided

```@docs
FerriteOperators.AbstractBilinearIntegrator
FerriteOperators.AbstractNonlinearIntegrator
FerriteOperators.AbstractLinearIntegrator
```

## The Setup Interface

The main entry point for users is the function

```@docs
setup_operator
```

which takes a strategy, the integrator and a matching dof handler.
Here the strategy controls the type of parallelism, the used device (e.g. threaded CPU or GPU) and the integrator is the hub controlling what exactly will be assembled.

```@docs
SequentialCPUDevice
PolyesterDevice
```

```@docs
SequentialAssemblyStrategy
PerColorAssemblyStrategy
ElementAssemblyStrategy
```
