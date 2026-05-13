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

## Architecture Overview

FerriteOperators sits between Ferrite modeling code and solver code. It provides
a flexible job system to define generic finite element operators. These typically
assemble sparse matrices, residual vectors, or apply matrix-free actions with
user-defined element formulations. The task system allows to execute these
actions either sequentially or in parallel and on
different devices (CPU threads, GPUs, ...).

The assembly pipeline is built around four layers:

1. **Strategies** decide *how* to partition work into items (sequential, per-color, element-assembly / matrix-free).
2. **Devices** decide *where* to execute (e.g. sequential on the CPU, threaded via Polyester, or GPU via KernelAbstractions).
3. **Tasks** encode *what* to execute on a device.
4. **Workspaces** hold the pre-allocated per-worker scratch data (e.g. element cache, cell cache, local matrices/vectors, ...) allowing them to execute their assigned tasks independently.

These layers compose into a single generic device loop shared by all operator types, implemented as `FerriteOperators.execute_on_device!`:

```
for chunk in partitions
    parfor taskid in chunk
        reinit!(workspace, taskid)
        execute_single_task!(task, workspace)
    end
end
```

where the partition is computed at setup time by `compute_partition(strategy, sdh)` and
encodes the work distribution (single batch for sequential, color groups for per-color,
etc.). The device cache contains the workspace(s) for each parallel worker, constructed
by `setup_device_instances(device, obj, n_workers)`, which creates independent copies of
`obj` via `duplicate_for_device(device, obj)` for parallel execution. The function
works on any duplicable object, not only workspaces.

Square operators (bilinear, nonlinear, linear) use an [`AssemblyWorkspace`](@ref)
that holds the local element matrix `Ke`, unknown vector `ue`, residual vector
`re`, geometry cache, internal variable handler, and element cache.

Transfer operators (prolongation/restriction) use a [`TransferWorkspace`](@ref)
with the rectangular element matrix `Pe`, a transfer cell cache, and the
transfer element cache.

Adding a new operator type typically only requires defining a new task type and
implementing `execute_single_task!` for the appropriate workspace - the device loop,
strategy infrastructure, and parallel duplication remain unchanged.

## The Element Interface

For users the most important piece is the element interface.
Users need to provide some structs and corresponding dispatches to work with FerriteOperators.jl.

Essentially there are three super-types for elements

```@docs
FerriteOperators.AbstractVolumetricElement
FerriteOperators.AbstractSurfaceElementCache
FerriteOperators.AbstractInterfaceElementCache
assemble_element!
assemble_facet!
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

## Transfer Operators

Transfer operators assemble rectangular sparse matrices for prolongation and
restriction between two `DofHandler`s.

```@docs
setup_transfer_operator
setup_nested_transfer_operator
FerriteOperators.AbstractTransferIntegrator
FerriteOperators.AbstractTransferElementCache
```

## The Setup Interface

The main entry point for users is the function

```@docs
setup_operator
```

which takes a strategy, the integrator and a matching dof handler.
Here the strategy controls the type of parallelism, the used device (e.g. threaded CPU or GPU) and the integrator is the hub controlling what exactly will be assembled.

## Devices

```@docs
SequentialCPUDevice
PolyesterDevice
```

## Strategies

```@docs
SequentialAssemblyStrategy
PerColorAssemblyStrategy
ElementAssemblyStrategy
```
