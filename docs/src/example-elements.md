```@meta
CurrentModule = FerriteOperatorsExampleElements
```

# Example elements

## Nesting the two-stage protocol

[`SimpleNestedHomogenization`](@ref) is the two-stage protocol nested inside
itself: a macroscopic bar whose stress at every quadrature point is the
homogenized response of a MICRO finite element problem, itself a condensed
element ([`SimpleRelaxingBar`](@ref)) with its own per-quadrature-point internal
variable. The [two-stage protocol](elements.md#The-two-stage-protocol) is the
contract at both levels, and the nesting is FerriteOperators inside
FerriteOperators: the micro problem is an ordinary operator built by
`setup_operator` and driven by `condense_internal!`/`update_linearization!`,
held as an element-cache field.

**Where each phase runs.** Phase one ([`condense_cell!`](@ref)) runs the micro
Newton per macro quadrature point. Every one of its iterations is itself a
phase one followed by a phase two — `condense_internal!` solves the micro
material's internal variables at the current micro state, then
`update_linearization!` evaluates the micro residual and CONDENSED micro tangent
there. Phase two of the macro element is a pure evaluation: the residual
re-evaluates the micro residual at the stored micro state (a sweep, no solve),
and the Jacobian reads the homogenized tangent phase one stored.

**The storage classes**, one field per class
([storage classes](elements.md#Storage-classes-for-elements-with-local-problems)):

| class | field | lifetime |
|---|---|---|
| immutable problem structure | `micro` — micro grid, `DofHandler`, driven/free dof split, the micro measure | built once in `setup_element_cache`, ALIASED across workers |
| per-worker solve workspace | `workspace` — the worker's OWN micro operator and its state/history/residual buffers | rebuilt by `duplicate_for_device`, never shared |
| per-item state | `tangents` — an [`ItemStates`](@ref) store of each quadrature point's `(frozen, consistent)` homogenized tangent | aliased, item slots disjoint per worker, dropped by [`rollback_state!`](@ref) |

The micro operator belongs to the second class and not the first: it owns a
matrix, element caches and corrector stores that a sweep writes, so sharing one
across workers is the race an aliased cache must never introduce.

**Where the micro state lives.** The whole micro state `[micro ū; micro q]` of
every macro quadrature point rides the macro `[ū; q]` tail — the macro element's
`get_number_of_internal_dofs_per_element` returns `nqp × (micro ū + micro q)`
per cell — so trial write-back, [`rollback_state!`](@ref) and
[`commit_state!`](@ref) carry the micro states with no second mechanism. The
micro Newton starts from the COMMITTED micro state rather than the current trial
tail, which is what keeps phase one a function of `(ū, qprev)` alone: warm
starting from the tail would make the converged `q` depend on how many times the
caller condensed on the way there.

**The composition of implicit function theorems.** The micro tangent `K` that
the micro Newton steps with already has the micro internal variables eliminated
(the micro element's own `Consistent` correction). Eliminating the micro
equilibrium on top of it is the second implicit function theorem, and it is the
Schur complement of that same `K` onto the driven dofs:

```math
\frac{\mathrm{d}\bar\sigma}{\mathrm{d}\bar\varepsilon}
  = \frac{1}{|\Omega_{micro}|}\, x^d \cdot
    \left(K_{dd} - K_{df} K_{ff}^{-1} K_{fd}\right) x^d .
```

The [`CorrectionMode`](@ref) composes with it: the `FrozenQ` partial of the
macro element is the same contraction over the micro problem's own FROZEN
tangent — no micro equilibrium, no micro internal response — so the macro mode
selects which micro tangent the contraction runs over. The
[`CondensationReport`](@ref) composes the same way: a micro condensation that did
not converge makes the macro-local problem unconverged, so a failure at either
level surfaces in the report the macro `condense_internal!` returns.

**The boundary.** A nested element is ANALYTIC-PROVIDER territory. No generic or
AD route reaches through two levels: the decorator's generic `Consistent`
combination ([`condensed_corrector`](@ref)) and the [`local_conditions!`](@ref)
route both differentiate ONE element kernel, and a kernel that runs a nested
operator's sweeps is not eltype-generic — it hands Float64 buffers to another
operator. Those routes are also sized for one internal dof per quadrature point,
which the `nqp × (micro ū + micro q)` count of a nested element is not.
`SimpleNestedHomogenization` therefore declares [`provides_analytic`](@ref) for
the Jacobian-shaped kinds it serves and nothing else, and the sweep's parameter
bag is not passed down: parameter and time sensitivities through two levels need
the same implicit function theorem composition once more, one level deeper.

## Reference

```@autodocs
Modules = [FerriteOperatorsExampleElements]
```
