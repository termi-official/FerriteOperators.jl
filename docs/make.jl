using FerriteOperators
using FerriteOperatorsExampleElements
using Documenter

DocMeta.setdocmeta!(FerriteOperators, :DocTestSetup, :(using FerriteOperators); recursive=true)
DocMeta.setdocmeta!(FerriteOperatorsExampleElements, :DocTestSetup, :(using FerriteOperators, FerriteOperatorsExampleElements); recursive=true)

makedocs(;
    modules=[FerriteOperators, FerriteOperatorsExampleElements],
    authors="Dennis Ogiermann <termi-official@users.noreply.github.com> and contributors",
    sitename="FerriteOperators.jl",
    format=Documenter.HTML(;
        canonical="https://termi-official.github.io/FerriteOperators.jl",
        edit_link="main",
        assets=String[],
        # Page size is not a quality criterion here: the nav sidebar renders
        # into every page, so the thresholds mostly measure nav length, and
        # chasing them fragments the API reference. Disabled by decision.
        size_threshold = nothing,
        size_threshold_warn = nothing,
    ),
    warnonly=true,
    pages=[
        "Home" => "index.md",
        "Writing elements" => "elements.md",
        "Element API reference" => "element-api.md",
        "Provided integrators and caches" => "provided-elements.md",
        "Example elements" => "example-elements.md",
        "Operators and entry points" => "operators.md",
        "Operator API reference" => "operator-api.md",
        "Assembly engine API reference" => "engine-api.md",
        "Patch items" => "patches.md",
        "Migration to 0.4" => "migration.md",
        "Developer documentation" => [
            "The layer contract" => "devdocs/design.md",
            "Design rationale" => "devdocs/rationale.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/termi-official/FerriteOperators.jl",
    devbranch="main",
    push_preview = true,
)
