using FerriteOperators
using Documenter

DocMeta.setdocmeta!(FerriteOperators, :DocTestSetup, :(using FerriteOperators); recursive=true)

makedocs(;
    modules=[FerriteOperators],
    authors="Dennis Ogiermann <termi-official@users.noreply.github.com> and contributors",
    sitename="FerriteOperators.jl",
    format=Documenter.HTML(;
        canonical="https://termi-official.github.io/FerriteOperators.jl",
        edit_link="main",
        assets=String[],
    ),
    warnonly=true,
    pages=[
        "Home" => "index.md",
        "Writing elements" => "elements.md",
        "Operators and entry points" => "operators.md",
        "Patch items" => "patches.md",
        "The layer contract" => "design.md",
        "Migration to v2" => "migration.md",
    ],
)

deploydocs(;
    repo="github.com/termi-official/FerriteOperators.jl",
    devbranch="main",
    push_preview = true,
)
