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
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/termi-official/FerriteOperators.jl",
    devbranch="main",
)
