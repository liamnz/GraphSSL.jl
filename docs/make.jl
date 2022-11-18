using Documenter, GraphSSL

makedocs(
    sitename = "GraphSSL",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)