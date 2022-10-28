module GraphSSL

using DataFrames
using Distances
using LinearAlgebra
using IterativeSolvers
using Distributions
using Random

export ssl_predict, generate_crescent_moon

include("utils.jl")
include("internals.jl")
include("data-generation.jl")

function ssl_predict(data::AbstractDataFrame, target, features; id = nothing, k = 5, cmn = true, dist_type = Euclidean(), weighted = false, epsilon = 2, exact = false)

    X, Y, classes, u = prepare_input_data(data, target, features, id = id)

    A = construct_graph(X, k; dist_type = dist_type, weighted = weighted, ϵ = epsilon)

    Ŷ = solve_harmonic_function(A, Y; exact = exact)

    if cmn
        preds = predict_class(Ŷ, u, classes, prior = sum(Y, dims = 1))
    else
        preds = predict_class(Ŷ, u, classes)
    end

end

end
