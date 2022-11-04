module GraphSSL

using DataFrames
using Distances
using LinearAlgebra
using IterativeSolvers
using Distributions
using Random

export predict, radial_basis, generate_crescent_moon

include("data-generation.jl")
include("weighting-functions.jl")
include("internals.jl")

# Dataframe interface
function predict(data::AbstractDataFrame, target, features, id = nothing; cmn::Bool = true, k::Integer = 5, dist_type::Union{PreMetric, SemiMetric, Metric} = Euclidean(), weighting::Function = x -> radial_basis(x, 2), exact::Bool = false)

    X, Y, classes, u = prepare_input_data(data, target, features, id)

    A = construct_graph(X, k; dist_type = dist_type, weighting = weighting)

    Ŷ = solve_harmonic_function(A, Y; exact = exact)

    if cmn
        preds = assign_class(Ŷ, classes; id = u, prior = sum(Y, dims = 1))
    else
        preds = assign_class(Ŷ, classes; id = u)
    end

end

# Adjacency matrix interface
function predict(A::AbstractMatrix, target::AbstractVector; id = nothing, cmn = true, exact = false)

    classes = unique(target)
    Y = [target .== classes[1] target .== classes[2]]  


    Ŷ = solve_harmonic_function(A, Y; exact = exact)

    if cmn
        preds = assign_class(Ŷ, classes; id = id, prior = sum(Y, dims = 1))
    else
        preds = assign_class(Ŷ, classes; id = id)
    end

end

end # module

