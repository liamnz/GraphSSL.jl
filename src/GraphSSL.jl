module GraphSSL

using DataFrames
using Distances
using LinearAlgebra
using IterativeSolvers
using Distributions
using Random

export predict, generate_crescent_moon

include("utils.jl")
include("internals.jl")
include("data-generation.jl")

# Dataframe interface
function predict(data::AbstractDataFrame, target, features, id = nothing; cmn::Bool = true, k::Integer = 5, dist_type::Union{PreMetric, SemiMetric, Metric} = Euclidean(), weighted::Bool = false, epsilon::Number = 2, exact::Bool = false)

    X, Y, classes, u = prepare_input_data(data, target, features, id)

    A = construct_graph(X, k; dist_type = dist_type, weighted = weighted, ϵ = epsilon)

    Ŷ = solve_harmonic_function(A, Y; exact = exact)

    if cmn
        preds = assign_class(Ŷ, classes; id = u, prior = sum(Y, dims = 1))
    else
        preds = assign_class(Ŷ, classes; id = u)
    end

end

# Adjacency graph interface
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

