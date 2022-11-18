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
"""
    predict(data::AbstractDataFrame, target, features, id = nothing; cmn::Bool = true, k::Integer = 5, dist_type::Union{PreMetric, SemiMetric, Metric} = Euclidean(), weighting::Function = x -> radial_basis(x, 2), exact::Bool = false)

Use Zhou's Graph-based Semi-Supervised learning algorithm to predict the missing
labels in a set of partially labelled data.

The algorithm proceeds by first constructing a sparse graph of the input data
represented by an adjancency matrix. The graph is derived from a nearest
neighbour calculation, with two observations being connected in the graph if
either of them are within the other's set of nearest neighbours. 

Then a linear system is constructed from certain labelled and unlabelled
partitions of the adjacency matrix is solved. This linear system is derived from
the theory of Gaussian Random Fields and Harmonic Functions as articulated by
Zhou in his 2005 PhD thesis. 

The solution to this system can be intrepreted as the probability
that an unlabelled observation reaches a labelled point of a given class in a
random walk on the graph. The unlabelled observations are then are assigned to
the class with the higher probability.

The design of the graph (ie adjancency matrix) has the most influence on the
quality of the predictions, so this function provides many options to tweak how
the graph is constructed. The number of options may be overwhelming at first so
the function has sensible defaults which should be acceptable in most cases.

# Arguments
 * `data`: A data-frame which contains a `target` variable of partially labelled
   observations and `features` which can be used to predict the
   unlabelled observations. Unlabelled observatios in `target` should be
   represented as `missing` values.
 * `target`: A scalar value for the column in `data` which represents the target
   variable. This should be a valid data-frame indexing value, usually
   `AbstractString` or `Symbol`, but there are many flexible options see
   [DataFrames.jl for more
   information](https://dataframes.juliadata.org/stable/lib/indexing/).
 * `features`: A value for the columns in `data` which represent the predictors
   variables. This should be a valid data-frame indexing value, usually an array
   of `AbstractString`s or `Symbol`s, but there are many flexible options see
   [DataFrames.jl for more
   information](https://dataframes.juliadata.org/stable/lib/indexing/).
 * `id`: The name of a variable used to identify an observation in `data`, this
   will be appended to the returned predictions. If `id = nothing` a row ID will
   be created instead.
 * `cmn`: Should 'Class Mass Normalisation' be used to assign classes to the
  unlabelled observations? CMN is a heuristic proposed by Zhou which adjusts the
  class probability estimates by the prior frequency of the classes in the
  labelled data. In the presence of unbalanced classes, this usually leads to
  better class predictions than using the naive probability estimates from the
  initial solution. If the classes are already balanced it has the same effect
  as using the naive probabilities, so `CMN = True` by default.
 * `k`: The number of neighbours used to construct the sparse graph. A small
   number of neighbours is usually sufficient. As `k` gets larger then
   `dist_type` and `weighting` play more of a role in the solution.
 * `dist_type`: The kind of distance measure used to determine whether one
   observation is near another and therefore connected in the graph. `dist_type`
   expects a `PreMetric`, `SemiMetric` or `Metric` type as defined by the
   [Distances.jl](https://github.com/JuliaStats/Distances.jl) package. Note
   that, depending on the kind of data one has constructed, Euclidean distance
   may not always be a good way to characterise the distance between
   observations, especially in high-dimensional settings.
 * `weighting`: A function used to transform the distance value provided by
   `dist_type` and create a *weighted* adjancency matrix for the graph, where the
   weights represent how 'closely' connected the observations are to each other.
   A radial basis function is used by default but the user can provide an
   arbitrary anonymous function for more exotic weights. To use an unweighted
   adjacency matrix set `weighting = nothing`.
 * `exact`: Should the linear system be solved exactly? An exact solution to the
   linear system involves a matrix inversion which is computationally expensive,
   even for modest sized datasets of a few thousand observations. When `exact =
   False` the system will be solved approximately using the Conjugate Gradient
   method as implemented by the
   [IterativeSolvers.jl](https://iterativesolvers.julialinearalgebra.org/stable/)
   package. While this iterative method is approximate, it is *much* faster and
   usually an extremely accurate approximation to the exact solution.
 
# Failure modes

The algorithm requires the solution to a linear system, but occaisionaly you
might encounter an error that the system is not solvable. This may be to do with
the structure of the graph, eg you may have inadvertantly created a sub-graph
where some unlablled observations are completely disconnected from any labelled
obersvations. 

**For the linear system to be solveable each unlabelled observation must be able
to traverse the graph to reach at least one labelled observation.**

If you encounter an insolubility error try increasing `k` so that potentially
isloated sub-graphs are reconnected to the main graph.
"""
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

