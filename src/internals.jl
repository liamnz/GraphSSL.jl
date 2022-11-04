# Preprocessing ----------------------------------------------------------------
function prepare_input_data(data, target, features, id = nothing)

    # Get IDs for the unlabeled observations so that these can be attributed to
    # the predictions later on. If no ID variable is given, return a row number
    # instead.
    u_mask = ismissing.(data[:, target])

    if isnothing(id)
        row_id = collect(1:nrow(data))
        u = row_id[u_mask]
    else 
        u = data[u_mask, id]
    end

    labels = data[.!u_mask, target]
    classes = unique(labels)
    Y = falses(length(labels), 2)
    Y[:, 1] = labels .== classes[1]
    Y[:, 2] = labels .== classes[2]

    # Re-order the rows of the data matrix so that the labelled observations are
    # at the start of the matrix, this is what is expected to solve the harmonic
    # function later on.
    reorder = [findall(.!u_mask); findall(u_mask)]
    X = Matrix(data[reorder, features])

    (X, Y, classes, u)

end

# Graph Construction -----------------------------------------------------------
# We'll construct a sparse graph by connecting observations based on a proximity
# metric. The graph will be represented by an 'Adjacency Matrix', which is a
# symmetric indicator matrix where a 1 indicates that those observations are
# connected, otherwise 0. Connected observations can then be optionally weighted by
# some function of their distance to represent how 'closely' connected they are.

function construct_graph(X, k; dist_type = Euclidean(), weighting = nothing)

    n = size(X, 1)

    # First, compute distances on the entire data
    D = pairwise(dist_type, X, dims = 1)

    # Update the diagonal of the distance matrix with NaNs so that an observation's
    # distance to itself is not considered in the nearest neighbours computation.
    D[diagind(D)] .= NaN

    # Preallocate a 'Nearest Neighbour' matrix and the Adjacency matrix
    NN = A = falses(n, n)

    # For every observation in the distance matrix
    for i in 1:n
        
        # Find the column indices for the nearest k neighbours. 
        nearest_neighbours = partialsortperm(D[i, :], 1:k)

        # Assign an indicator for the neighbours of each observation
        NN[i, nearest_neighbours] .= true

    end

# For the scary math to to work, the adjacency matrix (A) must be symmetric and
# non-negative. The Nearest Neighbours matrix is non-negative but not
# necessarily symmetric. For example, if observations α and β have a neighbour
# γ, it does not necessarily mean that α and β are also neighbours. So we can't
# use the Nearest Neighbour matrix as the Adjacency matrix, BUT if we connect
# observations α and β, when α is in β's nearest neighbourhood OR when β is in
# α's nearest neighbours then the resulting matrix is symmetric and we have a
# proper Adjaceny matrix that connects observations in a graph structure. We can
# achieve this by doing an elementwise 'OR' comparison with the nearest neighbour
# matrix and its tranpose.
    A = NN .| transpose(NN)

    if !issymmetric(A)
        throw(ErrorException("The computed Adjacency matrix is not symmetric, something has gone wrong but I don't know what."))
    end

    # Use a weighted adjacency matrix?
    if !isnothing(weighting)
        connected = findall(A)
        W = zeros(n, n)
        W[connected] = weighting.(D[connected])
        return W
    else 
        return A
    end

end

# Solve the Harmonic Function --------------------------------------------------
function solve_harmonic_function(A, Y; exact = false)

    # The solution is computed on the 'Laplacian' of the graph (Δ), which is the
    # 'Degree matrix' minus the Adjacency matrix: Δ = D - A. Where the Degree
    # matrix (D) is a diagonal matrix which counts how many connections each
    # observation has in the graph (or the total weight of the connections in a
    # weighted graph).
    D = Diagonal(vec(sum(A, dims = 2)))
    Δ = D - A

    # The harmonic function is the solution to the linear system
    #
    #   Δᵤᵤfᵤ = -Δᵤₗfₗ
    #
    # And therefore
    #
    #   fᵤ = -Δᵤᵤ⁻¹ Δᵤₗfₗ
    #
    # Where:
    #   - fᵤ is a matrix of class probability estimates for the unlabeled
    #     observations
    #   - fₗ is matrix of binary indicators for the classes of the labelled
    #     observations
    #   - Δᵤᵤ is the partition of the Laplacian matrix for the rows and columns
    #     of the unlabelled observations
    #   - Δᵤₗ is the partition of the Laplacian matrix for the unlabelled rows
    #     and the labelled columns

    # Prepare the matrices for the linear system.
    n = size(Δ, 1)
    l = size(Y, 1)
    Δᵤᵤ = Δ[(l + 1):n, (l + 1):n]
    Δᵤₗ = Δ[(l + 1):n, 1:l]
    fₗ = Y

    # The exact solution can be computed using Julia's built-in linear solver.
    # But that involves computing the matrix inverse of Δᵤᵤ directly which is
    # computationally expensive for even modest sized datasets of a few thousand
    # unlabelled observations.
    if exact

        fᵤ = -(Δᵤᵤ \ Δᵤₗ * fₗ)

    else

        # As per Zhou (2005), we can instead solve the linear system using the
        # iterative 'Conjugate Gradient' (CG) method which provides an accurate
        # approximation to the extact solution but with considerably less
        # computation.

        # Need to convert the input data into Floats because the cg() solver
        # function can't handle Integer or BitVector inputs (this will increase
        # the data size but hopefully thats a worthwhile tradeoff for the major
        # speed boost)
        Δᵤᵤ = convert(Matrix{Float64}, Δᵤᵤ)
        Δᵤₗ = convert(Matrix{Float64}, Δᵤₗ)
        fₗ = convert(Matrix{Float64}, fₗ)

        # Compute the solution using the CG iterative solver. N.B. using a the
        # diagnonal of Δᵤᵤ as a (left) precondtioner as recommended in Zhu
        # (2005) to aid convergence of the CG algorithm.
        fᵤ = Matrix{Float64}(undef, n - l, 2)
        fᵤ[:, 1] = -cg(Δᵤᵤ , Δᵤₗ * fₗ[:, 1] ; Pl = Diagonal(Δᵤᵤ))

        # As the CG solver finds an approximate solution it is possible for the
        # values to occaisionally be outside the interval (0, 1) by a small
        # amount, so when that happens we'll clamp the values so that they
        # remain in a valid interval for probabilities.
        fᵤ[:, 1] = clamp.(fᵤ[:, 1], eps(), 1 - eps())

        # Add the complement
        fᵤ[:, 2] = 1 .- fᵤ[:, 1]
    
    end

    fᵤ

end

# Prediction -------------------------------------------------------------------
function assign_class(Ŷ, classes; id = nothing, prior = nothing)

    preds = DataFrame(
        prob_class1 = Ŷ[:, 1],
        prob_class2 = Ŷ[:, 2],
    )

    if !isnothing(id)

        preds[:, :id] = id
        select!(preds, [:id, :prob_class1, :prob_class2])

    end

    if !isnothing(prior)

        class_mass = sum(Ŷ, dims = 1)
        Y_cmn = Ŷ .* prior ./ class_mass
        cmn_pred = Y_cmn .== findmax(Y_cmn, dims = 2)[1]
        preds[!, :cmn_class1] = Y_cmn[:, 1]
        preds[!, :cmn_class2] = Y_cmn[:, 2]
        preds[!, :pred_class] = ifelse.(cmn_pred[:, 1], classes[1], classes[2])

    else

        prob_pred = Ŷ .== findmax(Ŷ, dims = 2)[1]
        preds[!, :pred_class] = ifelse.(prob_pred[:, 1], classes[1], classes[2])
        
    end

    preds

end
