"""
    generate_crescent_moon(n::Integer, u::Integer, σ::AbstractFloat = 1)

Generate labelled and unlabelled crescent moon shaped data for two classes.

This function can be used to generate 2-dimensional data suitable for
semi-supervised learning algorithms. The data generated are for 2 classes with a
complex non-linear relationship.

# Arguments
 - `n`: the number of observations per class
 - `u`: the number of *unlabelled* observations per class (`u` must be no larger than `n`)
 - `σ`: the standard deviation of the noise on the arc of the crescent moon shape

"""
function generate_crescent_moon(n::Integer, u::Integer, σ::Number = 1)

    if u >= n throw(ArgumentError("Unlabelled data `u` must be less than class size `n`")) end

    # Generate points for the first class
    r = rand(Uniform(0, pi), n)
    x = @.  5cos(r) - 2.5 + rand(Normal(0, σ))
    y = @. 10sin(r) - 2.5 + rand(Normal(0, σ))
    c1 = [x y]

    # Generate points for the second class
    r = rand(Uniform(pi, 2pi), n)
    x = @.  5cos(r) + 2.5 + rand(Normal(0, σ))
    y = @. 10sin(r) + 2.5 + rand(Normal(0, σ))
    c2 = [x y]

    # Generate class labels
    labels = [repeat(["+"], n); repeat(["-"], n)]

    # Unlabel some data
    unlabels = allowmissing(labels)
    unlabels[[(n - u + 1):n; (2n - u + 1):2n]] .= missing

    # Package the result as a nice data-frame
    hcat(
        DataFrame(class_truth = labels, class_observed = unlabels), 
        DataFrame([c1; c2], ["x1", "x2"])
    )

end
