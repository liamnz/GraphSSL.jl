# Default weighting function to use when a weighted adjacency matrix is
# requested by the user.
function radial_basis(r, ϵ)
    exp(-(r / ϵ)^2)
end