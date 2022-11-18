"""
Gaussian radial basis function
"""
function radial_basis(r, ϵ)
    exp(-(r / ϵ)^2)
end