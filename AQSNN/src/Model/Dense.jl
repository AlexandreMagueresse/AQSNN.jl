#########
# Dense #
#########
struct Dense{T,F} <: Function
  weight::Param{Matrix{T}}
  bias::Param{Vector{T}}
  activation::F
end

function Dense(
  nIn::Int, nOut::Int, activation::F,
  weightInitialiser::AbstractInitialiser{T},
  biasInitialiser::AbstractInitialiser{T}
) where {T,F<:Function}
  Dense{T,F}(
    Param(Knet.atype(weightInitialiser(nIn, nOut, (nOut, nIn)))),
    Param(Knet.atype(biasInitialiser(nIn, nOut, (nOut,)))),
    # Param((weightInitialiser(nIn, nOut, (nOut, nIn)))),
    # Param((biasInitialiser(nIn, nOut, (nOut,)))),
    activation
  )
end

###########
# Forward #
###########
function ∇⁰(d::Dense, X)
  d.activation.(d.weight * X .+ d.bias)
end

(d::Dense)(X) = ∇⁰(d, X)
