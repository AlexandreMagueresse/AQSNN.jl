abstract type AbstractQuadrature{T,M,E} end

###############
# initialise! #
###############
function initialise!(::AbstractQuadrature)
  @abstractmethod
end

function initialise!(quadratures::AbstractVector{<:AbstractQuadrature})
  @simd for quadrature in quadratures
    initialise!(quadrature)
  end
end

#############
# integrate #
#############
function integrate(::AbstractIntegrand, ::AbstractQuadrature)
  @abstractmethod
end

function Base.:*(integrand::AbstractIntegrand, quadrature::AbstractQuadrature)
  integrate(integrand, quadrature)
end

function Base.:*(
  integrand::AbstractIntegrand,
  quadratures::AbstractVector{<:AbstractQuadrature{T}}
) where {T}
  s = zero(T)
  @simd for quadrature in quadratures
    s += integrand * quadrature
  end
  s
end
