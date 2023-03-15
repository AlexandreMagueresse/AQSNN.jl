###################
# GeneralPolytope #
###################
struct GeneralPolytope{T,M,E,C,S,I} <: AbstractPolytope{T,M,E}
  context::C
  store::S
  indices::I

  function GeneralPolytope(context::C, store::S, indices::I) where {C,S,I}
    T = eltype(context)
    M = mandim(context)
    E = embdim(context)
    new{T,M,E,C,S,I}(context, store, indices)
  end
end

getStore(polytope::GeneralPolytope) = polytope.store
getIndices(polytope::GeneralPolytope) = polytope.indices

##########
# Traits #
##########
isSimplex(::Type{<:GeneralPolytope}) = false
isReference(::Type{<:GeneralPolytope}) = false
isConvex(::Type{<:GeneralPolytope}) = false

####################
# AbstractPolytope #
####################
getContext(polytope::GeneralPolytope) = polytope.context
