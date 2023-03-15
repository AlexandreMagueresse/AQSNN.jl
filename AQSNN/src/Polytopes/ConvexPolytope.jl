##################
# ConvexPolytope #
##################
struct ConvexPolytope{T,M,E,C,S,I} <: AbstractPolytope{T,M,E}
  context::C
  store::S
  indices::I

  function ConvexPolytope(context::C, store::S, indices::I) where {C,S,I}
    T = eltype(context)
    M = mandim(context)
    E = embdim(context)
    new{T,M,E,C,S,I}(context, store, indices)
  end
end

function ConvexPolytope(polytope, indices)
  ConvexPolytope(getContext(polytope), getStore(polytope), indices)
end

getStore(polytope::ConvexPolytope) = polytope.store
getIndices(polytope::ConvexPolytope) = polytope.indices

##########
# Traits #
##########
isSimplex(::Type{<:ConvexPolytope}) = false
isReference(::Type{<:ConvexPolytope}) = false
isConvex(::Type{<:ConvexPolytope}) = true

####################
# AbstractPolytope #
####################
getContext(polytope::ConvexPolytope) = polytope.context
