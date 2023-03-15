abstract type ReferencePolytope{T,M,E} <: AbstractPolytope{T,M,E} end

#########
# Point #
#########
struct Point{T,E,C,S,I} <: ReferencePolytope{T,0,E}
  context::C
  store::S
  indices::I

  function Point(context::C, store::S, indices::I) where {C,S,I}
    @assert mandim(context) == 0
    @assert length(indices) == 1

    T = eltype(context)
    E = embdim(context)
    new{T,E,C,S,I}(context, store, indices)
  end
end

getStore(polytope::Point) = polytope.store
getIndices(polytope::Point) = polytope.indices

getContext(polytope::Point) = polytope.context

isSimplex(::Type{<:Point}) = true
isReference(::Type{<:Point}) = true
isConvex(::Type{<:Point}) = true

###########
# Segment #
###########
struct Segment{T,E,C,S,I} <: ReferencePolytope{T,1,E}
  context::C
  store::S
  indices::I

  function Segment(context::C, store::S, indices::I) where {C,S,I}
    @assert mandim(context) == 1
    @assert length(indices) == 2

    T = eltype(context)
    E = embdim(context)
    new{T,E,C,S,I}(context, store, indices)
  end
end

getStore(polytope::Segment) = polytope.store
getIndices(polytope::Segment) = polytope.indices

getContext(polytope::Segment) = polytope.context

isSimplex(::Type{<:Segment}) = true
isReference(::Type{<:Segment}) = true
isConvex(::Type{<:Segment}) = true

############
# Triangle #
############
struct Triangle{T,E,C,S,I} <: ReferencePolytope{T,2,E}
  context::C
  store::S
  indices::I

  function Triangle(context::C, store::S, indices::I) where {C,S,I}
    @assert mandim(context) == 2
    @assert length(indices) == 3

    T = eltype(context)
    E = embdim(context)
    new{T,E,C,S,I}(context, store, indices)
  end
end

getStore(polytope::Triangle) = polytope.store
getIndices(polytope::Triangle) = polytope.indices

getContext(polytope::Triangle) = polytope.context

isSimplex(::Type{<:Triangle}) = true
isReference(::Type{<:Triangle}) = true
isConvex(::Type{<:Triangle}) = true

##############
# Quadrangle #
##############
struct Quadrangle{T,E,C,S,I} <: ReferencePolytope{T,2,E}
  context::C
  store::S
  indices::I

  function Quadrangle(context::C, store::S, indices::I) where {C,S,I}
    @assert mandim(context) == 2
    @assert length(indices) == 4

    T = eltype(context)
    E = embdim(context)
    new{T,E,C,S,I}(context, store, indices)
  end
end

getStore(polytope::Quadrangle) = polytope.store
getIndices(polytope::Quadrangle) = polytope.indices

getContext(polytope::Quadrangle) = polytope.context

isSimplex(::Type{<:Quadrangle}) = false
isReference(::Type{<:Quadrangle}) = true
isConvex(::Type{<:Quadrangle}) = true

########################
# General constructors #
########################
function SimplexPolytope(polytope, indices)
  context = getContext(polytope)
  store = getStore(polytope)

  L = length(indices)
  if L == 1
    Point(context, store, indices)
  elseif L == 2
    Segment(context, store, indices)
  elseif L == 3
    Triangle(context, store, indices)
  else
    @notreachable
  end
end

function ReferencePolytope(polytope, indices)
  context = getContext(polytope)
  store = getStore(polytope)

  L = length(indices)
  if L == 1
    Point(context, store, indices)
  elseif L == 2
    Segment(context, store, indices)
  elseif L == 3
    Triangle(context, store, indices)
  elseif L == 4
    Quadrangle(context, store, indices)
  else
    @notreachable
  end
end
