abstract type AbstractPolytope{T,M,E} end

##############
# Parameters #
##############
eltype(::AbstractPolytope{T}) where {T} = T
length(polytope::AbstractPolytope) = length(getIndices(polytope))

mandim(::AbstractPolytope{T,M}) where {T,M} = M
embdim(::AbstractPolytope{T,M,E}) where {T,M,E} = E

####################
# Internal getters #
####################
function getStore(::AbstractPolytope)
  @abstractmethod
end

function getIndices(::AbstractPolytope)
  @abstractmethod
end

####################
# External getters #
####################
function getVertex(polytope::AbstractPolytope, i::Int)
  getindex(getStore(polytope), getindex(getIndices(polytope), i))
end

function getEdgeEnd(polytope::AbstractPolytope, i::Int)
  if i == length(polytope)
    1
  else
    i + 1
  end
end

###################
# AbstractContext #
###################
function getContext(::AbstractPolytope)
  @abstractmethod
end

getBasis(polytope::AbstractPolytope) = getBasis(getContext(polytope))
getNormal(polytope::AbstractPolytope) = getNormal(getContext(polytope))

##########
# Traits #
##########
isSimplex(polytope::AbstractPolytope) = isSimplex(typeof(polytope))
isSimplex(::Type{<:AbstractPolytope}) = false

isReference(polytope::AbstractPolytope) = isReference(typeof(polytope))
isReference(::Type{<:AbstractPolytope}) = false

isConvex(polytope::AbstractPolytope) = isConvex(typeof(polytope))
isConvex(::Type{<:AbstractPolytope}) = false
