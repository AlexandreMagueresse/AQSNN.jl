################
# AbstractMesh #
################
abstract type AbstractMesh{T,M,E,P<:AbstractPolytope{T,M,E}} end

##############
# Parameters #
##############
eltype(::AbstractMesh{T}) where {T} = T
mandim(::AbstractMesh{T,M}) where {T,M} = M
embdim(::AbstractMesh{T,M,E}) where {T,M,E} = E

function getPolytope(::AbstractMesh)
  @abstractmethod
end

function getCells(::AbstractMesh)
  @abstractmethod
end

########
# Base #
########
function length(mesh::AbstractMesh)
  length(getCells(mesh))
end

function eachindex(mesh::AbstractMesh)
  eachindex(getCells(mesh))
end

function getindex(::AbstractMesh, ::Int)
  @abstractmethod
end

function firstindex(::AbstractMesh)
  @abstractmethod
end

function lastindex(::AbstractMesh)
  @abstractmethod
end

function iterate(mesh::AbstractMesh, i::Int=1)
  if i > length(mesh)
    nothing
  else
    (getindex(mesh, i), i + 1)
  end
end
