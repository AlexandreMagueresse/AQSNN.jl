abstract type AbstractContext{T,M,E} end

##############
# Parameters #
##############
eltype(::AbstractContext{T}) where {T} = T
mandim(::AbstractContext{T,M}) where {T,M} = M
embdim(::AbstractContext{T,M,E}) where {T,M,E} = E

function getBasis(::AbstractContext)
  @abstractmethod
end

function getNormal(::AbstractContext)
  @abstractmethod
end

###############
# Dimension 0 #
###############
struct Context0D{T,E,N} <: AbstractContext{T,0,E}
  n::N

  function Context0D{T,E}(n::N) where {T,E,N}
    new{T,E,N}(n)
  end
end

getBasis(context::Context0D) = ()
getNormal(context::Context0D) = context.n

###############
# Dimension 1 #
###############
struct Context1D{T,E,U,N} <: AbstractContext{T,1,E}
  u::U
  n::N

  function Context1D{T,E}(u::U, n::N) where {T,E,U,N}
    new{T,E,U,N}(u, n)
  end
end

getBasis(context::Context1D) = (context.u,)
getNormal(context::Context1D) = context.n

###############
# Dimension 2 #
###############
struct Context2D{T,E,U,V,N} <: AbstractContext{T,2,E}
  u::U
  v::V
  n::N

  function Context2D{T,E}(u::U, v::V, n::N) where {T,E,U,V,N}
    new{T,E,U,V,N}(u, v, n)
  end
end

getBasis(context::Context2D) = (context.u, context.v)
getNormal(context::Context2D) = context.n
