abstract type AbstractInitialiser{T} end

#################
# HeInitialiser #
#################
struct HeInitialiser{T} <: AbstractInitialiser{T}
end

function (init::HeInitialiser{T})(
  nIn::Int, nOut::Int, dims::NTuple{N,Int}
) where {T,N}
  u = Normal(zero(T), sqrt(2 / (nIn + nOut)))
  convert(Array{T}, rand(u, dims))
end

######################
# UniformInitialiser #
######################
struct UniformInitialiser{T} <: AbstractInitialiser{T}
  low::T
  high::T
end

function (init::UniformInitialiser{T})(
  ::Int, ::Int, dims::NTuple{N,Int}
) where {T,N}
  u = Uniform(init.low, init.high)
  convert(Array{T}, rand(u, dims))
end

###################
# ZeroInitialiser #
###################
struct ZeroInitialiser{T} <: AbstractInitialiser{T}
end

function (init::ZeroInitialiser{T})(
  ::Int, ::Int, dims::NTuple{N,Int}
) where {T,N}
  zeros(T, dims)
end
