abstract type AbstractIntegrand end

struct DomainIntegrand{F<:Function} <: AbstractIntegrand
  f::F
end

struct BoundaryIntegrand{F<:Function} <: AbstractIntegrand
  f::F
end

∫Ω(f::Function) = DomainIntegrand(f)
∫Γ(f::Function) = BoundaryIntegrand(f)

##############
# Operations #
##############
+(f::Function, g::Function) = (args...) -> f(args...) .+ g(args...)
-(f::Function, g::Function) = (args...) -> f(args...) .- g(args...)
*(f::Function, g::Function) = (args...) -> f(args...) .* g(args...)
^(f::Function, α::Real) = (args...) -> f(args...) .^ α
LinearAlgebra.dot(f::Function, g::Function) = (args...) -> sum(f(args...) .* g(args...), dims=1)
norm2(f::Function) = (args...) -> sum(f(args...) .^ 2, dims=1)
