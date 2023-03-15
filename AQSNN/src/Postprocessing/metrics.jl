######
# L1 #
######
function L1(
  ∇⁰f::Function, dΩ::AbstractQuadrature;
  init::Bool=true
)
  init && initialise!(dΩ)
  ∫Ω(x -> abs.(∇⁰f(x))) * dΩ
end

function L1(
  ∇⁰u::Sequential, ∇⁰û::Function,
  dΩ::AbstractQuadrature;
  relative::Bool=true
)
  initialise!(dΩ)
  L = L1(∇⁰u - ∇⁰û, dΩ, init=false)
  relative && (L /= L1(∇⁰û, dΩ, init=false))
  L
end

######
# L2 #
######
function L2(
  ∇⁰f::Function, dΩ::AbstractQuadrature;
  init::Bool=true,
  pow::Bool=true
)
  init && initialise!(dΩ)
  L = ∫Ω(∇⁰f^2) * dΩ
  pow && (L = sqrt(L))
  L
end

function L2(
  ∇⁰u::Sequential, ∇⁰û::Function,
  dΩ::AbstractQuadrature;
  relative::Bool=true
)
  initialise!(dΩ)
  L = L2(∇⁰u - ∇⁰û, dΩ, init=false, pow=false)
  relative && (L /= L2(∇⁰û, dΩ, init=false, pow=false))
  sqrt(L)
end

######
# L∞ #
######
function L∞(
  ∇⁰f::Function, dΩ::DomainMonteCarloQuadrature;
  init::Bool=true
)
  init && initialise!(dΩ)
  maximum(abs.(∇⁰f(getPoints(dΩ))))
end

function L∞(
  ∇⁰u::Sequential, ∇⁰û::Function,
  dΩ::DomainMonteCarloQuadrature;
  relative::Bool=true
)
  initialise!(dΩ)
  L = L∞(∇⁰u - ∇⁰û, dΩ, init=false)
  relative && (L /= L∞(∇⁰û, dΩ, init=false))
  L
end

######
# H1 #
######
function H1(
  ∇⁰f::Function, ∇¹f::Function,
  dΩ::AbstractQuadrature{T,D};
  init::Bool=true,
  pow::Bool=true
) where {T,D}
  init && initialise!(dΩ)
  L = L2(∇⁰f, dΩ, init=false, pow=false)
  for i in 1:D
    L += L2(x -> ∇¹f(x)[i:i, :], dΩ, init=false, pow=false)
  end
  pow && (L = sqrt(L))
  L
end

function H1(
  ∇⁰u::Sequential, ∇⁰û::Function, ∇¹û::Function,
  dΩ::AbstractQuadrature;
  relative::Bool=true
)
  ∇¹u = x -> ∇¹(∇⁰u, x)
  initialise!(dΩ)
  L = H1(∇⁰u - ∇⁰û, ∇¹u - ∇¹û, dΩ, init=false, pow=false)
  relative && (L /= H1(∇⁰û, ∇¹û, dΩ, init=false, pow=false))
  sqrt(L)
end

######
# H2 #
######
function H2(
  ∇⁰f::Function, ∇¹f::Function, ∇²f::Function,
  dΩ::AbstractQuadrature{T,D};
  init::Bool=true,
  pow::Bool=true
) where {T,D}
  init && initialise!(dΩ)
  L = H1(∇⁰f, ∇¹f, dΩ, init=false, pow=false)
  for i in 1:D
    for j in 1:D
      L += L2(x -> ∇²f(x)[i, j, :], dΩ, init=false, pow=false)
    end
  end
  pow && (L = sqrt(L))
  L
end

function H2(
  ∇⁰u::Sequential, ∇⁰û::Function, ∇¹û::Function, ∇²û::Function,
  dΩ::AbstractQuadrature;
  relative::Bool=true
)
  ∇¹u = x -> ∇¹(∇⁰u, x)
  ∇²u = x -> ∇²(∇⁰u, x)
  initialise!(dΩ)
  L = H2(∇⁰u - ∇⁰û, ∇¹u - ∇¹û, ∇²u - ∇²û, dΩ, init=false, pow=false)
  relative && (L /= H2(∇⁰û, ∇¹û, ∇²û, dΩ, init=false, pow=false))
  sqrt(L)
end
