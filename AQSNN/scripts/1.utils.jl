using GeoInterface
using StaticArrays
using Statistics

###########
# Domains #
###########
Ω₁ = CartesianDomain(-1.0f0, +1.0f0)
Ω₂ = CartesianDomain(-1.0f0, +1.0f0, -1.0f0, +1.0f0)

SV2 = SVector{2,Float32}
Ω₃ = Domain(Val(2), [
  SV2(-1.0, +0.0), SV2(-0.5, +1.0), SV2(+0.0, +0.5), SV2(+0.5, +1.0),
  SV2(+1.0, +0.0), SV2(+0.5, -1.0), SV2(+0.0, -0.5), SV2(-0.5, -1.0)
])

Ωs = Dict(
  "cartesian1d" => Ω₁,
  "cartesian2d" => Ω₂,
  "rhombi" => Ω₃
)

###############
# Activations #
###############
ρs = Dict(
  "ReLUε" => Absε(1.0f-1),
  "tanh" => Tanh(),
)

#################
# Forcing terms #
#################
# xpy
function ∇⁰xpy_1(X)
  zero.(X)
end

function ∇¹xpy_1(X)
  zero.(X)
end

function ∇²xpy_1(X)
  T = eltype(X)
  N = size(X, 2)
  ∇²F = zeros(T, 1, 1, N)
  ∇²F
end

function Δxpy_1(X)
  -X
end

function ∇⁰xpy_2(XY)
  X = XY[1:1, :]
  Y = XY[2:2, :]
  zero.(X)
end

function ∇¹xpy_2(XY)
  X = XY[1:1, :]
  Y = XY[2:2, :]
  [
    zero.(X)
    zero.(Y)
  ]
end

function ∇²xpy_2(XY)
  X = XY[1:1, :]
  T = eltype(X)
  N = size(X, 2)
  ∇²F = zeros(T, 2, 2, N)
  ∇²F
end

function Δxpy_2(XY)
  X = XY[1:1, :]
  Y = XY[2:2, :]
  -(X .+ Y)
end

# tanh(α (r^2 - 1/4))
function ∇⁰well_1(X)
  α = 10
  T = eltype(X)
  tanh.(α .* (X .^ 2 .- T(0.5)^2))
end

function ∇¹well_1(X)
  α = 10
  2 .* α .* X .* (1 .- ∇⁰well_1(X) .^ 2)
end

function ∇²well_1(X)
  α = 10
  T = eltype(X)
  N = size(X, 2)
  ∇²F = zeros(T, 1, 1, N)
  ∇²F[1, 1, :] = 2 .* α * (1 .- 4 .* α .* X .^ 2 .* ∇⁰well_1(X)) .* (1 .- ∇⁰well_1(X) .^ 2)
  ∇²F
end

function Δwell_1(X)
  α = 10
  2 .* α * (1 .- 4 .* α .* X .^ 2 .* ∇⁰well_1(X)) .* (1 .- ∇⁰well_1(X) .^ 2)
end

function ∇⁰well_2(XY)
  α = 10
  T = eltype(XY)
  X = XY[1:1, :]
  Y = XY[2:2, :]
  tanh.(α .* (X .^ 2 .+ Y .^ 2 .- T(0.5)^2))
end

function ∇¹well_2(XY)
  α = 10
  X = XY[1:1, :]
  Y = XY[2:2, :]
  [
    2 .* α .* X .* (1 .- ∇⁰well_2(XY) .^ 2)
    2 .* α .* Y .* (1 .- ∇⁰well_2(XY) .^ 2)
  ]
end

function ∇²well_2(XY)
  α = 10
  X = XY[1:1, :]
  Y = XY[2:2, :]
  T = eltype(X)
  N = size(X, 2)
  ∇²F = zeros(T, 2, 2, N)
  ∇²F[1, 1, :] = 2 .* α * (1 .- 4 .* α .* X .^ 2 .* ∇⁰well_2(XY)) .* (1 .- ∇⁰well_2(XY) .^ 2)
  ∇²F[1, 2, :] = -8 .* α^2 .* X .* Y .* ∇⁰well_2(XY) .* (1 .- ∇⁰well_2(XY) .^ 2)
  ∇²F[2, 1, :] = ∇²F[1, 2, :]
  ∇²F[2, 2, :] = 2 .* α * (1 .- 4 .* α .* Y .^ 2 .* ∇⁰well_2(XY)) .* (1 .- ∇⁰well_2(XY) .^ 2)
  ∇²F
end

function Δwell_2(XY)
  α = 10
  X = XY[1:1, :]
  Y = XY[2:2, :]
  4 .* α * (1 .- 2 .* α .* (X .^ 2 .+ Y .^ 2) .* ∇⁰well_2(XY)) .* (1 .- ∇⁰well_2(XY) .^ 2)
end

# sinc(απx)
function ∂¹sinc(α, x)
  ifelse(x == 0,
    zero(x),
    (cospi(α * x) - sinc(α * x)) / x
  )
end

function ∂²sinc(α, x::T) where {T}
  ifelse(x == 0,
    T(-(α * pi)^2 / 3),
    (-α * pi * x * sinpi(α * x) - 2 * cospi(α * x) + 2 * sinc(α * x)) / x^2
  )
end

function ∇⁰sinc_1(X)
  α = 3
  sinc.(α .* X)
end

function ∇¹sinc_1(X)
  α = 3
  ∂¹sinc.(α, X)
end

function ∇²sinc_1(X)
  α = 3
  T = eltype(X)
  N = size(X, 2)
  ∇²F = zeros(T, 1, 1, N)
  ∇²F[1, 1, :] = ∂²sinc.(α, X)
  ∇²F
end

function Δsinc_1(X)
  α = 3
  ∂²sinc.(α, X)
end

function ∇⁰sinc_2(XY)
  α = 2
  X = XY[1:1, :]
  Y = XY[2:2, :]
  sinc.(α .* X) .* sinc.(α .* Y)
end

function ∇¹sinc_2(XY)
  α = 2
  X = XY[1:1, :]
  Y = XY[2:2, :]
  [
    ∂¹sinc.(α, X) .* sinc.(α .* Y)
    sinc.(α .* X) .* ∂¹sinc.(α, Y)
  ]
end

function ∇²sinc_2(XY)
  α = 2
  X = XY[1:1, :]
  Y = XY[2:2, :]
  T = eltype(X)
  N = size(X, 2)
  ∇²F = zeros(T, 2, 2, N)
  ∇²F[1, 1, :] = ∂²sinc.(α, X) .* sinc.(α .* Y)
  ∇²F[1, 2, :] = ∂¹sinc.(α, X) .* ∂¹sinc.(α, Y)
  ∇²F[2, 1, :] = ∇²F[1, 2, :]
  ∇²F[2, 2, :] = sinc.(α .* X) .* ∂²sinc.(α, Y)
  ∇²F
end

function Δsinc_2(XY)
  α = 2
  X = XY[1:1, :]
  Y = XY[2:2, :]
  ∂²sinc.(α, X) .* sinc.(α .* Y) .+ sinc.(α .* X) .* ∂²sinc.(α, Y)
end

fs = Dict(
  "xpy" => Dict(
    1 => [∇⁰xpy_1, ∇¹xpy_1, ∇²xpy_1, Δxpy_1],
    2 => [∇⁰xpy_2, ∇¹xpy_2, ∇²xpy_2, Δxpy_2]
  ),
  "well" => Dict(
    1 => [∇⁰well_1, ∇¹well_1, ∇²well_1, Δwell_1],
    2 => [∇⁰well_2, ∇¹well_2, ∇²well_2, Δwell_2]
  ),
  "sinc" => Dict(
    1 => [∇⁰sinc_1, ∇¹sinc_1, ∇²sinc_1, Δsinc_1],
    2 => [∇⁰sinc_2, ∇¹sinc_2, ∇²sinc_2, Δsinc_2]
  )
)

ρsfs = Dict(
  "well" => "tanh",
  "sinc" => "ReLUε",
  "xpy" => "ReLUε",
)

##############
# Objectives #
##############
function makeInterpolation(∇⁰f, ∇¹f, ∇²f, Δf)
  Objective(∇⁰, ∇⁰f, ∇⁰f)
end

function makePoissonStrong(∇⁰f, ∇¹f, ∇²f, Δf)
  Objective(AQSNN.Δ, Δf, ∇⁰f)
end

function makePoissonNitsche(∇⁰f, ∇¹f, ∇²f, Δf)
  a(u, v, β, dΩ, dΓ) = ∫Ω(dot(∇(u), ∇(v))) * dΩ - ∫Γ((n, x) -> ∂ₙ(u, n, x) .* v(x)) * dΓ - ∫Γ((n, x) -> ∂ₙ(v, n, x) .* u(x)) * dΓ + β * (∫Γ((n, x) -> u(x) .* v(x)) * dΓ)
  l(v, β, dΩ, dΓ) = -(∫Ω(v * Δf) * dΩ) - ∫Γ((n, x) -> ∂ₙ(v, n, x) .* ∇⁰f(x)) * dΓ + β * (∫Γ((n, x) -> ∇⁰f(x) .* v(x)) * dΓ)
  Objective(a, l)
end

function makePoissonBroken(∇⁰f, ∇¹f, ∇²f, Δf)
  a(u, v, β, dΩ, dΓ) = ∫Ω(dot(∇(u), ∇(v))) * dΩ + β * (∫Γ((n, x) -> u(x) .* v(x)) * dΓ)
  l(v, β, dΩ, dΓ) = -(∫Ω(v * Δf) * dΩ) + β * (∫Γ((n, x) -> ∇⁰f(x) .* v(x)) * dΓ)
  Objective(a, l)
end

objectives = Dict(
  "interpolation" => makeInterpolation,
  "poissonStrong" => makePoissonStrong,
  "poissonNitsche" => makePoissonNitsche,
  "poissonBroken" => makePoissonBroken,
)

##################
# Default values #
##################
function getDefaults(
  T, E, problem, ρname, A,
  η, γ, NE, β, ν, NΩ, NΓ, P, O, α, δ
)
  if η == -1
    η = 1.0f-2
  end

  if γ == -1
    γ = T(1)
  end

  if NE == -1
    NE = 5000
  end

  if β == -1
    if problem == "interpolation"
      β = 1.0f0
    else
      β = 1.0f2
    end
  end

  if ν == -1
    ν = 10
  end

  if NΩ == -1
    NΩ = (E == 1) ? 1000 : 10000
  end

  if NΓ == -1
    NΓ = (E == 1) ? 2 : 100
  end

  if P == -1
    if ρname == "ReLUε"
      P = 3
    elseif ρname == "tanh"
      P = 5
    end
  end

  if O == -1
    O = 5
  end

  if α == -1
    α = T(0)
  end

  if δ == -1
    δ = length(A) + 1
  end

  η, γ, NE, β, ν, NΩ, NΓ, P, O, α, δ
end

#####################
# Naming convention #
#####################
function expName(
  domain, problem, forcingTerm, activation,
  learningRate, epochs, penalty, frequency, seed, integration;
  kwargs...
)
  folder = joinpath("data", get(kwargs, :folder, ""))
  folder = joinpath(folder, join((domain, problem, forcingTerm, activation), "_"))
  folder = joinpath(folder, integration)

  file = join((learningRate, epochs, penalty, frequency), "_")
  if integration == "MC"
    file = join((file, string(kwargs[:NΩ]), string(kwargs[:NΓ])), "_")
  elseif integration == "AQ"
    file = join((file, string(kwargs[:P]), string(kwargs[:O]), string(kwargs[:α]), string(kwargs[:δ])), "_")
  end
  file = join((file, string(seed)), "_")

  folder, file * ".jld2"
end
