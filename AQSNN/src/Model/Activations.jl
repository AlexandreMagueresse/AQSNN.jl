# tanh_hard, arctan, exp
# ∇arctan, cos², bellᵏ, splineᵏ
abstract type AbstractActivation <: Function end

abstract type AbstractReLU <: AbstractActivation end
abstract type AbstractSigmoid <: AbstractActivation end

############
# Identity #
############
struct Identity <: AbstractActivation
end

function ∇⁰identity(x)
  x
end

function ∇¹identity(x)
  one(x)
end

function ∇²identity(x)
  zero(x)
end

∇⁰(::Identity, x) = ∇⁰identity(x)
∇¹(::Identity, x) = ∇¹identity(x)
∇²(::Identity, x) = ∇²identity(x)

(f::Identity)(x) = ∇⁰(f, x)
∇(f::Identity, x) = ∇¹(f, x)

@primitive ∇⁰identity(x), dy, y (dy .* ∇¹identity.(x))
@primitive ∇¹identity(x), dy, y (dy .* ∇²identity.(x))

########
# ReLU #
########
struct ReLU <: AbstractReLU
end

function ∇⁰relu(x)
  ifelse(x > 0, x, zero(x))
end

function ∇¹relu(x)
  ifelse(x > 0, one(x), zero(x))
end

function ∇²relu(x)
  zero(x)
end

∇⁰(::ReLU, x) = ∇⁰relu(x)
∇¹(::ReLU, x) = ∇¹relu(x)
∇²(::ReLU, x) = ∇²relu(x)

(f::ReLU)(x) = ∇⁰(f, x)
∇(f::ReLU, x) = ∇¹(f, x)

@primitive ∇⁰relu(x), dy, y (dy .* ∇¹relu.(x))
@primitive ∇¹relu(x), dy, y (dy .* ∇²relu.(x))

########
# absε #
########
struct Absε{T} <: AbstractReLU
  εp::T
  ε::T

  function Absε(ε::T) where {T<:Real}
    new{T}(ε, 2 * ε)
  end
end

function ∇⁰absε(ε, x)
  h = x / ε
  (x + ε * sqrt(1 + h^2)) / 2
end

function ∇¹absε(ε, x)
  h = x / ε
  (1 + h / sqrt(1 + h^2)) / 2
end

function ∇²absε(ε, x)
  h = x / ε
  1 / (1 + h^2)^(3 / 2) / 2 / ε
end

∇⁰(f::Absε, x) = ∇⁰absε(f.ε, x)
∇¹(f::Absε, x) = ∇¹absε(f.ε, x)
∇²(f::Absε, x) = ∇²absε(f.ε, x)

(f::Absε)(x) = ∇⁰(f, x)
∇(f::Absε, x) = ∇¹(f, x)

@primitive ∇⁰absε(ε, x), dy, y zero.(dy) (dy .* ∇¹absε.(ε, x))
@primitive ∇¹absε(ε, x), dy, y zero.(dy) (dy .* ∇²absε.(ε, x))

########
# logε #
########
struct Logε{T} <: AbstractReLU
  εp::T
  ε::T

  function Logε(ε::T) where {T<:Real}
    new{T}(ε, ε / log(2))
  end
end

function ∇⁰logε(ε, x)
  x̃ = x / ε
  ∇⁰relu(x) + ε * log(1 + exp(-abs(x̃)))
end

function ∇¹logε(ε, x)
  x̃ = x / 2 / ε
  (1 + tanh(x̃)) / 2
end

function ∇²logε(ε, x)
  x̃ = x / 2 / ε
  (1 - tanh(x̃)^2) / 4 / ε
end

∇⁰(f::Logε, x) = ∇⁰logε(f.ε, x)
∇¹(f::Logε, x) = ∇¹logε(f.ε, x)
∇²(f::Logε, x) = ∇²logε(f.ε, x)

(f::Logε)(x) = ∇⁰(f, x)
∇(f::Logε, x) = ∇¹(f, x)

@primitive ∇⁰logε(ε, x), dy, y zero.(dy) (dy .* ∇¹logε.(ε, x))
@primitive ∇¹logε(ε, x), dy, y zero.(dy) (dy .* ∇²logε.(ε, x))

########
# erfε #
########
struct Erfε{T} <: AbstractReLU
  εp::T
  ε::T

  function Erfε(ε::T) where {T<:Real}
    new{T}(ε, 2 * ε * sqrt(pi))
  end
end

function ∇⁰erfε(ε, x)
  x̃ = x / ε
  (x + x * erf(x̃) + ε / sqrt(pi) * exp(-x̃^2)) / 2
end

function ∇¹erfε(ε, x)
  x̃ = x / ε
  (1 + erf(x̃)) / 2
end

function ∇²erfε(ε, x)
  x̃ = x / ε
  exp(-x̃^2) / ε / sqrt(pi)
end

∇⁰(f::Erfε, x) = ∇⁰erfε(f.ε, x)
∇¹(f::Erfε, x) = ∇¹erfε(f.ε, x)
∇²(f::Erfε, x) = ∇²erfε(f.ε, x)

(f::Erfε)(x) = ∇⁰(f, x)
∇(f::Erfε, x) = ∇¹(f, x)

@primitive ∇⁰erfε(ε, x), dy, y zero.(dy) (dy .* ∇¹erfε.(ε, x))
@primitive ∇¹erfε(ε, x), dy, y zero.(dy) (dy .* ∇²erfε.(ε, x))

############
# spline²ε #
############
struct Spline²ε{T} <: AbstractReLU
  εp::T
  ε::T

  function Spline²ε(ε::T) where {T<:Real}
    new{T}(ε, ε)
  end
end

function ∇⁰spline²ε(ε, x)
  if x < -ε
    zero(x)
  elseif x > ε
    x
  else
    (x + ε)^2 / 4 / ε
  end
end

function ∇¹spline²ε(ε, x)
  if x < -ε
    zero(x)
  elseif x > ε
    one(x)
  else
    (x + ε) / 2 / ε
  end
end

function ∇²spline²ε(ε, x)
  if abs(x) > ε
    zero(x)
  else
    inv(ε) / 2
  end
end

∇⁰(f::Spline²ε, x) = ∇⁰spline²ε(f.ε, x)
∇¹(f::Spline²ε, x) = ∇¹spline²ε(f.ε, x)
∇²(f::Spline²ε, x) = ∇²spline²ε(f.ε, x)

(f::Spline²ε)(x) = ∇⁰(f, x)
∇(f::Spline²ε, x) = ∇¹(f, x)

@primitive ∇⁰spline²ε(ε, x), dy, y zero.(dy) (dy .* ∇¹spline²ε.(ε, x))
@primitive ∇¹spline²ε(ε, x), dy, y zero.(dy) (dy .* ∇²spline²ε.(ε, x))

############
# spline³ε #
############
struct Spline³ε{T} <: AbstractReLU
  εp::T
  ε::T

  function Spline³ε(ε::T) where {T<:Real}
    new{T}(ε, ε)
  end
end

function ∇⁰spline³ε(ε, x)
  if x < -ε
    zero(x)
  elseif x > ε
    x
  else
    ((x + ε)^3 - 2 * x^2 * ∇⁰relu(x)) / 6 / ε^2
  end
end

function ∇¹spline³ε(ε, x)
  if x < -ε
    zero(x)
  elseif x > ε
    one(x)
  else
    ((x + ε)^2 - 2 * x * ∇⁰relu(x)) / 2 / ε^2
  end
end

function ∇²spline³ε(ε, x)
  if abs(x) > ε
    zero(x)
  else
    ((x + ε) - 2 * ∇⁰relu(x)) / ε^2
  end
end

∇⁰(f::Spline³ε, x) = ∇⁰spline³ε(f.ε, x)
∇¹(f::Spline³ε, x) = ∇¹spline³ε(f.ε, x)
∇²(f::Spline³ε, x) = ∇²spline³ε(f.ε, x)

(f::Spline³ε)(x) = ∇⁰(f, x)
∇(f::Spline³ε, x) = ∇¹(f, x)

@primitive ∇⁰spline³ε(ε, x), dy, y zero.(dy) (dy .* ∇¹spline³ε.(ε, x))
@primitive ∇¹spline³ε(ε, x), dy, y zero.(dy) (dy .* ∇²spline³ε.(ε, x))

########
# tanh #
########
struct Tanh <: AbstractSigmoid
end

function ∇⁰tanh(x)
  tanh(x)
end

function ∇¹tanh(x)
  1 - tanh(x)^2
end

function ∇²tanh(x)
  t = tanh(x)
  -2 * (t - t^3)
end

∇⁰(::Tanh, x) = ∇⁰tanh(x)
∇¹(::Tanh, x) = ∇¹tanh(x)
∇²(::Tanh, x) = ∇²tanh(x)

(f::Tanh)(x) = ∇⁰(f, x)
∇(f::Tanh, x) = ∇¹(f, x)

@primitive ∇⁰tanh(x), dy, y (dy .* ∇¹tanh.(x))
@primitive ∇¹tanh(x), dy, y (dy .* ∇²tanh.(x))
