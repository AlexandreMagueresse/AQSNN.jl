struct Tanh{T} <: RegisteredFunction{T}
end

∇⁰(::Tanh, x) = tanh(x)

∇¹(::Tanh, x) = 1 - tanh(x)^2

ϕ₋(::Tanh{T}) where {T<:Real} = (T(0), T(-1))

ϕ₊(::Tanh{T}) where {T<:Real} = (T(0), T(+1))

function ∫ρ(::Tanh, x, y)
  log(cosh(y) / cosh(x))
end

function ∫ρx(::Tanh, x, y)
  ∫ = y * log(2 * cosh(y)) - x * log(2 * cosh(x))
  ∫ -= (y^2 - x^2) / 2
  ∫ -= (ReLi₂(-exp(-2 * y)) - ReLi₂(-exp(-2 * x))) / 2
  ∫
end

function ∫ρ²(::Tanh, x, y)
  (y - x) - (tanh(y) - tanh(x))
end

function ∫ρϕ₋²(::Tanh, x)
  2 * log(1 + exp(2 * x)) - (1 + tanh(x))
end

function ∫ρϕ₊²(::Tanh, x)
  2 * log(1 + exp(-2 * x)) - (1 - tanh(x))
end
