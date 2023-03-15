struct ReLU{T} <: RegisteredFunction{T}
  ε::T
  γ²::T

  function ReLU(ε::T) where {T}
    new{T}(ε, (2 * ε)^2)
  end
end

function ∇⁰(ρ::ReLU, x)
  (x + sqrt(x^2 + ρ.γ²)) / 2
end

function ∇¹(ρ::ReLU, x)
  (1 + x / sqrt(x^2 + ρ.γ²)) / 2
end

ϕ₋(::ReLU{T}) where {T<:Real} = (T(0), T(0))

ϕ₊(::ReLU{T}) where {T<:Real} = (T(1), T(0))

function ∫ρ(ρ::ReLU, x, y)
  ρy, ρx = ρ(y), ρ(x)
  ∫ = (y * ρy - x * ρx) / 2
  ∫ += ρ.γ² * log(ρy / ρx) / 4
  ∫
end

function ∫ρx(ρ::ReLU, x, y)
  ∫ = sqrt(ρ.γ² + y^2)^3 - sqrt(ρ.γ² + x^2)^3
  ∫ += y^3 - x^3
  ∫ / 6
end

function ∫ρ²(ρ::ReLU, x, y)
  ρy, ρx = ρ(y), ρ(x)
  ∫ = ρy * (y * ρy + ρ.γ²)
  ∫ -= ρx * (x * ρx + ρ.γ²)
  ∫ / 3
end

function ∫ρϕ₋²(ρ::ReLU, x)
  ρx = ρ(x)
  ∫ = ρx * (x * ρx + ρ.γ²)
  ∫ / 3
end

function ∫ρϕ₊²(ρ::ReLU, x)
  ρx = ρ(x) - x
  ∫ = ρx * (x * ρx - ρ.γ²)
  -∫ / 3
end
