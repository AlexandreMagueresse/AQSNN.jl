abstract type RegisteredFunction{T} end

∇⁰(::RegisteredFunction, _) = @abstractmethod

(ρ::RegisteredFunction)(x) = ∇⁰(ρ, x)

∇¹(::RegisteredFunction, _) = @abstractmethod

ϕ₋(::RegisteredFunction) = @abstractmethod

ϕ₊(::RegisteredFunction) = @abstractmethod

∫ρ(::RegisteredFunction, _, _) = @abstractmethod

∫ρx(::RegisteredFunction, _, _) = @abstractmethod

∫ρ²(::RegisteredFunction, _, _) = @abstractmethod

∫ρϕ₋²(::RegisteredFunction, _) = @abstractmethod

∫ρϕ₊²(::RegisteredFunction, _) = @abstractmethod

function ∫ρϕ²(ρ::RegisteredFunction, x₋, x₊, a, b)
  # ∫(ρ(x) - (ax + b))² dx between x₋ and x₊
  ∫ = ∫ρ²(ρ, x₋, x₊)

  ∫ -= 2 * a * ∫ρx(ρ, x₋, x₊)
  ∫ -= 2 * b * ∫ρ(ρ, x₋, x₊)

  ∫ += a^2 / 3 * (x₊^3 - x₋^3)
  ∫ += a * b * (x₊^2 - x₋^2)
  ∫ += b^2 * (x₊ - x₋)

  ∫
end
