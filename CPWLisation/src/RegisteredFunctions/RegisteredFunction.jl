abstract type RegisteredFunction{T} end

"""
Return the value of ρ at x.
"""
∇⁰(ρ::RegisteredFunction, x) = @abstractmethod

(ρ::RegisteredFunction)(x) = ∇⁰(ρ, x)

"""
Return the value of ρ' at x.
"""
∇¹(ρ::RegisteredFunction, x) = @abstractmethod

"""
Return the value of the tangent to ρ at -∞ at x.
"""
ϕ₋(ρ::RegisteredFunction) = @abstractmethod

"""
Return the value of the tangent to ρ at +∞ at x.
"""
ϕ₊(ρ::RegisteredFunction) = @abstractmethod

"""
Return the integral of x ⟼ ρ(x) between x₋ and x₊.
"""
∫ρ(::RegisteredFunction, x₋, x₊) = @abstractmethod

"""
Return the integral of x ⟼ ρ(x) * x between x₋ and x₊.
"""
∫ρx(::RegisteredFunction, x₋, x₊) = @abstractmethod

"""
Return the integral of x ⟼ ρ(x)^2 between x₋ and x₊.
"""
∫ρ²(::RegisteredFunction, x₋, x₊) = @abstractmethod

"""
Return the integral of x ⟼ (ρ(x) - ϕ₋(x))^2 between -∞ and x₊, where ϕ₋ is the
tangent to ρ at -∞.
"""
∫ρϕ₋²(ρ::RegisteredFunction, x₊) = @abstractmethod

"""
Return the integral of x ⟼ (ρ(x) - ϕ₊(x))^2 between x₋ and +∞, where ϕ₊ is the
tangent to ρ at +∞.
"""
∫ρϕ₊²(ρ::RegisteredFunction, x₋) = @abstractmethod

"""
Return the integral of x ⟼ (ρ(x) - a * x - b)^2 between x₋ and x₊.
"""
function ∫ρϕ²(ρ::RegisteredFunction, x₋, x₊, a, b)
  ∫ = ∫ρ²(ρ, x₋, x₊)

  ∫ -= 2 * a * ∫ρx(ρ, x₋, x₊)
  ∫ -= 2 * b * ∫ρ(ρ, x₋, x₊)

  ∫ += a^2 / 3 * (x₊^3 - x₋^3)
  ∫ += a * b * (x₊^2 - x₋^2)
  ∫ += b^2 * (x₊ - x₋)

  ∫
end
