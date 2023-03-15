##########
# Struct #
##########
struct Objective
  a::Function
  l::Function
end

function Objective(D::Function, f::Function, g::Function)
  a(u, v, β, dΩ, dΓ) = ∫Ω(D(u) * D(v)) * dΩ + β * (∫Γ((n, x) -> u(x) .* v(x)) * dΓ)
  l(v, β, dΩ, dΓ) = ∫Ω(D(v) * f) * dΩ + β * (∫Γ((n, x) -> v(x) .* g(x)) * dΓ)
  Objective(a, l)
end

function linear(objective::Objective, v::Sequential, β, dΩ, dΓ)
  objective.l(v, β, dΩ, dΓ)
end

function bilinear(objective::Objective, u::Sequential, v::Sequential, β, dΩ, dΓ)
  objective.a(u, v, β, dΩ, dΓ)
end

function residual(objective::Objective, u::Sequential, β, dΩ, dΓ)
  bilinear(objective, u, u, β, dΩ, dΓ) / 2 - linear(objective, u, β, dΩ, dΓ)
end
