using AQSNN
using Random
using Statistics

using Plots
using LaTeXStrings

include("0.utils.jl")
colorBlue = RGB(0.1216, 0.4667, 0.7059)

mode = "l2" # "abs", "rel"

############
# Settings #
############
# Problem
problem = "poissonStrong"
Ωname = "cartesian1d"
Fname = "sinc"

# Model
ρname = ρsfs[Fname]
A = (10, 10)

# Optimisation
η = 1.0f-2
ν = 10
NE = 5000
γ = 1
β = -1
seed = 1

# Monte-Carlo
MCs = [
  # (50, 1),
  (100, 1),
  # (500, 1)
]

# Adaptive
AQs = [
  # (2, 2, 0.0f0, 3),
  (2, 5, 0.0f0, 3),
  # (2, 10, 0.0f0, 3),
  # (3, 2, 0.0f0, 3),
  (3, 5, 0.0f0, 3),
  # (3, 10, 0.0f0, 3),
  # (5, 2, 0.0f0, 3),
  (5, 5, 0.0f0, 3),
  # (5, 10, 0.0f0, 3),
  # (7, 2, 0.0f0, 3),
  # (7, 5, 0.0f0, 3),
  # (7, 10, 0.0f0, 3),
]

############
# Geometry #
############
Ω = Ωs[Ωname]

Ωτ = simplexify(Ω)
Ωκ = convexify(Ω)

Γ = Boundary(Ω)
Γτ = simplexify.(Γ)
Γκ = convexify.(Γ)

T = Float32
E = embdim(Ω)

##################
# Default values #
##################
γ = T(γ)

η, γ, NE, β, ν, _, _, _, _, _, _ = getDefaults(
  T, E, problem, ρname, A,
  η, γ, NE, β, ν, -1, -1, -1, -1, -1, -1
)

###########
# Problem #
###########
∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]
ρ = ρs[ρname]
Afull = (E, A..., 1)

objective = objectives[problem](∇⁰f, ∇¹f, ∇²f, Δf)

##################
# L2 / abs / rel #
##################
Plots.plot(layout=(1, 1))

X, F = nothing, nothing
if mode != "l2"
  X = [-1:0.01:+1;;]'
  F = ∇⁰f(X)
end

for (NΩ, NΓ) in MCs
  folder, filename = expName(
    Ωname, problem, Fname, ρname,
    η, NE, β, ν, seed, "MC";
    NΩ=NΩ, NΓ=NΓ
  )
  folder = split(folder, "/")
  insert!(folder, 2, "exploration")
  folder = join(folder, "/")

  u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
  NΩ, NΓ = dict[:NΩ], dict[:NΓ]

  if mode == "l2"
    X = [el[1] for el in dict[:histL2]]
    Y = [el[2] for el in dict[:histL2]]
    Plots.plot!(X, Y, yscale=:log10, label="MC ($NΩ)")
  elseif mode == "abs"
    U = u(X)
    Plots.plot!(X[1, :], abs.(U[1, :] .- F[1, :]), label="MC ($NΩ)")
  elseif mode == "rel"
    U = u(X)
    Plots.plot!(X[1, :], U[1, :], label="MC ($NΩ)")
  end
end

for (P, O, α, δ) in AQs
  folder, filename = expName(
    Ωname, problem, Fname, ρname,
    η, NE, β, ν, seed, "AQ";
    P=P, O=O, α=α, δ=δ
  )
  folder = split(folder, "/")
  insert!(folder, 2, "exploration")
  folder = join(folder, "/")

  u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
  NΩ, NΓ = trunc(Int, mean(dict[:NΩ])), sum([trunc(Int, mean(Nγ)) for Nγ in dict[:NΓ]])

  if mode == "l2"
    X = [el[1] for el in dict[:histL2]]
    Y = [el[2] for el in dict[:histL2]]
    Plots.plot!(X, Y, yscale=:log10, label="AQ ($NΩ, $P, $O)")
  elseif mode == "abs"
    U = u(X)
    Plots.plot!(X[1, :], abs.(U[1, :] .- F[1, :]), label="AQ ($NΩ, $P, $O)")
  elseif mode == "rel"
    U = u(X)
    Plots.plot!(X[1, :], U[1, :], label="AQ ($NΩ, $P, $O)")
  end
end

if mode == "l2"
  Plots.plot!(xlabel="Epoch", ylabel=L"\log \ \|\!\!\|u - \hat{u}\,\|\!\!\|_{L^2(\Omega)}")
  display(Plots.plot!(legend=:topright))
  Plots.savefig("l2.pdf")
elseif mode == "abs"
  Plots.plot!(xlabel="X", ylabel=L"\|u - \hat{u}\,\|")
  display(Plots.plot!(legend=:topright))
  Plots.savefig("abs.pdf")
elseif mode == "rel"
  Plots.plot!(X[1, :], F[1, :], label="true")
  Plots.plot!(xlabel="X", ylabel=L"u")
  display(Plots.plot!(legend=:topright))
  Plots.savefig("rel.pdf")
end

########
# Mesh #
########
# P = 0
# α, δ = 1.0f0, 2

# Ωlin = Lineariser(Ωκ, ρ, P, depth=δ)
# linearise!(u, Ωlin)

# Collapse + simplexify
# dΩ = AdaptiveQuadrature(u, Ωlin, O, α, false)
# println(length(dΩ.points))

# Only collapse
# collapse!(Ωlin, α)
# println(length(Ωlin))

# plot(layout=(1, 1))
# for cycle in Ωlin.cycles
#   for i in eachindex(cycle)
#     j = (i == length(cycle)) ? 1 : i + 1
#     p₋, p₊ = Ωlin.points[cycle[i]], Ωlin.points[cycle[j]]
#     plot!([p₋[1], p₊[1]], [p₋[2], p₊[2]], c=colorBlue)
#   end
# end
# display(plot!(legend=false))
# savefig("mesh_$(α)_$(δ)_$(dΩ.points).pdf")
