using AQSNN
using Random
using Statistics

include("0.utils.jl")

############
# Settings #
############
# Problem
problem = "poissonNitsche"
Ωname = "cartesian2d"
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

# Integration
integration = "AQ"

# # Monte-Carlo
NΩ = 10000
NΓ = 1000

# # Adaptive
P = 3
O = 2
α = 0.0
δ = -1

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
α = T(α)

η, γ, NE, β, ν, NΩ, NΓ, P, O, α, δ = getDefaults(
  T, E, problem, ρname, A,
  η, γ, NE, β, ν, NΩ, NΓ, P, O, α, δ
)

###########
# Problem #
###########
∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]
ρ = ρs[ρname]
Afull = [E, A..., 1]

objective = objectives[problem](∇⁰f, ∇¹f, ∇²f, Δf)

#########
# Model #
#########
Random.seed!(seed)

if integration == "AQ"
  folder, filename = expName(
    Ωname, problem, Fname, ρname,
    η, NE, β, ν, seed, integration;
    P=P, O=O, α=α, δ=δ
  )
else
  folder, filename = expName(
    Ωname, problem, Fname, ρname,
    η, NE, β, ν, seed, integration;
    NΩ=NΩ, NΓ=NΓ
  )
end

folder = split(folder, "/")
insert!(folder, 2, "exploration")
folder = join(folder, "/")
u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)

if integration == "AQ"
  println(trunc(Int, mean(dict[:NΩ])), " ", sum([trunc(Int, mean(Nγ)) for Nγ in dict[:NΓ]]))
else
  println(dict[:NΩ], " ", dict[:NΓ])
end

####################
# Fine integration #
####################
if Ωname == "cartesian1d"
  dΩexp = MeshQuadrature(CartesianMesh(-1, +1, 100), 10)
elseif Ωname == "cartesian2d"
  dΩexp = MeshQuadrature(CartesianMesh(-1, +1, -1, +1, 100, 100), 10)
end

# println("L1\t$(L1(u, ∇⁰f, dΩexp))")
println("L2\t$(L2(u, ∇⁰f, dΩexp))")
# println("H1\t$(H1(u, ∇⁰f, ∇¹f, dΩexp))")

# Plot model
# plotModel(u, Ωτ, N=10000, showGrad=false)

# Comparison with solution
# plotComp(u, ∇⁰f, Ωτ, N=(E == 1) ? 1000 : 10000)

# Pointwise difference with solution
# plotDiff(u, ∇⁰f, Ωτ, N=(E == 1) ? 1000 : 10000)

# Basis functions
# plotBasis(u, Ωτ, N=1000, showGrad=false)

# Energy / time
# plotTrain(dict[:histResidual], logx=false, logy=false)

# L2 norm / time
# plotTrain(dict[:histL2], logx=false, logy=true)

# Mesh
# Ωlin = Lineariser(Ωκ, ρ, 5)
# dΩ = AdaptiveQuadrature(u, Ωlin, 5, 0.0f0, false)
# initialise!(dΩ)
# linearise!(u, Ωlin)
# plotLineariser(Ωlin)
# plotProjection(u, Ωτ, Ωlin, N=1000)

# plot(layout=(1, 1))
# for cycle in Ωlin.cycles
#   for i in eachindex(cycle)
#     j = (i == length(cycle)) ? 1 : i + 1
#     p₋, p₊ = Ωlin.points[cycle[i]], Ωlin.points[cycle[j]]
#     plot!([p₋[1], p₊[1]], [p₋[2], p₊[2]], c=colorBlue)
#   end
# end
# display(plot!(legend=false))
# savefig("mesh.pdf")
