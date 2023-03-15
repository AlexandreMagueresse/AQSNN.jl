using AQSNN
using Random
using Statistics
using Printf

include("0.utils.jl")

############
# Settings #
############
# Leave -1 for default

# Problem
problem = "poissonNitsche" # "poissonStrong"
Ωname = "cartesian1d" # "cartesian2d", "rhombi"
Fname = "sinc" # "well", "xpy"

# Model
ρname = ρsfs[Fname] # "tanh", "ReLUε"
A = (10, 10) # (20, 20)

# Optimisation
η = 1.0f-3 # learning rate # 1.0f-2
ν = 5000 # resampling frequency # 1, 10, 100, 500, 1000, 5000
NE = 5000 # number of epochs # 10000
γ = 1 # decay rate for the learning rate
β = -1 # penalty/nitsche coefficient
seed = 1 # seed # 1 to 10

# Integration
integration = "AQ" # "MC"

# # Monte-Carlo
NΩ = 50 # 100, 1000, 5000, 10000
NΓ = 1 # 1 in 1D, NΩ / 10 in 2D

# # Adaptive
P = 5 # number of pieces
O = 10 # order of the quadrature
α = -1 # merging threshold (percentage of the median)
δ = -1 # depth of linearisation, do not touch

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

######################
# Preliminary checks #
######################
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
mkpath(folder)
println(filename)

file = joinpath(folder, filename)
if isfile(file)
  throw("File exists")
end

# Model
Random.seed!(seed)
u = Sequential{T,E}(Afull, ρ)

# Integration
if integration == "MC"
  dΩ = MonteCarloQuadrature(Ωτ, NΩ)
  dΓ = MonteCarloQuadrature(Γτ, NΓ)
elseif integration == "AQ"
  Ωlin = Lineariser(Ωκ, ρ, P, depth=δ)
  dΩ = AdaptiveQuadrature(u, Ωlin, O, α, false)

  Γlin = Lineariser.(Γκ, ρ, P, depth=δ)
  dΓ = AdaptiveQuadrature.(u, Γlin, O, α, false)
end

############
# Training #
############
ω = paramsWeights(u)
θ = paramsAll(u)

trainings = [
  Training(
    NE,
    _ -> β,
    dΩ,
    dΓ,
    ν,
    [
      ADAM(θ, η, γ=γ),
      # SGD(θ, η),
      # LBFGS(θ),
      # LinearSolver()
    ],
  ),
]

# Training
histResidual = Vector{Tuple{Int,Float32}}()
histL2 = Vector{Tuple{Int,Float32}}()

dΩexp = nothing
if Ωname == "cartesian1d"
  dΩexp = MeshQuadrature(CartesianMesh(-1, +1, 100), 10)
elseif Ωname == "cartesian2d"
  dΩexp = MeshQuadrature(CartesianMesh(-1, +1, -1, +1, 100, 100), 10)
end

function learningCurve(epoch, u, objective, β, dΩ, dΓ)
  # AQSNN.save(joinpath(folder, "checkpoints", "$(epoch).jld2"), Ω, u)

  r = residual(objective, u, β, dΩ, dΓ)
  push!(histResidual, (epoch, r))

  if !isnothing(dΩexp)
    l = L2(u, ∇⁰f, dΩexp)
    push!(histL2, (epoch, l))

    sl = @sprintf("%.3E", l)
    sr = @sprintf("%.3E", r)

    if integration == "AQ"
      NΩAQ = trunc(Int, mean(dΩ.npoints))
      NΓAQ = trunc(Int, sum([mean(dγ.npoints) for dγ in dΓ]))
      println(epoch, " ", sl, " ", sr, " ", NΩAQ, " ", NΓAQ)
    else
      println(epoch, " ", sl, " ", sr, " ", NΩ, " ", NΓ)
    end
  else
    sr = @sprintf("%.3E", r)

    if integration == "AQ"
      NΩAQ = trunc(Int, mean(dΩ.npoints))
      NΓAQ = trunc(Int, sum([mean(dγ.npoints) for dγ in dΓ]))
      println(epoch, " ", sr, " ", NΩAQ, " ", NΓAQ)
    else
      println(epoch, " ", sr, " ", NΩ, " ", NΓ)
    end
  end

  # if mod(epoch, 500) == 0
  #   Ωlin = Lineariser(Ωκ, ρ, P)
  #   linearise!(u, Ωlin)
  #   plotLineariser(Ωlin)
  #   plotProjection(u, Ωτ, Ωlin, N=1000)
  # end
end

info = @timed train!(u, trainings, objective, [learningCurve])

AQSNN.save(file, Ω, u,
  histResidual=histResidual,
  histL2=histL2,
  NΩ=(integration == "AQ") ? dΩ.npoints : NΩ,
  NΓ=(integration == "AQ") ? [dγ.npoints for dγ in dΓ] : NΓ,
  time=info.time,
)
