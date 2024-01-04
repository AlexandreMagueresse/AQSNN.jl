using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("1.utils.jl")

problem = "poissonNitsche"
Ωname = "cartesian2d"
Fnames = ["sinc", "well"]

# Model
A = (10, 10)

# Optimisation
η = 1.0f-2
ν = 10
NE = 5000
γ = 1
β = 1.0f2
seeds = 1:10

# Monte-Carlo
NΩΓs = Dict(
  "sinc" => (10000, 1000),
  "well" => (5000, 500)
)

# Adaptive
POs = Dict(
  "sinc" => (3, 5),
  "well" => (3, 5)
)

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

####################
# Fine integration #
####################
dΩexp = MeshQuadrature(CartesianMesh(-1, +1, -1, +1, 100, 100), 10)

###################
# Hyperparameters #
###################
γ, β, α = T(γ), T(β), T(α)
Afull = [E, A..., 1]

###############
# Build table #
###############
for Fname in Fnames
  ∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]
  ρname = ρsfs[Fname]
  ρ = ρs[ρname]

  dictMC = Dict()
  dictAQ = Dict()

  output = joinpath("tables", "initialisation_$(problem)_$(Fname)_$(Ωname).jld2")
  isfile(output) && continue

  ######
  # MC #
  ######
  NΩ, NΓ = NΩΓs[Fname]
  for seed in seeds
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "MC";
      NΩ=NΩ, NΓ=NΓ
    )
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    t = dict[:time]
    dictMC[seed] = (NΩ=nΩ, NΓ=nΓ, l2=l2, t=t)
  end

  ######
  # AQ #
  ######
  (P, O) = POs[Fname]
  for seed in seeds
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "AQ";
      P=P, O=O, α=α, δ=δ
    )
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    t = dict[:time]
    dictAQ[seed] = (NΩ=nΩ, NΓ=nΓ, l2=l2, t=t)
  end

  FileIO.save(output, Dict("AQ" => dictAQ, "MC" => dictMC))
end

###############
# Print table #
###############
for Fname in Fnames
  ρname = ρsfs[Fname]
  output = joinpath("tables", "initialisation_$(problem)_$(Fname)_$(Ωname).jld2")
  table = FileIO.load(output)

  s = "# $(problem), $(Fname) ($(ρname)) #"
  println("#"^length(s))
  println(s)
  println("#"^length(s))

  ######
  # MC #
  ######
  dicts = table["MC"]
  NΩ, NΓ = NΩΓs[Fname]
  NΩs = []
  NΓs = []
  l2s = []
  ts = []
  for seed in seeds
    push!(NΩs, dicts[seed][:NΩ])
    push!(NΓs, dicts[seed][:NΓ])
    push!(l2s, dicts[seed][:l2])
    push!(ts, dicts[seed][:t])
  end
  NΩ = trunc(Int, mean(NΩs))
  NΓ = trunc(Int, mean(NΓs))
  l2 = mean(l2s)
  sl2 = std(l2s)
  ml2 = minimum(l2s)
  Ml2 = maximum(l2s)
  t = mean(ts)

  s = @sprintf("MC\t\t%i\t%i\t%.1f\t%.2E\t%.2E\t%.2E\t%.2E", NΩ, NΓ, t, ml2, l2, sl2, Ml2)
  println(s)

  ######
  # AQ #
  ######
  dicts = table["AQ"]
  P, O = POs[Fname]
  NΩs = []
  NΓs = []
  l2s = []
  ts = []
  for seed in seeds
    push!(NΩs, dicts[seed][:NΩ])
    push!(NΓs, dicts[seed][:NΓ])
    push!(l2s, dicts[seed][:l2])
    push!(ts, dicts[seed][:t])
  end
  NΩ = trunc(Int, mean(NΩs))
  NΓ = trunc(Int, mean(NΓs))
  l2 = mean(l2s)
  sl2 = std(l2s)
  ml2 = minimum(l2s)
  Ml2 = maximum(l2s)
  t = mean(ts)

  s = @sprintf("AQ %i %i\t\t%i\t%i\t%.1f\t%.2E\t%.2E\t%.2E\t%.2E", P, O, NΩ, NΓ, t, ml2, l2, sl2, Ml2)
  println(s)

  println()
end
