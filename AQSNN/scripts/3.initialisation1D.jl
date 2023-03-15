using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("0.utils.jl")

problem = "poissonNitsche"
Ωname = "cartesian1d"
Fnames = ["sinc", "well"]

# Model
A = (10, 10)

# Optimisation
η = 1.0f-2
ν = 10
NE = 5000
γ = 1
seeds = 1:10

# # Monte-Carlo
NΩΓs = Dict("sinc" => (100, 1), "well" => (200, 1))

# # Adaptive
POs = Dict("sinc" => (3, 5), "well" => (5, 10))
α = 0.0
δ = 3

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
dΩexp = MeshQuadrature(CartesianMesh(-1, +1, 100), 10)

##################
# Default values #
##################
γ = T(γ)
α = T(α)
Afull = [E, A..., 1]

##############
# Build dump #
##############
for Fname in Fnames
  ρname = ρsfs[Fname]
  ρ = ρs[ρname]
  ∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]

  _, _, _, β, _, _, _, _, _, _, _ = getDefaults(
    T, E, problem, ρname, A,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  )

  dictAQ = Dict()
  dictMC = Dict()

  output = joinpath("data", "dumps", "initialisation_$(problem)_$(Fname)_$(Ωname).jld2")
  isfile(output) && continue

  # AQ
  dictAQ = Dict()
  P, O = POs[Fname]
  for seed in seeds
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "AQ";
      P=P, O=O, α=α, δ=δ
    )
    folder = splitpath(folder)
    insert!(folder, 2, "initialisation")
    folder = joinpath(folder)
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    dictAQ[seed] = (NΩ=nΩ, NΓ=nΓ, l2=l2, t=dict[:time])
  end

  # MC
  dictMC = Dict()
  NΩ, NΓ = NΩΓs[Fname]
  for seed in seeds
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "MC";
      NΩ=NΩ, NΓ=NΓ
    )
    folder = splitpath(folder)
    insert!(folder, 2, "initialisation")
    folder = joinpath(folder)
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    dictMC[seed] = (NΩ=nΩ, NΓ=nΓ, l2=l2, t=dict[:time])
  end

  FileIO.save(output, Dict("AQ" => dictAQ, "MC" => dictMC))
end

##############
# Print dump #
##############
for Fname in Fnames
  ρname = ρsfs[Fname]
  output = joinpath("data", "dumps", "initialisation_$(problem)_$(Fname)_$(Ωname).jld2")
  dump = FileIO.load(output)

  s = "# $problem, $Fname ($ρname) #"
  println("#"^length(s))
  println(s)
  println("#"^length(s))

  # AQ
  dicts = dump["AQ"]
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

  # MC
  dicts = dump["MC"]
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

  println()
end
