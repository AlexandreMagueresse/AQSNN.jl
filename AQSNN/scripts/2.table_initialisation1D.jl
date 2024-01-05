using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("1.utils.jl")

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
β = 1.0f2
seeds = 1:10

# Monte-Carlo
NΩΓs = Dict(
  "sinc" => (100, 1),
  "well" => (200, 1)
)

# Adaptive
POs = Dict(
  "sinc" => (3, 5),
  "well" => (5, 10)
)
α = 0.0f0
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

  jld_path = joinpath("data", "tables", "initialisation_$(problem)_$(Fname)_$(Ωname).jld2")
  isfile(jld_path) && continue

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
  P, O = POs[Fname]
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

  mkpath(dirname(jld_path))
  FileIO.save(jld_path, Dict("AQ" => dictAQ, "MC" => dictMC))
end

###############
# Print table #
###############
tableLetter = 'a'

for Fname in Fnames
  ρname = ρsfs[Fname]
  jld_path = joinpath("data", "tables", "initialisation_$(problem)_$(Fname)_$(Ωname).jld2")
  jld = FileIO.load(jld_path)
  txt_path = joinpath("results", "tables", "table7$(tableLetter).txt")
  mkpath(dirname(txt_path))
  txt = open(txt_path, "w")

  s = "# $(problem), $(Fname) ($(ρname)) #"
  write(txt, "#"^length(s) * "\n")
  write(txt, s * "\n")
  write(txt, "#"^length(s) * "\n")
  s = @sprintf("D\t\t  P\t  O\t\t    NΩ\t    NΓ\t\tTime\t\tL2")
  write(txt, s * "\n")

  ######
  # MC #
  ######
  dicts = jld["MC"]
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
  l2 = mean(l2s)
  sl2 = std(l2s)
  ml2 = minimum(l2s)
  Ml2 = maximum(l2s)
  t = mean(ts)

  s = @sprintf("1\tMC\t\t\t\t% 6i\t% 6i\t\t%.1f\t\t%.2E\t%.2E\t%.2E\t%.2E", NΩ, 2, t, ml2, l2, sl2, Ml2)
  write(txt, s * "\n")

  ######
  # AQ #
  ######
  dicts = jld["AQ"]
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
  l2 = mean(l2s)
  sl2 = std(l2s)
  ml2 = minimum(l2s)
  Ml2 = maximum(l2s)
  t = mean(ts)

  s = @sprintf("1\tAQ\t% 3i\t% 3i\t\t% 6i\t% 6i\t\t%.1f\t\t%.2E\t%.2E\t%.2E\t%.2E", P, O, NΩ, 2, t, ml2, l2, sl2, Ml2)
  write(txt, s * "\n")

  close(txt)
  tableLetter += 1
end
