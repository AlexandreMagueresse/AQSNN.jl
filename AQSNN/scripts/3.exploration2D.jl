using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("0.utils.jl")

problem = "poissonNitsche"
Ωname = "cartesian2d"
Fnames = ["sinc", "well"]

# Model
A = (10, 10)

# Optimisation
η = -1
ν = -1
NE = -1
γ = 1 #  (1 / 10)^(1 / 50)
seed = 1

# # Monte-Carlo
NΩΓs = Dict(
  "sinc" => [(1000, 100), (5000, 500), (10000, 1000)],
  "well" => [(1000, 100), (5000, 500), (10000, 1000)]
)

# # Adaptive
P = -1
O = -1
α = -1
δ = -1

POs = Dict(
  "sinc" => [(2, 2), (2, 5), (3, 2)],
  "well" => [(3, 2), (3, 5), (5, 2)]
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

##################
# Default values #
##################
γ = T(γ)
α = T(α)
Afull = [E, A..., 1]

##############
# Build dump #
##############
dΩexp = MeshQuadrature(CartesianMesh(-1, +1, -1, +1, 100, 100), 10)

for Fname in Fnames
  ρname = ρsfs[Fname]
  ρ = ρs[ρname]
  ∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]

  η, γ, NE, β, ν, _, _, _, _, α, δ = getDefaults(
    T, E, problem, ρname, A,
    η, γ, NE, -1, ν, -1, -1, P, O, α, δ
  )

  output = joinpath("data", "dumps", "exploration_$(problem)_$(Fname)_$(Ωname).jld2")
  isfile(output) && continue

  # AQ
  dictAQ = Dict()
  for (P, O) in POs[Fname]
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "AQ";
      P=P, O=O, α=α, δ=δ
    )
    folder = splitpath(folder)
    insert!(folder, 2, "exploration")
    folder = joinpath(folder)
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    dictAQ[(P, O)] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
  end

  # MC
  dictMC = Dict()
  for (NΩ, NΓ) in NΩΓs[Fname]
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "MC";
      NΩ=NΩ, NΓ=NΓ
    )
    folder = splitpath(folder)
    insert!(folder, 2, "exploration")
    folder = joinpath(folder)
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    dictMC[(NΩ, NΓ)] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
  end

  FileIO.save(output, Dict("AQ" => dictAQ, "MC" => dictMC))
end

##############
# Print dump #
##############
for Fname in Fnames
  ρname = ρsfs[Fname]
  output = joinpath("data", "dumps", "exploration_$(problem)_$(Fname)_$(Ωname).jld2")
  dump = FileIO.load(output)

  s = "# $problem, $Fname ($ρname) #"
  println("#"^length(s))
  println(s)
  println("#"^length(s))

  dicts = dump["AQ"]
  for (P, O) in sort(collect(keys(dicts)))
    dict = dicts[(P, O)]
    NΩ = trunc(Int, dict.NΩ)
    NΓ = trunc(Int, dict.NΓ)
    l2 = dict.l2
    s = @sprintf("AQ %i %i\t\t%i %i\t%.2E", P, O, NΩ, NΓ, l2)
    println(s)
  end

  # MC
  dicts = dump["MC"]
  for (NΩ, NΓ) in sort(collect(keys(dicts)))
    dict = dicts[(NΩ, NΓ)]
    NΩ = trunc(Int, dict.NΩ)
    NΓ = trunc(Int, dict.NΓ)
    l2 = dict.l2
    s = @sprintf("MC\t\t%i %i\t%.2E", NΩ, NΓ, l2)
    println(s)
  end

  println()
end
