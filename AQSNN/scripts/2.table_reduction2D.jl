using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("1.utils.jl")

problems = "poissonNitsche"
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
seed = 1

# Monte-Carlo
NΩΓs = Dict(
  "sinc" => [(1000, 100), (5000, 500)],
  "well" => [(5000, 500), (10000, 1000)],
)

# Adaptive
POs = Dict(
  "sinc" => [(3, 2),],
  "well" => [(5, 5),]
)
αs = [0.0f0, 0.1f0, 0.25f0, 0.5f0]
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

  output = joinpath("tables", "reduction_$(problem)_$(Fname)_$(Ωname).jld2")
  # isfile(output) && continue

  ######
  # MC #
  ######
  for (NΩ, NΓ) in NΩΓs[Fname]
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "MC";
      NΩ=NΩ, NΓ=NΓ
    )
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    nΩ = mean(dict[:NΩ])
    nΓ = sum(mean, dict[:NΓ])
    l2 = L2(u, ∇⁰f, dΩexp)
    dictMC[(NΩ, NΓ)] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
  end

  ######
  # AQ #
  ######
  for (P, O) in POs[Fname]
    dictAQ[(P, O)] = Dict()

    for α in αs
      folder, filename = expName(
        Ωname, problem, Fname, ρname,
        η, NE, β, ν, seed, "AQ";
        P=P, O=O, α=α, δ=δ
      )
      u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
      nΩ = mean(dict[:NΩ])
      nΓ = sum(mean, dict[:NΓ])
      l2 = L2(u, ∇⁰f, dΩexp)
      dictAQ[(P, O)][α] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
    end
  end

  FileIO.save(output, Dict("AQ" => dictAQ, "MC" => dictMC))
end

###############
# Print table #
###############
for Fname in Fnames
  ρname = ρsfs[Fname]
  output = joinpath("tables", "reduction_$(problem)_$(Fname)_$(Ωname).jld2")
  table = FileIO.load(output)

  s = "# $(problem), $(Fname) ($(ρname)) #"
  println("#"^length(s))
  println(s)
  println("#"^length(s))

  ######
  # MC #
  ######
  dicts = table["MC"]
  for (NΩ, NΓ) in sort(collect(keys(dicts)))
    dict = dicts[(NΩ, NΓ)]
    l2 = dict.l2
    s = @sprintf("MC\t\t%i\t%.2E", NΩ, l2)
    println(s)
  end

  ######
  # AQ #
  ######
  dicts = table["AQ"]
  for (P, O) in sort(collect(keys(dicts)))
    dict = dicts[(P, O)]
    for α in αs
      NΩ = dict[α].NΩ
      l2 = dict[α].l2
      s = @sprintf("AQ %i %i\t\t%i\t%.2E", P, O, NΩ, l2)
      println(s)
    end
  end

  println()
end
