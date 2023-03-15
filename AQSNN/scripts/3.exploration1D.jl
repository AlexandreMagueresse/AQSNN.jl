using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("0.utils.jl")

problems = ["poissonStrong", "poissonNitsche"]
Ωname = "cartesian1d"
Fnames = ["sinc", "well"]

# Model
A = (10, 10)

# Optimisation
ηs = [1.0f-2, 1.0f-3]
νs = [1, 10, 100, 500, 1000, 5000]
NE = 5000
γ = 1
seed = 1

# # Monte-Carlo
NΩΓs = ((50, 1), (100, 1), (500, 1))

# # Adaptive
Ps = Dict("sinc" => [2, 3, 5], "well" => [3, 5, 7])
Os = [2, 5, 10]
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
for problem in problems
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

    output = joinpath("data", "dumps", "exploration_$(problem)_$(Fname)_$(Ωname).jld2")
    isfile(output) && continue

    for η in ηs
      # AQ
      dictAQ[η] = Dict()
      for P in Ps[Fname]
        for O in Os
          dictAQ[η][(P, O)] = Dict()

          for ν in νs
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
            dictAQ[η][(P, O)][ν] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
          end
        end
      end

      # MC
      dictMC[η] = Dict()
      for (NΩ, NΓ) in NΩΓs
        dictMC[η][(NΩ, NΓ)] = Dict()
        for ν in νs
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
            dictMC[η][(NΩ, NΓ)][ν] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
        end
      end
    end

    FileIO.save(output, Dict("AQ" => dictAQ, "MC" => dictMC))
  end
end

##############
# Print dump #
##############
for problem in problems
  for Fname in Fnames
    ρname = ρsfs[Fname]
    output = joinpath("data", "dumps", "exploration_$(problem)_$(Fname)_$(Ωname).jld2")
    dump = FileIO.load(output)

    s = "# $problem, $Fname ($ρname) #"
    println("#"^length(s))
    println(s)
    println("#"^length(s))

    for η in ηs
      println("-"^10)
      println(η)
      println("-"^10)
      s = @sprintf("\t\t\t%i\t\t%i\t\t%i\t\t%i\t\t%i\t\t%i", νs...)
      println(s)

      # AQ
      dicts = dump["AQ"][η]
      for (P, O) in sort(collect(keys(dicts)))
        dict = dicts[(P, O)]
        NΩ = trunc(Int, mean(dict[ν].NΩ for ν in νs))
        l2s = [dict[ν].l2 for ν in νs]
        s = @sprintf("AQ %i %i\t\t%i\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E", P, O, NΩ, l2s...)
        println(s)
      end

      # MC
      dicts = dump["MC"][η]
      for (NΩ, NΓ) in sort(collect(keys(dicts)))
        dict = dicts[(NΩ, NΓ)]
        NΩ = trunc(Int, mean(dict[ν].NΩ for ν in νs))
        l2s = [dict[ν].l2 for ν in νs]
        s = @sprintf("MC\t\t%i\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E", NΩ, l2s...)
        println(s)
      end
    end

    println()
  end
end
