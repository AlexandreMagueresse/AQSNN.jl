using AQSNN
using Random
using Statistics
using Printf
using FileIO

include("1.utils.jl")

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
β = 1.0f2
seed = 1

# Monte-Carlo
NΩΓs = ((50, 1), (100, 1))

# Adaptive
POs = Dict(
  "sinc" => [(2, 2), (2, 5), (2, 10), (3, 2), (3, 5), (5, 2)],
  "well" => [(3, 2), (3, 5), (3, 10), (5, 2), (5, 5), (7, 2)]
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
for problem in problems
  for Fname in Fnames
    ∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]
    ρname = ρsfs[Fname]
    ρ = ρs[ρname]

    dictMC = Dict()
    dictAQ = Dict()

    jld_path = joinpath("data", "tables", "exploration_$(problem)_$(Fname)_$(Ωname).jld2")
    isfile(jld_path) && continue

    for η in ηs
      ######
      # MC #
      ######
      dictMC[η] = Dict()
      for (NΩ, NΓ) in NΩΓs
        dictMC[η][(NΩ, NΓ)] = Dict()
        for ν in νs
          folder, filename = expName(
            Ωname, problem, Fname, ρname,
            η, NE, β, ν, seed, "MC";
            NΩ=NΩ, NΓ=NΓ
          )
          u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
          nΩ = mean(dict[:NΩ])
          nΓ = sum(mean, dict[:NΓ])
          l2 = L2(u, ∇⁰f, dΩexp)
          dictMC[η][(NΩ, NΓ)][ν] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
        end
      end

      ######
      # AQ #
      ######
      dictAQ[η] = Dict()
      for (P, O) in POs[Fname]
        dictAQ[η][(P, O)] = Dict()

        for ν in νs
          folder, filename = expName(
            Ωname, problem, Fname, ρname,
            η, NE, β, ν, seed, "AQ";
            P=P, O=O, α=α, δ=δ
          )
          u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
          nΩ = mean(dict[:NΩ])
          nΓ = sum(mean, dict[:NΓ])
          l2 = L2(u, ∇⁰f, dΩexp)
          dictAQ[η][(P, O)][ν] = (NΩ=nΩ, NΓ=nΓ, l2=l2)
        end
      end
    end

    mkpath(dirname(jld_path))
    FileIO.save(jld_path, Dict("AQ" => dictAQ, "MC" => dictMC))
  end
end

###############
# Print table #
###############
tableNumber = 2

for problem in problems
  for Fname in Fnames
    ρname = ρsfs[Fname]
    jld_path = joinpath("data", "tables", "exploration_$(problem)_$(Fname)_$(Ωname).jld2")
    jld = FileIO.load(jld_path)
    txt_path = joinpath("results", "tables", "table$(tableNumber).txt")
    mkpath(dirname(txt_path))
    txt = open(txt_path, "w")

    s = "# $(problem), $(Fname) ($(ρname)) #"
    write(txt, "#"^length(s) * "\n")
    write(txt, s * "\n")
    write(txt, "#"^length(s) * "\n")

    for η in ηs
      write(txt, "-"^10 * "\n")
      write(txt, "$(η)\n")
      write(txt, "-"^10 * "\n")
      s = @sprintf("\t  P\t  O\t\t  NΩ\t\tL2")
      write(txt, s * "\n")
      s = @sprintf("\t\t\t\t\t\t\t% 8i\t% 8i\t% 8i\t% 8i\t% 8i\t% 8i", νs...)
      write(txt, s * "\n")

      ######
      # MC #
      ######
      dicts = jld["MC"][η]
      for (NΩ, NΓ) in sort(collect(keys(dicts)))
        dict = dicts[(NΩ, NΓ)]
        NΩ = trunc(Int, mean(dict[ν].NΩ for ν in νs))
        l2s = [dict[ν].l2 for ν in νs]
        s = @sprintf("MC\t\t\t\t% 4i\t\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E", NΩ, l2s...)
        write(txt, s * "\n")
      end

      ######
      # AQ #
      ######
      dicts = jld["AQ"][η]
      for (P, O) in sort(collect(keys(dicts)))
        dict = dicts[(P, O)]
        NΩ = trunc(Int, mean(dict[ν].NΩ for ν in νs))
        l2s = [dict[ν].l2 for ν in νs]
        s = @sprintf("AQ\t% 3i\t% 3i\t\t% 4i\t\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E\t%.2E", P, O, NΩ, l2s...)
        write(txt, s * "\n")
      end
    end

    write(txt, "\n")
    close(txt)
    tableNumber += 1
  end
end
