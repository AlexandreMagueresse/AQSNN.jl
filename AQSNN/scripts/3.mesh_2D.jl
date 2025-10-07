using AQSNN
using Statistics
using Plots
using LaTeXStrings

include("1.utils.jl")

#########
# Fig10 #
#########
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
β = 1.0f2
seed = 8

# Monte Carlo
NΩ = 5000
NΓ = 500

# Adaptive
P = 3
O = 5
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

##############
# Load model #
##############
folder, filename = expName(
  Ωname, problem, Fname, ρname,
  η, NE, β, ν, seed, "AQ";
  P=P, O=O, α=α, δ=δ
)
ρ = ρs[ρname]
u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)

##############
# Lineariser #
##############
P = 3
O = 2
colorBlue = RGB(0.1216, 0.4667, 0.7059)

for (l, α) in (("a", 0.0f0), ("b", 0.25f0))
  Ωlin = Lineariser(Ωκ, ρ, P, depth=δ)
  linearise!(u, Ωlin)

  # Collapse + simplexify
  dΩ = AdaptiveQuadrature(u, Ωlin, O, α, false)

  # Only collapse
  if iszero(α)
    collapse!(Ωlin, α)
  end

  txt_path = joinpath("results", "figures", "fig10$(l).txt")
  mkpath(dirname(txt_path))
  txt = open(txt_path, "w")
  write(txt, "Number of integration points (NΩ): $(length(dΩ.points))\n")
  write(txt, "Number of regions: $(length(Ωlin))\n")
  close(txt)

  Plots.plot(layout=(1, 1))
  for cycle in Ωlin.cycles
    for i in eachindex(cycle)
      j = (i == length(cycle)) ? 1 : i + 1
      p₋, p₊ = Ωlin.points[cycle[i]], Ωlin.points[cycle[j]]
      plot!([p₋[1], p₊[1]], [p₋[2], p₊[2]], c=colorBlue)
    end
  end
  Plots.plot!(legend=false)
  plot_path = joinpath("results", "figures", "fig10$(l).svg")
  mkpath(dirname(plot_path))
  Plots.savefig(plot_path)
end
