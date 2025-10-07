using AQSNN
using Statistics
using Plots
using LaTeXStrings

include("1.utils.jl")

function comparison1d(name, mode, problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
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

  ###################
  # Hyperparameters #
  ###################
  γ, β, α = T(γ), T(β), T(α)

  ###########
  # Problem #
  ###########
  ∇⁰f, ∇¹f, ∇²f, Δf = fs[Fname][E]
  ρ = ρs[ρname]
  Afull = [E, A..., 1]

  objective = objectives[problem](∇⁰f, ∇¹f, ∇²f, Δf)

  ########
  # Plot #
  ########
  Plots.plot(layout=(1, 1))

  X, F = nothing, nothing
  if mode != "l2"
    X = [-1:0.01:+1;;]'
    F = ∇⁰f(X)
  end

  for (NΩ, NΓ) in NΩΓs
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "MC";
      NΩ=NΩ, NΓ=NΓ
    )
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    NΩ, NΓ = dict[:NΩ], dict[:NΓ]

    if mode == "l2"
      X = [x for (x, y) in dict[:histL2]]
      Y = [y for (x, y) in dict[:histL2]]
      Plots.plot!(X, Y, yscale=:log10, lw=2, label="MC ($NΩ)")
    elseif mode == "abs"
      U = u(X)
      Plots.plot!(X[1, :], abs.(U[1, :] .- F[1, :]), lw=2, label="MC ($NΩ)")
    elseif mode == "rel"
      U = u(X)
      Plots.plot!(X[1, :], U[1, :], lw=2, label="MC ($NΩ)")
    end
  end

  for (P, O) in POs
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, "AQ";
      P=P, O=O, α=α, δ=δ
    )
    u, dict = AQSNN.load(joinpath(folder, filename), Ω, ρ)
    NΩ, NΓ = trunc(Int, mean(dict[:NΩ])), sum([trunc(Int, mean(Nγ)) for Nγ in dict[:NΓ]])

    if mode == "l2"
      X = [x for (x, y) in dict[:histL2]]
      Y = [y for (x, y) in dict[:histL2]]
      Plots.plot!(X, Y, yscale=:log10, lw=2, label="AQ ($NΩ, $P, $O)")
    elseif mode == "abs"
      U = u(X)
      Plots.plot!(X[1, :], abs.(U[1, :] .- F[1, :]), lw=2, label="AQ ($NΩ, $P, $O)")
    elseif mode == "rel"
      U = u(X)
      Plots.plot!(X[1, :], U[1, :], lw=2, label="AQ ($NΩ, $P, $O)")
    end
  end

  if mode == "l2"
    Plots.plot!(xlabel="Epoch", ylabel=L"\log \ \|\!\!\|u - \hat{u}\,\|\!\!\|_{L^2(\Omega)}")
    Plots.plot!(legend=:topright)
  elseif mode == "abs"
    Plots.plot!(xlabel="X", ylabel=L"\|u - \hat{u}\,\|")
    Plots.plot!(legend=:topright)
  elseif mode == "rel"
    Plots.plot!(X[1, :], F[1, :], lw=2, ls=:dot, label="exact")
    Plots.plot!(xlabel="X", ylabel=L"u")
    Plots.plot!(legend=:topright)
  end
  plot_path = joinpath("results", "figures", name * ".pdf")
  mkpath(dirname(plot_path))
  Plots.savefig(plot_path)
end

########
# Fig4 #
########
# Problem
problem = "poissonStrong"
Ωname = "cartesian1d"
Fname = "well"

# Model
ρname = ρsfs[Fname]
A = (10, 10)

# Optimisation
η = 1.0f-2
ν = 1
NE = 5000
γ = 1
β = 1.0f2
seed = 1

# Monte-Carlo
NΩΓs = [(100, 1),]

# Adaptive
POs = [(3, 5), (5, 2), (7, 2)]
α = 0.0f0
δ = 3

comparison1d("fig4a", "l2", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig4b", "abs", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig4c", "rel", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)

########
# Fig5 #
########
# Problem
problem = "poissonStrong"
Ωname = "cartesian1d"
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
seed = 1

# Monte-Carlo
NΩΓs = [(100, 1),]

# Adaptive
POs = [(2, 5), (3, 5), (5, 5)]
α = 0.0f0
δ = 3

comparison1d("fig5a", "l2", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig5b", "abs", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig5c", "rel", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)

########
# Fig6 #
########
# Problem
problem = "poissonNitsche"
Ωname = "cartesian1d"
Fname = "well"

# Model
ρname = ρsfs[Fname]
A = (10, 10)

# Optimisation
η = 1.0f-2
ν = 10
NE = 5000
γ = 1
β = 1.0f2
seed = 1

# Monte-Carlo
NΩΓs = [(100, 1),]

# Adaptive
POs = [(3, 2), (5, 2), (7, 5)]
α = 0.0f0
δ = 3

comparison1d("fig6a", "l2", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig6b", "abs", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig6c", "rel", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)

########
# Fig7 #
########
# Problem
problem = "poissonNitsche"
Ωname = "cartesian1d"
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
seed = 1

# Monte-Carlo
NΩΓs = [(100, 1), (500, 1)]

# Adaptive
POs = [(2, 5), (3, 5), (5, 5)]
α = 0.0f0
δ = 3

comparison1d("fig7a", "l2", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig7b", "abs", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
comparison1d("fig7c", "rel", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩΓs, POs, α, δ)
