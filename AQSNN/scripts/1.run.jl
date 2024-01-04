using AQSNN
using Random
using Statistics
using Printf

include("1.utils.jl")

function train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, integration; kwargs...)
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
  β = T(β)
  NΩ = get(kwargs, :NΩ, -1)
  NΓ = get(kwargs, :NΓ, -1)
  P = get(kwargs, :P, -1)
  O = get(kwargs, :O, -1)
  α = T(get(kwargs, :α, -1))
  δ = get(kwargs, :δ, -1)

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
      η, NE, β, ν, seed, integration; kwargs...
    )
  else
    folder, filename = expName(
      Ωname, problem, Fname, ρname,
      η, NE, β, ν, seed, integration; kwargs...
    )
  end
  mkpath(folder)
  println(filename)

  file = joinpath(folder, filename)
  if isfile(file)
    println("File exists")
    return
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
end

###################
# Exploration, 1D #
###################
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
NΩΓs = ((50, 1), (100, 1), (500, 1))

# Adaptive
POs = Dict(
  "sinc" => [(2, 2), (2, 5), (2, 10), (3, 2), (3, 5), (5, 2), (5, 5)],
  "well" => [(3, 2), (3, 5), (3, 10), (5, 2), (5, 5), (7, 2), (7, 5)]
)
α = 0.0f0
δ = 3

# This will run the following
# MC:     144 exps, around  2 700 s
# AQ:     336 exps, around  6 500 s
# Total:  480 exps, around  9 200 s (2h 30min)
for problem in problems
  for Fname in Fnames
    ρname = ρsfs[Fname]
    for η in ηs
      for ν in νs
        for (NΩ, NΓ) in NΩΓs
          train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
        end
        for (P, O) in POs[Fname]
          train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
        end
      end
    end
  end
end

###################
# Exploration, 2D #
###################
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
seed = 1

# Monte-Carlo
NΩΓs = Dict(
  "sinc" => [(1000, 100), (5000, 500), (10000, 1000)],
  "well" => [(1000, 100), (5000, 500), (10000, 1000)]
)

# Adaptive
POs = Dict(
  "sinc" => [(2, 2), (2, 5), (3, 2)],
  "well" => [(3, 2), (3, 5), (5, 2)]
)
α = 0.0f0
δ = 3

# This will run the following
# MC:      6 exps, around 2 500 s
# AQ:      6 exps, around 2 200 s
# Total:  12 exps, around 4 700 s (1h 20min)
for Fname in Fnames
  ρname = ρsfs[Fname]
  for (NΩ, NΓ) in NΩΓs[Fname]
    train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
  end
  for (P, O) in POs[Fname]
    train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
  end
end

######################
# Initialisation, 1D #
######################
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
  "sinc" => [(100, 1),],
  "well" => [(200, 1),]
)

# Adaptive
POs = Dict(
  "sinc" => [(3, 5),],
  "well" => [(5, 10),]
)
α = 0.0f0
δ = 3

# This will run the following
# MC:     20 exps, around 400 s
# AQ:     20 exps, around 400 s
# Total:  40 exps, around 800 s (15min)
for seed in seeds
  for Fname in Fnames
    ρname = ρsfs[Fname]
    for (NΩ, NΓ) in NΩΓs[Fname]
      train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
    end
    for (P, O) in POs[Fname]
      train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
    end
  end
end

######################
# Initialisation, 2D #
######################
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
  "sinc" => [(10000, 1000),],
  "well" => [(5000, 500),]
)

# Adaptive
POs = Dict(
  "sinc" => [(3, 5),],
  "well" => [(3, 5),]
)
α = 0.0f0
δ = 3

# This will run the following
# MC:     20 exps, around  9 500 s
# AQ:     20 exps, around  8 100 s
# Total:  40 exps, around 17 600 s (5h)
for seed in seeds
  for Fname in Fnames
    ρname = ρsfs[Fname]
    for (NΩ, NΓ) in NΩΓs[Fname]
      train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
    end
    for (P, O) in POs[Fname]
      train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
    end
  end
end

#################
# Reduction, 1D #
#################
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
seed = 1

# Monte-Carlo
NΩΓs = ((100, 1), (200, 1))

# Adaptive
POs = Dict(
  "sinc" => [(5, 10),],
  "well" => [(5, 10),]
)
αs = [0.0f0, 0.5f0, 1.0f0]
δ = 3

# This will run the following
# MC:      4 exps, around 100 s
# AQ:      6 exps, around 150 s
# Total:  10 exps, around 250 s (5min)
for Fname in Fnames
  ρname = ρsfs[Fname]
  for (NΩ, NΓ) in NΩΓs
    train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
  end
  for (P, O) in POs[Fname]
    for α in αs
      train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
    end
  end
end

#################
# Reduction, 2D #
#################
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

# This will run the following
# MC:      4 exps, around 1 700 s
# AQ:      8 exps, around 3 500 s
# Total:  12 exps, around 5 200 s (1h 30min)
for Fname in Fnames
  ρname = ρsfs[Fname]
  for (NΩ, NΓ) in NΩΓs[Fname]
    train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
  end
  for (P, O) in POs[Fname]
    for α in αs
      train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
    end
  end
end

##########
# Rhombi #
##########
problem = "poissonNitsche"
Ωname = "rhombi"
Fname = "xpy"

# Model
A = (20, 20)

# Optimisation
η = 1.0f-2
ν = 10
NE = 1
γ = 1
β = 1.0f2
seed = 4

# Monte-Carlo
NΩ = 6000
NΓ = 300

# Adaptive
P = 3
O = 2
α = 0.0f0
δ = 3

# This will run the following
# MC:     1 exps, around   650 s
# AQ:     1 exps, around   700 s
# Total:  2 exps, around 1 350 s (20min)
ρname = ρsfs[Fname]
train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "MC"; NΩ, NΓ)
train(problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, "AQ"; P, O, α, δ)
