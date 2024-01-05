using Gridap
using GridapGmsh
using WriteVTK
using AQSNN

include("1.utils.jl")

function comparison2d(name, problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩ, NΓ, P, O, α, δ)
  Ω = Ωs[Ωname]
  ρname = ρsfs[Fname]
  ρ = ρs[ρname]

  ######
  # MC #
  ######
  folder, filename = expName(
    Ωname, problem, Fname, ρname,
    η, NE, β, ν, seed, "MC";
    NΩ=NΩ, NΓ=NΓ
  )
  uMC, dictMC = AQSNN.load(joinpath(folder, filename), Ω, ρ)

  ######
  # AQ #
  ######
  folder, filename = expName(
    Ωname, problem, Fname, ρname,
    η, NE, β, ν, seed, "AQ";
    P=P, O=O, α=α, δ=δ
  )
  uAQ, dictAQ = AQSNN.load(joinpath(folder, filename), Ω, ρ)

  #######
  # FEM #
  #######
  # Import mesh
  if Ωname == "cartesian2d"
    model = GmshDiscreteModel(joinpath("data", "meshes", "square.msh"))
  elseif Ωname == "rhombi"
    model = GmshDiscreteModel(joinpath("data", "meshes", "rhombi.msh"))
  else
    throw("Unknown domain")
  end

  # Forcing term
  function g(x)
    if Fname == "well"
      α = 10
      tanh(α * (x[1]^2 + x[2]^2 - 0.5^2))
    elseif Fname == "sinc"
      α = 2
      sinc(α * x[1]) * sinc(α * x[2])
    elseif Fname == "xpy"
      0
    else
      throw("Unknown forcing term")
    end
  end

  function f(x)
    if Fname == "well"
      α = 10
      u = g(x)
      -(4 * α * (1 - 2 * α * (x[1]^2 + x[2]^2) * u) * (1 - u^2))
    elseif Fname == "sinc"
      α = 2
      -(∂²sinc(α, x[1]) * sinc(α * x[2]) + sinc(α * x[1]) * ∂²sinc(α, x[2]))
    elseif Fname == "xpy"
      x[1] + x[2]
    else
      throw("Unknown forcing term")
    end
  end

  # FE space
  order = 1
  reffe = ReferenceFE(lagrangian, Float64, order)

  V = TestFESpace(model, reffe, dirichlet_tags="boundary")
  U = TrialFESpace(V, g)

  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2 * order)

  # Weak form
  a(u, v) = ∫(Gridap.∇(v) ⋅ Gridap.∇(u)) * dΩ
  l(v) = ∫(f * v) * dΩ
  op = AffineFEOperator(a, l, U, V)

  # Solve PDE with FEM
  uh = solve(op)

  #########
  # Error #
  #########
  dΩexp = Measure(Ω, 10)
  ∫ex = sqrt(sum(∫(uh * uh) * dΩexp))

  uuMC(x) = uMC([x[1]; x[2]])[1]
  eMC = uh - uuMC
  l2MC = sqrt(sum(∫(eMC * eMC) * dΩexp)) / ∫ex

  uuAQ(x) = uAQ([x[1]; x[2]])[1]
  eAQ = uh - uuAQ
  l2AQ = sqrt(sum(∫(eAQ * eAQ) * dΩexp)) / ∫ex

  txt_path = joinpath("results", "figures", name * ".txt")
  mkpath(dirname(txt_path))
  txt = open(txt_path, "w")
  write(txt, "MC\t$(l2MC)\n")
  write(txt, "AQ\t$(l2AQ)\n")
  close(txt)

  ########
  # Save #
  ########
  # Extract node coordinates
  npoints = length(model.grid.node_coordinates)
  points = Matrix{Float64}(undef, 2, npoints)
  for i in 1:npoints
    points[1, i] = model.grid.node_coordinates[i].data[1]
    points[2, i] = model.grid.node_coordinates[i].data[2]
  end
  cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, ids) for ids in model.grid.cell_node_ids]

  vtk_path = joinpath("results", "figures", name)
  mkpath(dirname(vtk_path))
  vtk_grid(vtk_path, points, cells) do vtk
    vtk["FEM"] = evaluate(uh, [Gridap.Point(points[1, i], points[2, i]) for i in 1:npoints])
    vtk["MC"] = uMC(points)[1, :]
    vtk["AQ"] = uAQ(points)[1, :]
  end
end

########
# Fig8 #
########
# Problem
problem = "poissonNitsche"
Ωname = "cartesian2d"
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
NΩ = 5000
NΓ = 500

# Adaptive
P = 3
O = 5
α = 0.0f0
δ = 3

comparison2d("fig8", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩ, NΓ, P, O, α, δ)
src_path = joinpath("data", "pvsm", "fig8.pvsm")
dst_path = joinpath("results", "figures", "fig8.pvsm")
cp(src_path, dst_path, force=true)

########
# Fig9 #
########
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
seed = 1

# Monte-Carlo
NΩ = 5000
NΓ = 500

# Adaptive
P = 2
O = 5
α = 0.0f0
δ = 3

comparison2d("fig9", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩ, NΓ, P, O, α, δ)
src_path = joinpath("data", "pvsm", "fig9.pvsm")
dst_path = joinpath("results", "figures", "fig9.pvsm")
cp(src_path, dst_path, force=true)

#########
# Fig11 #
#########
# Problem
problem = "poissonNitsche"
Ωname = "rhombi"
Fname = "xpy"

# Model
ρname = ρsfs[Fname]
A = (20, 20)

# Optimisation
η = 1.0f-2
ν = 10
NE = 10000
γ = 1
β = 1.0f2
seed = 1

# Monte-Carlo
NΩ = 6000
NΓ = 300

# Adaptive
P = 3
O = 2
α = 0.0f0
δ = 3

comparison2d("fig11", problem, Ωname, Fname, ρname, A, η, ν, NE, γ, β, seed, NΩ, NΓ, P, O, α, δ)
src_path = joinpath("data", "pvsm", "fig11.pvsm")
dst_path = joinpath("results", "figures", "fig11.pvsm")
cp(src_path, dst_path, force=true)
