using Gridap
using GridapGmsh
using WriteVTK
using AQSNN

include("0.utils.jl")

############
# Settings #
############
Ωname = "rhombi"
Fname = "xpy"

Ω = Ωs[Ωname]
ρname = ρsfs[Fname]
ρ = ρs[ρname]

# Load MC and AQ models
folder = joinpath("data", "rhombi/rhombi_poissonNitsche_xpy_ReLUε")
uMC, dictMC = AQSNN.load(joinpath(folder, "MC", "0.01_10000_100.0_10_6000_300_1.jld2"), Ω, ρ)
uAQ, dictAQ = AQSNN.load(joinpath(folder, "AQ", "0.01_10000_100.0_10_3_2_0.0_3_1.jld2"), Ω, ρ)

################
# FEM Solution #
################
# Import mesh
if Ωname == "cartesian2d"
  model = GmshDiscreteModel(joinpath("data", "meshes", "square.msh"))
elseif Ωname == "rhombi"
  model = GmshDiscreteModel(joinpath("data", "meshes", "rhombi.msh"))
else
  throw("Unknown domain")
end

# Define forcing term
if Fname == "well"
  function g(x)
    α = 10
    tanh(α * (x[1]^2 + x[2]^2 - 0.5^2))
  end

  function f(x)
    α = 10
    u = g(x)
    -(4 * α * (1 - 2 * α * (x[1]^2 + x[2]^2) * u) * (1 - u^2))
  end
elseif Fname == "sinc"
  function g(x)
    α = 2
    sinc(α * x[1]) * sinc(α * x[2])
  end

  function f(x)
    α = 2
    -(∂²sinc(α, x[1]) * sinc(α * x[2]) + sinc(α * x[1]) * ∂²sinc(α, x[2]))
  end
elseif Fname == "xpy"
  function g(x)
    0
  end

  function f(x)
    x[1] + x[2]
  end
else
  throw("Unknown forcing term")
end

# FE space
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)

V = TestFESpace(model, reffe, dirichlet_tags="boundary")
U = TrialFESpace(V, g)

Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)

# Definition of the PDE
a(u, v) = ∫(Gridap.∇(v) ⋅ Gridap.∇(u)) * dΩ
l(v) = ∫(f * v) * dΩ
op = AffineFEOperator(a, l, U, V)

# Solve PDE with FEM
uh = solve(op)

# Evaluate solution at mesh nodes
npoints = length(model.grid.node_coordinates)
points = Matrix{Float64}(undef, 2, npoints)
for i in 1:npoints
  points[1, i] = model.grid.node_coordinates[i].data[1]
  points[2, i] = model.grid.node_coordinates[i].data[2]
end
cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, ids) for ids in model.grid.cell_node_ids]

###########################
# Evaluation of the error #
###########################
dΩexp = Measure(Ω, 10)
∫ex = sqrt(sum(∫(uh * uh) * dΩexp))

uuMC(x) = uMC([x[1]; x[2]])[1]
eMC = uh - uuMC
l2MC = sqrt(sum(∫(eMC * eMC) * dΩexp)) / ∫ex
println(l2MC)

uuAQ(x) = uAQ([x[1]; x[2]])[1]
eAQ = uh - uuAQ
l2AQ = sqrt(sum(∫(eAQ * eAQ) * dΩexp)) / ∫ex
println(l2AQ)

########
# Save #
########
vtk_grid(joinpath("data", "FEM_MC_AQ"), points, cells) do vtk
  vtk["FEM"] = evaluate(uh, [Gridap.Point(points[1, i], points[2, i]) for i in 1:npoints])
  vtk["MC"] = uMC(points)[1, :]
  vtk["AQ"] = uAQ(points)[1, :]
end
