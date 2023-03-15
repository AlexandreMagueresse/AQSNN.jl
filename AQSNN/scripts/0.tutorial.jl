using AQSNN
using StaticArrays

using BenchmarkTools
using Random

T = Float32
SV1 = SVector{1,T}
SV2 = SVector{2,T}
SV3 = SVector{3,T}

ε = 1.0f-4

shapes = Dict(
  "point1D" => Domain(Val(0), [SV1(1)]),
  "point2D" => Domain(Val(0), [SV2(1, 0)]),
  "point3D" => Domain(Val(0), [SV3(1, 0, 0)]),
  "line1D" => Domain(Val(1), [SV1(-1), SV1(+1)]),
  "line2D" => Domain(Val(1), [SV2(1, 0), SV2(0, 1)]),
  "line3D" => Domain(Val(1), [SV3(1, 0, 0), SV3(0, 0, 1)]),
  "tri2D" => Domain(Val(2), [SV2(0, 0), SV2(1, 0), SV2(0, 1)]),
  "tri3D" => Domain(Val(2), [SV3(1, 0, 0), SV3(0, 1, 0), SV3(0, 0, 1)]),
  "quad2D" => Domain(Val(2), [SV2(-1, -1), SV2(+1, -1), SV2(+1, +1), SV2(-1, +1)]),
  "quad3D" => Domain(Val(2), [SV3(-1, -1, +1), SV3(+1, -1, +1), SV3(+1, +1, -1), SV3(-1, +1, -1)]),
  "circle2D" => RegularDomain(1.0f0, 100),
  "ill" => Domain(Val(2), [SV2(-1, 0), SV2(-1 + ε, ε), SV2(+1, 0), SV2(-1 + ε, -ε)]),
  "star2D" => Domain(Val(2), [SV2(-1, -1), SV2(0, -0.5), SV2(+1, -1), SV2(0.5, 0), SV2(+1, +1), SV2(0, 0.5), SV2(-1, +1), SV2(-0.5, 0)]),
  "star3D" => Domain(Val(2), [SV3(0, 0, 1), SV3(0.4, 0.2, 0.4), SV3(1, 0, 0), SV3(0.4, 0.4, 0.2), SV3(0, 1, 0), SV3(0.2, 0.4, 0.4)]),
  "bat" => Domain(Val(2), [SV2(0, 0), SV2(-1, -1), SV2(-3, +1), SV2(-1, +1), SV2(-1, +2), SV2(0, 1), SV2(+1, +2), SV2(+1, +1), SV2(+3, +1), SV2(+1, -1)]),
)

############
# Geometry #
############
for name in sort(collect(keys(shapes)))
  Ω = shapes[name]

  if mandim(Ω) > 0
    Γ = Boundary(Ω)
    plotPolytope(Ω, Γ)
  else
    plotPolytope(Ω)
  end

  τs = SimplexMesh(Ω)
  κs = ConvexMesh(τs)
  ρs = ReferenceMesh(κs)

  plotMesh(τs)
  plotMesh(κs)
  plotMesh(ρs)
end

###############
# Integration #
###############
function f(ps)
  if size(ps, 1) == 1
    X = view(ps, 1:1, :)
    X .^ 0
  elseif size(ps, 1) == 2
    X = view(ps, 1:1, :)
    Y = view(ps, 2:2, :)
    X .^ 0 .* Y .^ 0
  elseif size(ps, 1) == 3
    X = view(ps, 1:1, :)
    Y = view(ps, 2:2, :)
    Z = view(ps, 3:3, :)
    X .^ 0 .* Y .^ 0 .* Z .^ 0
  end
end

for name in sort(collect(keys(shapes)))
  Ω = shapes[name]
  !isReference(Ω) && continue
  Ωτ = SimplexMesh(Ω)

  dΩmc = MonteCarloQuadrature(Ωτ, 1000)
  rq = ReferenceQuadrature(Ω, 10)
  w, p = transport(rq, Ω)

  ∫mc = ∫Ω(f) * dΩmc
  ∫rq = sum(w' .* f(p))

  println(name, " ", ∫mc, " ", ∫rq, " ", abs(∫mc - ∫rq))
end

#################
# Visualisation #
#################
for name in sort(collect(keys(shapes)))
  Ω = shapes[name]

  E = embdim(Ω)
  ρ = Tanh()
  architecture = [E, 10, 1]
  u = Sequential{T,E}(architecture, ρ)

  Ωτ = SimplexMesh(Ω)
  plotModel(u, Ωτ, N=5000, showGrad=false)
  plotBasis(u, Ωτ, N=5000, showGrad=false)

  if mandim(Ω) > 0
    Γ = Boundary(Ω)
    Γτs = SimplexMesh.(Γ)
    plotModel(u, Ωτ, Γτs, N=1000, showGrad=false)
    plotBasis(u, Ωτ, Γτs, N=1000, showGrad=false)
  end
end

#################
# Linearisation #
#################
for name in sort(collect(keys(shapes)))
  Random.seed!(1)
  Ω = shapes[name]

  E = embdim(Ω)
  # ρ = ReLU()
  ρ = Tanh()
  architecture = [E, 10, 10, 1]
  u = Sequential{T,E}(
    architecture, ρ,
    weightInitialiser=UniformInitialiser{T}(-2, +2),
    biasInitialiser=UniformInitialiser{T}(-1, +1)
  )

  Ωτ = SimplexMesh(Ω)
  Ωκ = ConvexMesh(Ωτ)

  P = 2
  lin = Lineariser(Ωκ, ρ, P)
  linearise!(u, lin)

  plotLineariser(lin)
  plotProjection(u, Ωτ, lin, N=1000)
  collapse!(lin)
  plotLineariser(lin)
  plotProjection(u, Ωτ, lin, N=1000)

  dΩmc = MonteCarloQuadrature(Ωτ, 1000)
  dΩaq = AdaptiveQuadrature(u, lin, 3, 0.0f0, false)
  println(dΩaq.npoints)

  ∫mc = ∫Ω(u) * dΩmc
  ∫aq = ∫Ω(u) * dΩaq

  println(∫mc, " ", ∫aq)

  if mandim(Ω) > 0
    Γ = Boundary(Ω)
    Γτs = SimplexMesh.(Γ)
    Γκs = ConvexMesh.(Γτs)

    P = 2
    lins = Lineariser.(Γκs, ρ, P)
    linearise!(u, lins)

    plotModel(u, Ωτ, Γτs, N=1000, showGrad=false)
    plotLineariser(lins)
    plotProjection(u, Ωτ, Γτs, lins)
  end
end
