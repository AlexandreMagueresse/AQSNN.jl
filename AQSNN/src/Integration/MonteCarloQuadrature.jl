##############################
# DomainMonteCarloQuadrature #
##############################
struct DomainMonteCarloQuadrature{T,M,E,S} <: AbstractQuadrature{T,M,E}
  sampler::S
  measure::T
end

function MonteCarloQuadrature(mesh::SimplexMesh{T,M,E}, N::Int) where {T,M,E}
  sampler = PolytopeSampler(mesh, N)
  meas = measure(getPolytope(mesh))
  S = typeof(sampler)
  DomainMonteCarloQuadrature{T,M,E,S}(sampler, meas)
end

################################
# BoundaryMonteCarloQuadrature #
################################
struct BoundaryMonteCarloQuadrature{T,M,E,S} <: AbstractQuadrature{T,M,E}
  npoints::Int
  npointsEach::Vector{Int}
  samplers::S

  measures::Vector{T}
  Σmeasures::Vector{T}
end

function MonteCarloQuadrature(meshes::AbstractVector{<:SimplexMesh{T,M,E}}, N::Int) where {T,M,E}
  npointsEach = zeros(Int, length(meshes))
  samplers = [PolytopeSampler(mesh, 1) for mesh in meshes]

  measures = [measure(getPolytope(mesh)) for mesh in meshes]
  Σmeasures = cumsum(measures)
  Σmeasures ./= Σmeasures[end]

  S = typeof(samplers)
  bmcq = BoundaryMonteCarloQuadrature{T,M,E,S}(N, npointsEach, samplers, measures, Σmeasures)
  initialise!(bmcq)
  bmcq
end

##############
# initialise #
##############
function initialise!(mc::DomainMonteCarloQuadrature)
  resample!(mc.sampler)
  nothing
end

function initialise!(::BoundaryMonteCarloQuadrature{T,0}) where {T}
  nothing
end

function initialise!(mc::BoundaryMonteCarloQuadrature{T,1,E}) where {T,E}
  mc.npointsEach .*= 0
  for _ in 1:mc.npoints
    i = searchsortedfirst(mc.Σmeasures, rand())
    mc.npointsEach[i] += 1
  end

  for (sampler, n) in zip(mc.samplers, mc.npointsEach)
    sampler.npoints = n
    sampler.points = Matrix{T}(undef, E, n)
    resample!(sampler)
  end
  nothing
end

####################
# integrate domain #
####################
function integrate(integrand::DomainIntegrand, mc::DomainMonteCarloQuadrature)
  points = getSamples(mc.sampler)
  mc.measure * mean(integrand.f(points))
end

######################
# integrate boundary #
######################
# old version with given number on each edge
function integrate(integrand::BoundaryIntegrand, mc::DomainMonteCarloQuadrature)
  normal = getNormal(getPolytope(getMesh(mc.sampler)))
  points = getSamples(mc.sampler)
  mc.measure * mean(integrand.f(normal, points))
end

function integrate(integrand::BoundaryIntegrand, mc::BoundaryMonteCarloQuadrature{T}) where {T}
  ∫ = zero(T)
  for (meas, sampler) in zip(mc.measures, mc.samplers)
    if sampler.npoints > 0
      normal = getNormal(getPolytope(getMesh(sampler)))
      points = getSamples(sampler)
      ∫ += meas * mean(integrand.f(normal, points))
    end
  end
  ∫
end
