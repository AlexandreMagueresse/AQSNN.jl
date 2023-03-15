###################
# PolytopeSampler #
###################
mutable struct PolytopeSampler{T,M<:SimplexMesh}
  const mesh::M
  const Σmeasures::Vector{T}
  npoints::Int
  points::Matrix{T}

  function PolytopeSampler(mesh::SimplexMesh, N::Int)
    measures = map(measure, mesh)
    Σmeasures = cumsum(measures)
    Σmeasures ./= Σmeasures[end]

    some = mesh[1]
    T = eltype(some)
    E = embdim(some)
    points = Matrix{T}(undef, E, N)

    M = typeof(mesh)
    sampler = new{T,M}(mesh, Σmeasures, N, points)
    resample!(sampler)
    sampler
  end
end

getMesh(sampler::PolytopeSampler) = sampler.mesh

getSamples(sampler::PolytopeSampler) = sampler.points

#############
# resample! #
#############
function resample!(sampler::PolytopeSampler)
  for col in 1:sampler.npoints
    i = searchsortedfirst(sampler.Σmeasures, rand())
    _resample!(sampler, col, getMesh(sampler)[i])
  end
  nothing
end

function _resample!(::PolytopeSampler, ::Int, ::AbstractPolytope)
  @notimplemented
end

function _resample!(sampler::PolytopeSampler, col::Int, polytope::Point)
  p = getVertex(polytope, 1)
  sampler.points[:, col] = p
  nothing
end

function _resample!(sampler::PolytopeSampler{T}, col::Int, polytope::Segment) where {T}
  p₁, p₂ = getVertex(polytope, 1), getVertex(polytope, 2)
  r = rand(T)
  @. sampler.points[:, col] = (1 - r) * p₁ + r * p₂
  nothing
end

function _resample!(sampler::PolytopeSampler{T}, col::Int, polytope::Triangle) where {T}
  ε = eps(T)
  p₁, p₂, p₃ = getVertex(polytope, 1), getVertex(polytope, 2), getVertex(polytope, 3)
  r₁, r₂, r₃ = -log(rand(T) + ε), -log(rand(T) + ε), -log(rand(T) + ε)
  r = r₁ + r₂ + r₃
  @. sampler.points[:, col] = (r₁ * p₁ + r₂ * p₂ + r₃ * p₃) / r
  nothing
end
