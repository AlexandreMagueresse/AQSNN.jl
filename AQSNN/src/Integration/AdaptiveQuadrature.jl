mutable struct AdaptiveQuadrature{T,M,E,S,L,Q} <: AbstractQuadrature{T,M,E}
  const model::S
  const lineariser::L
  const quadratures::Q
  const collapseRate::T
  const useSimplex::Bool

  weights::Vector{T}
  points::Matrix{T}
  const npoints::Vector{Int}
end

const manifold2references = Dict(
  0 => (Point,),
  1 => (Segment,),
  2 => (Triangle, Quadrangle),
)

function AdaptiveQuadrature(
  model::S, lin::L, order::Int, collapseRate::T=T(0.2), useSimplex::Bool=false
) where {S<:Sequential,T,M,E,L<:AbstractLineariser{T,M,E}}
  quadratures = Dict(
    Symbol(Reference) => ReferenceQuadrature(Reference{T,M,E}, order)
    for Reference in manifold2references[M]
  )

  weights = Vector{T}(undef, 0)
  points = Matrix{T}(undef, E, 0)
  npoints = Vector{Int}()

  Q = typeof(quadratures)
  ai = AdaptiveQuadrature{T,M,E,S,L,Q}(
    model, lin, quadratures,
    collapseRate, useSimplex,
    weights, points, npoints
  )
  initialise!(ai)
  ai
end

getOrder(aq::AdaptiveQuadrature) = getOrder(first(values(aq.quadratures)))
getWeights(aq::AdaptiveQuadrature) = aq.weights
getPoints(aq::AdaptiveQuadrature) = aq.points

##############
# initialise #
##############
function initialise!(aq::AdaptiveQuadrature{T}) where {T}
  linearise!(aq.model, aq.lineariser)
  if aq.collapseRate > 0
    collapse!(aq.lineariser, aq.collapseRate)
  end

  weights = Vector{Vector{T}}()
  points = Vector{Matrix{T}}()
  for cell in aq.lineariser
    subcells = SimplexMesh(cell)
    for subcell in subcells
      rq = aq.quadratures[typeof(subcell).name.name]
      w, p = transport(rq, subcell)
      push!(weights, w)
      push!(points, p)
    end
  end

  aq.weights = vcat(weights...)
  aq.points = hcat(points...)
  push!(aq.npoints, length(aq.weights))
end

function initialise!(aq::AdaptiveQuadrature{T,2}) where {T}
  try
    linearise!(aq.model, aq.lineariser)
  catch
    println("Linearisation failed")
    return nothing
  end

  if aq.collapseRate > 0
    try
      collapse!(aq.lineariser, aq.collapseRate)
    catch
      println("Collapsing failed")
      return nothing
    end
  end

  weights = Vector{Vector{T}}()
  points = Vector{Matrix{T}}()
  ε = 10 * eps(T)
  i = 0
  for cell in aq.lineariser
    i += 1

    # Remove aligned points
    store = getStore(cell)
    indices = getIndices(cell)
    N = length(indices)

    p₀ = store[indices[1]]
    u, v = getBasis(cell)
    subindices = [indices[1]]

    i₋, s₋, t₋ = 1, zero(T), zero(T)
    p = store[indices[2]]
    i₀, s₀, t₀ = indices[2], dot₋(p, p₀, u), dot₋(p, p₀, v)
    for i in 3:N+1
      (i == N + 1) && (i = 1)
      p = store[indices[i]]
      i₊, s₊, t₊ = indices[i], dot₋(p, p₀, u), dot₋(p, p₀, v)

      # Check whether (p₋ - p₀) and (p₊ - p₀) are collinear
      # (s, t)₀ = (s̄, t̄) + α (Δs, Δt)
      # Project on (u, v)
      # Δs (t₀ - t̆) - Δt (s₀ - s̄) = 0
      s̄, Δs = middif(s₋, s₊)
      t̄, Δt = middif(t₋, t₊)

      if abs(det2(Δs, Δt, s₀ - s̄, t₀ - t̄)) < ε
        i₀, s₀, t₀ = i₊, s₊, t₊
      else
        push!(subindices, i₀)
        i₋, s₋, t₋ = i₀, s₀, t₀
        i₀, s₀, t₀ = i₊, s₊, t₊
      end
    end

    # Check first point
    if length(subindices) > 3
      i₋, i₀, i₊ = subindices[end], subindices[1], subindices[2]
      p = store[i₋]
      s₋, t₋ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      p = store[i₀]
      s₀, t₀ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      p = store[i₊]
      s₊, t₊ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      s̄, Δs = middif(s₋, s₊)
      t̄, Δt = middif(t₋, t₊)
      if abs(det2(Δs, Δt, s₀ - s̄, t₀ - t̄)) < ε
        popfirst!(subindices)
      end
    end

    if length(subindices) > 2
      cell = ConvexPolytope(cell, subindices)
    end

    try
      subcells = aq.useSimplex ? SimplexMesh(cell) : ReferenceMesh(cell)
      for subcell in subcells
        rq = aq.quadratures[typeof(subcell).name.name]
        w, p = transport(rq, subcell)
        push!(weights, w)
        push!(points, p)
      end
    catch
      continue
    end
  end

  aq.weights = vcat(weights...)
  aq.points = hcat(points...)
  push!(aq.npoints, length(aq.weights))
end

#############
# integrate #
#############
function integrate(integrand::DomainIntegrand, aq::AdaptiveQuadrature)
  F = integrand.f(aq.points)
  sum(aq.weights' .* F)
end

function integrate(integrand::BoundaryIntegrand, aq::AdaptiveQuadrature)
  normal = getNormal(getPolytope(getMesh(aq.lineariser)))
  sum(aq.weights' .* integrand.f(normal, aq.points))
end
