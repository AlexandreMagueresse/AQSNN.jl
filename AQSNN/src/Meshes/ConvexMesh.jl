##############
# ConvexMesh #
##############
struct ConvexMesh{T,M,E,P,I} <: AbstractMesh{T,M,E,P}
  polytope::P
  indices::I

  function ConvexMesh(mesh::SimplexMesh{T,M,E,P}) where {T,M,E,P}
    polytope = getPolytope(mesh)
    indices = _convexify(polytope, getCells(mesh))
    I = typeof(indices)
    new{T,M,E,P,I}(polytope, indices)
  end
end

function ConvexMesh(polytope::AbstractPolytope)
  mesh = SimplexMesh(polytope)
  ConvexMesh(mesh)
end

convexify(x) = ConvexMesh(x)

#############################
# Extension of AbstractMesh #
#############################
getPolytope(mesh::ConvexMesh) = mesh.polytope
getCells(mesh::ConvexMesh) = mesh.indices

function getindex(mesh::ConvexMesh, i::Int)
  polytope = getPolytope(mesh)
  ConvexPolytope(polytope, getindex(getCells(mesh), i))
end

function firstindex(mesh::ConvexMesh)
  polytope = getPolytope(mesh)
  ConvexPolytope(polytope, firstindex(getCells(mesh)))
end

function lastindex(mesh::ConvexMesh)
  polytope = getPolytope(mesh)
  ConvexPolytope(polytope, lastindex(getCells(mesh)))
end

#########
# Utils #
#########
# https://link.springer.com/chapter/10.1007/3-540-12689-9_105
# https://github.com/ivanfratric/polypartition

mutable struct ConvexData
  id::Int
  indices::Vector{Int}
  length::Int
end

##############
# _convexify #
##############
function _convexify(::AbstractPolytope)
  @notimplemented
end

function _convexify(::AbstractPolytope{T,0}, simplices) where {T}
  simplices
end

function _convexify(::AbstractPolytope{T,1}, simplices) where {T}
  simplices
end

function _convexify(::ConvexPolytope, simplices)
  simplices
end

function _convexify(::Point, simplices)
  simplices
end

function _convexify(::Segment, simplices)
  simplices
end

function _convexify(::Triangle, simplices)
  simplices
end

function _convexify(polytope::Quadrangle, ::Any)
  indices = getIndices(polytope)
  ((indices[1], indices[2], indices[3], indices[4]),)
end

function _convexify(polytope::AbstractPolytope{T,2}, simplices) where {T}
  store = getStore(polytope)
  u, v = getBasis(polytope)
  p₀ = getVertex(polytope, 1)

  # Initialise iterator
  first = CircularNode{ConvexData}()
  a, b, c = simplices[1]
  first.data = ConvexData(1, [a, b, c], 3)
  last = first
  for i in 2:length(simplices)
    next = CircularNode{ConvexData}()
    a, b, c = simplices[i]
    next.data = ConvexData(i, [a, b, c], 3)
    last.next = next
    next.prev = last
    last = next
  end
  first.prev = last
  last.next = first

  # Loop over diagonals
  iter_max = div(length(polytope) * (length(polytope) + 1), 2)
  i₁₋, i₁₊, i₂₋, i₂₊ = -1, -1, -1, -1
  once₁ = false
  poly₁ = first
  while !once₁ || (poly₁.data.id > 1)
    once₁ = true
    i₁₋ = 0
    while i₁₋ < poly₁.data.length
      i₁₋ += 1
      # Select one diagonal [d₋ d₊]
      d₋ = poly₁.data.indices[i₁₋]
      i₁₊ = (i₁₋ == poly₁.data.length) ? 1 : i₁₋ + 1
      d₊ = poly₁.data.indices[i₁₊]

      # Find the other polygon that has [d₋ d₊] as an edge
      isDiagonal = false
      once₂ = false
      poly₂ = poly₁.next
      while !once₂ || (poly₂.data.id > poly₁.data.id)
        once₂ = true
        i₂₊ = 0
        while i₂₊ < poly₂.data.length
          i₂₊ += 1
          (poly₂.data.indices[i₂₊] != d₊) && continue
          i₂₋ = (i₂₊ == poly₂.data.length) ? 1 : i₂₊ + 1
          (poly₂.data.indices[i₂₋] != d₋) && continue
          isDiagonal = true
          break
        end
        isDiagonal && break
        poly₂ = poly₂.next
      end
      iter_max -= 1
      (iter_max < 1) && throw(AssertionError("Polygon probably self-intersecting or contains holes"))
      !isDiagonal && continue

      # Check that both angles are convex
      p = store[d₋]
      s₋, t₋ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      p = store[d₊]
      s₊, t₊ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      s̄₀, Δs₀ = middif(s₋, s₊)
      t̄₀, Δt₀ = middif(t₋, t₊)

      # Intersect (d₋, d₊) with (d₋₁, d₋₂)
      # In poly₁: d₋₁ d₋ (i₁₋) d₊ (i₁₊)
      # In poly₂:     d₊ (i₂₊) d₋ (i₂₋) d₋₂
      d₋₁ = poly₁.data.indices[(i₁₋ == 1) ? poly₁.data.length : i₁₋ - 1]
      d₋₂ = poly₂.data.indices[(i₂₋ == poly₂.data.length) ? 1 : i₂₋ + 1]

      p = store[d₋₁]
      s₋₁, t₋₁ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      p = store[d₋₂]
      s₋₂, t₋₂ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      s̄, Δs = middif(s₋₁, s₋₂)
      t̄, Δt = middif(t₋₁, t₋₂)

      m₁₁, m₁₂, f₁ = Δs, -Δs₀, s̄₀ - s̄
      m₂₁, m₂₂, f₂ = Δt, -Δt₀, t̄₀ - t̄
      α, β, χ = sol2(m₁₁, m₁₂, m₂₁, m₂₂, f₁, f₂)
      χ && continue
      (abs(α) > 1) && continue
      (abs(β) > 1) && continue

      # Intersect (d₋, d₊) with (d₊₁, d₊₂)
      # In poly₁:     d₋ (i₁₋) d₊ (i₁₊) d₊₁
      # In poly₂: d₊₂ d₊ (i₂₊) d₋ (i₂₋)
      d₊₁ = poly₁.data.indices[(i₁₊ == poly₁.data.length) ? 1 : i₁₊ + 1]
      d₊₂ = poly₂.data.indices[(i₂₊ == 1) ? poly₂.data.length : i₂₊ - 1]

      p = store[d₊₁]
      s₋₁, t₋₁ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      p = store[d₊₂]
      s₋₂, t₋₂ = dot₋(p, p₀, u), dot₋(p, p₀, v)
      s̄, Δs = middif(s₋₁, s₋₂)
      t̄, Δt = middif(t₋₁, t₋₂)

      m₁₁, m₁₂, f₁ = Δs, -Δs₀, s̄₀ - s̄
      m₂₁, m₂₂, f₂ = Δt, -Δt₀, t̄₀ - t̄
      α, β, χ = sol2(m₁₁, m₁₂, m₂₁, m₂₂, f₁, f₂)
      χ && continue
      (abs(α) > 1) && continue
      (abs(β) > 1) && continue

      # Append poly₂ to poly₁
      j = i₂₋
      idx = i₁₋ + 1
      while true
        j = (j == poly₂.data.length) ? 1 : j + 1
        (j == i₂₊) && break
        insert!(poly₁.data.indices, idx, poly₂.data.indices[j])
        idx += 1
      end
      poly₁.data.length = length(poly₁.data.indices)

      # Remove poly₂
      poly₂.prev.next = poly₂.next
      poly₂.next.prev = poly₂.prev

      # Keep i₁₋ (will be incremented in the next loop)
      i₁₋ -= 1
    end
    poly₁ = poly₁.next
  end

  convexes = Vector{Vector{Int}}()
  sizehint!(convexes, length(simplices))

  once = false
  poly = poly₁
  while !once || (poly.data.id != poly₁.data.id)
    once = true
    # push!(convexes, getIndices(polytope)[poly.data.indices])
    push!(convexes, poly.data.indices)
    poly = poly.next
  end

  convexes
end
