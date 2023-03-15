###############
# SimplexMesh #
###############
struct SimplexMesh{T,M,E,P,I} <: AbstractMesh{T,M,E,P}
  polytope::P
  indices::I

  function SimplexMesh(polytope::P) where {T,M,E,P<:AbstractPolytope{T,M,E}}
    indices = _simplexify(polytope)
    I = typeof(indices)
    new{T,M,E,P,I}(polytope, indices)
  end
end

simplexify(x) = SimplexMesh(x)

#############################
# Extension of AbstractMesh #
#############################
getPolytope(mesh::SimplexMesh) = mesh.polytope
getCells(mesh::SimplexMesh) = mesh.indices

function getindex(mesh::SimplexMesh, i::Int)
  polytope = getPolytope(mesh)
  SimplexPolytope(polytope, getindex(getCells(mesh), i))
end

function firstindex(mesh::SimplexMesh)
  polytope = getPolytope(mesh)
  SimplexPolytope(polytope, firstindex(getCells(mesh)))
end

function lastindex(mesh::SimplexMesh)
  polytope = getPolytope(mesh)
  SimplexPolytope(polytope, lastindex(getCells(mesh)))
end

#########
# Utils #
#########
# https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
# https://www.cs.jhu.edu/~misha/Spring16/05.pdf
mutable struct SimplexData{T}
  index::Int
  angle::T
  isEar::Bool
  isConvex::Bool
end

function markEar!(this, list, polytope)
  this.data.isEar = true
  prev, next = prevnext(list, this)
  prevp = getVertex(polytope, prev.data.index)
  thisp = getVertex(polytope, this.data.index)
  nextp = getVertex(polytope, next.data.index)

  for item in list
    (item == this) && continue
    (item == prev) && continue
    (item == next) && continue
    p = getVertex(polytope, item.data.index)
    if inTriangle(p, prevp, thisp, nextp)
      this.data.isEar = false
      break
    end
  end

  nothing
end

###############
# _simplexify #
###############
function _simplexify(::AbstractPolytope)
  @notimplemented
end

function _simplexify(::AbstractPolytope{T,0}) where {T}
  indices = getIndices(polytope)
  ((indices[1],),)
end

function _simplexify(::AbstractPolytope{T,1}) where {T}
  indices = getIndices(polytope)
  ((indices[1], indices[2]),)
end

function _simplexify(polytope::ConvexPolytope{T,2}) where {T}
  indices = getIndices(polytope)
  [(indices[1], indices[i], indices[i+1]) for i in 2:length(polytope)-1]
end

function _simplexify(polytope::Point)
  indices = getIndices(polytope)
  ((indices[1],),)
end

function _simplexify(polytope::Segment)
  indices = getIndices(polytope)
  ((indices[1], indices[2]),)
end

function _simplexify(polytope::Triangle)
  indices = getIndices(polytope)
  ((indices[1], indices[2], indices[3]),)
end

function _simplexify(polytope::Quadrangle)
  indices = getIndices(polytope)
  ((indices[1], indices[2], indices[3]), (indices[1], indices[3], indices[4]))
end

function _simplexify(polytope::AbstractPolytope{T,2}) where {T}
  N = length(polytope)
  indices = getIndices(polytope)
  u, v = getBasis(polytope)
  ε = eps(T)

  # Initialise circular structure
  list = CircularVector{SimplexData{T}}()
  for i in 1:N
    data = SimplexData{T}(i, zero(T), false, false)
    push!(list, data)
  end

  # Check convexity of vertices and compute cos of angle
  for this in list
    prev, next = prevnext(list, this)
    prevp = getVertex(polytope, prev.data.index)
    thisp = getVertex(polytope, this.data.index)
    nextp = getVertex(polytope, next.data.index)

    this.data.isConvex = convexAngle(prevp, thisp, nextp, u, v)
    this.data.angle = 1 - cosAngle(prevp, thisp, nextp)^2
  end

  # Mark ears separately since we rely on reflex points
  for this in list
    this.data.isConvex && markEar!(this, list, polytope)
  end

  # Extract simplices
  simplices = Vector{Tuple{Int,Int,Int}}()
  sizehint!(simplices, N - 2)
  for _ in 1:N-3
    # Find ear with largest angle
    best = first(list)
    found = best.data.isEar
    for this in list
      if this.data.isEar && (!found || (this.data.angle > best.data.angle))
        best = this
        found = true
      end
    end

    # Extract simplex and remove ear
    prev, next = prevnext(list, best)

    prevp = getVertex(polytope, prev.data.index)
    thisp = getVertex(polytope, best.data.index)
    nextp = getVertex(polytope, next.data.index)

    # Don't add degenerate triangles
    if areaTriangle(prevp, thisp, nextp, u, v) > ε
      push!(simplices, (indices[prev.data.index], indices[best.data.index], indices[next.data.index]))
    end
    delete!(list, best)

    # Update neighbours
    for this in (prev, next)
      prev, next = prevnext(list, this)
      prevp = getVertex(polytope, prev.data.index)
      thisp = getVertex(polytope, this.data.index)
      nextp = getVertex(polytope, next.data.index)
      this.data.angle = 1 - cosAngle(prevp, thisp, nextp)^2

      # If neighbour was not convex, check if it is now
      turnedConvex = false
      if !this.data.isConvex
        this.data.isConvex = convexAngle(prevp, thisp, nextp, u, v)
        turnedConvex = this.data.isConvex
      end

      # If neighbour was an ear or turned convex, check if it is (still) an ear
      if this.data.isEar || turnedConvex
        markEar!(this, list, polytope)
      end
    end
  end

  best = first(list)
  prev, next = prevnext(list, best)

  prevp = getVertex(polytope, prev.data.index)
  thisp = getVertex(polytope, best.data.index)
  nextp = getVertex(polytope, next.data.index)
  # Don't add degenerate triangles
  if areaTriangle(prevp, thisp, nextp, u, v) > ε
    push!(simplices, (indices[prev.data.index], indices[best.data.index], indices[next.data.index]))
  end
  simplices
end
