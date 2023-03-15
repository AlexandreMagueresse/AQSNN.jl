#################
# ReferenceMesh #
#################
struct ReferenceMesh{T,M,E,P,I} <: AbstractMesh{T,M,E,P}
  polytope::P
  indices::I

  function ReferenceMesh(convexMesh::ConvexMesh{T,M,E,P}) where {T,M,E,P}
    polytope = getPolytope(convexMesh)
    indices = _referenceify(polytope, getCells(convexMesh))
    I = typeof(indices)
    new{T,M,E,P,I}(polytope, indices)
  end

  function ReferenceMesh(polytope::P, indices::I) where {T,M,E,P<:AbstractPolytope{T,M,E},I}
    new{T,M,E,P,I}(polytope, indices)
  end
end

function ReferenceMesh(simplexMesh::SimplexMesh)
  convexMesh = ConvexMesh(simplexMesh)
  ReferenceMesh(convexMesh)
end

function ReferenceMesh(polytope::AbstractPolytope)
  convexMesh = ConvexMesh(polytope)
  ReferenceMesh(convexMesh)
end

referenceify(x) = ReferenceMesh(x)

#############################
# Extension of AbstractMesh #
#############################
getPolytope(mesh::ReferenceMesh) = mesh.polytope
getCells(mesh::ReferenceMesh) = mesh.indices

function getindex(mesh::ReferenceMesh, i::Int)
  polytope = getPolytope(mesh)
  ReferencePolytope(polytope, getindex(getCells(mesh), i))
end

function firstindex(mesh::ReferenceMesh)
  polytope = getPolytope(mesh)
  ReferencePolytope(polytope, firstindex(getCells(mesh)))
end

function lastindex(mesh::ReferenceMesh)
  polytope = getPolytope(mesh)
  ReferencePolytope(polytope, lastindex(getCells(mesh)))
end

#################
# _referenceify #
#################
function _referenceify(polytope::AbstractPolytope{T}, convexes) where {T}
  references = Vector{Tuple{Int64,Vararg{Int64}}}()
  for convex in convexes
    _referenceify!(polytope, references, convex)
  end
  references
end

function _referenceify(polytope::AbstractPolytope{T,2}, convexes; removeAligned::Bool=true, kwargs...) where {T}
  references = Vector{Tuple{Int64,Vararg{Int64}}}()
  for convex in convexes
    _referenceify!(polytope, references, convex)
  end
  references
end

function _referenceify!(::AbstractPolytope, ::Any, ::Any)
  @notimplemented
end

function _referenceify!(::AbstractPolytope{T,0}, references, convex) where {T}
  push!(references, convex)
end

function _referenceify!(::AbstractPolytope{T,1}, references, convex) where {T}
  push!(references, convex)
end

const decompositions = Dict(
  3 => ((1, 2, 3),),
  4 => ((1, 2, 3, 4),),
  5 => ((1, 2, 3, 4), (4, 5, 1)),
  6 => ((1, 2, 3, 4), (4, 5, 6, 1)),
  7 => ((1, 2, 3, 4), (4, 5, 6, 7), (7, 1, 4)),
  8 => ((1, 2, 3, 4), (4, 5, 6, 7), (7, 8, 1, 4)),
  9 => ((1, 2, 3, 4), (4, 5, 6, 7), (7, 8, 9, 1), (1, 4, 7)),
  10 => ((1, 2, 3, 4), (4, 5, 6, 7), (7, 8, 9, 10), (10, 1, 4, 7))
)

function _referenceify!(polytope::AbstractPolytope{T,2}, references, convex) where {T}
  L = length(convex)
  if L < 3
    throw(SingularException(0))
  elseif haskey(decompositions, L)
    for cycle in decompositions[L]
      if length(cycle) == 3
        i, j, k = cycle
        push!(references, (convex[i], convex[j], convex[k]))
      else
        i, j, k, l = cycle
        push!(references, (convex[i], convex[j], convex[k], convex[l]))
      end
    end
  else
    half = div(L + 1, 2)
    _referenceify!(polytope, references, convex[1:half])
    rest = convex[half:end]
    push!(rest, convex[1])
    _referenceify!(polytope, references, rest)
  end
end
