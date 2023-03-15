function Boundary(::AbstractPolytope)
  @notimplemented
end

function Boundary(::AbstractPolytope{T,0}) where {T}
  @notreachable
end

function Boundary(polytope::AbstractPolytope{T,1}) where {T}
  E = embdim(polytope)
  p₋, p₊ = getVertex(polytope, 1), getVertex(polytope, 2)
  n = normalize(p₊ .- p₋)
  [
    Point(Context0D{T,E}(-n), getStore(polytope), 1:1),
    Point(Context0D{T,E}(+n), getStore(polytope), 2:2)
  ]
end

function Boundary(polytope::AbstractPolytope{T,2}) where {T}
  E = embdim(polytope)
  L = length(polytope)

  U = typeof(getVertex(polytope, 1))
  N = typeof(getVertex(polytope, 1))
  C = Context1D{T,E,U,N}
  S = typeof(getStore(polytope))
  I = Tuple{Int,Int}
  boundaries = Vector{Segment{T,E,C,S,I}}()
  sizehint!(boundaries, L)

  u, v = getBasis(polytope)

  for i in 1:L
    j = getEdgeEnd(polytope, i)
    p₋, p₊ = getVertex(polytope, i), getVertex(polytope, j)

    d = normalize(p₊ .- p₋)
    n = normalize(dot(d, v) .* u .- dot(d, u) .* v)

    context = Context1D{T,E}(d, n)
    push!(boundaries, Segment(context, getStore(polytope), (i, j)))
  end

  boundaries
end
