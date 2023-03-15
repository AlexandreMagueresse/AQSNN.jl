function CartesianMesh(point::AbstractPolytope{T,0,E}, ::Int) where {T,E}
  ReferenceMesh(point, [getIndices(point)[1]])
end

function CartesianMesh(x₋::T1, x₊::T2, Nx::Int) where {T1,T2}
  T = promote_type(T1, T2, Float32)
  hx = (x₊ - x₋) / Nx
  store = Vector{SVector{1,T}}()
  for i in 0:Nx
    push!(store, SVector{1,T}(x₋ + i * hx))
  end
  context = Context1D{T,1}(SVector{1,T}(1), SVector{2,T}(0, 1))
  polytope = GeneralPolytope(context, store, (1, Nx + 1))
  ReferenceMesh(polytope, [(i, i + 1) for i in 1:Nx-1])
end

function CartesianMesh(segment::AbstractPolytope{T,1,E}, N::Int) where {T,E}
  p₋, p₊ = getVertex(segment, 1), getVertex(segment, 2)
  u, = getBasis(segment)
  dist = norm₋(p₊, p₋)
  h = dist / N
  store = Vector{SVector{E,T}}()
  for i in 0:N
    push!(store, SVector{E,T}(p₋ .+ i * h .* u))
  end
  context = getContext(segment)
  polytope = GeneralPolytope(context, store, (1, N + 1))
  ReferenceMesh(polytope, [(i, i + 1) for i in 1:N-1])
end

function CartesianMesh(x₋::T1, x₊::T2, y₋::T3, y₊::T4, Nx::Int, Ny::Int) where {T1,T2,T3,T4}
  T = promote_type(T1, T2, T3, T4, Float32)
  hx, hy = (x₊ - x₋) / Nx, (y₊ - y₋) / Ny
  store = Vector{SVector{2,T}}()
  for j in 0:Ny
    for i in 0:Nx
      push!(store, SVector{2,T}(x₋ + i * hx, y₋ + j * hy))
    end
  end
  indices_tri = []
  indices_quad = []
  k = 1
  for _ in 1:Ny
    for _ in 1:Nx
      push!(indices_quad, (k, k + 1, k + Nx + 2, k + Nx + 1))
      k += 1
    end
    k += 1
  end

  context = Context2D{T,2}(SVector{2,T}(1, 0), SVector{2,T}(0, 1), SVector{3,T}(0, 0, 1))
  polytope = GeneralPolytope(context, store, (1, Nx + 1, (Nx + 1) * Ny + 1, (Nx + 1) * (Ny + 1)))
  ReferenceMesh(polytope, indices_quad)
end
