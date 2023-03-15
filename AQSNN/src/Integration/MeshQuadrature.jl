struct MeshQuadrature{T,M,E,R} <: AbstractQuadrature{T,M,E}
  mesh::R
  weights::Vector{T}
  points::Matrix{T}
end

function MeshQuadrature(mesh::ReferenceMesh{T,0,E}, order::Int) where {T,E}
  refquad = ReferenceQuadrature(Point{T,E}, order)

  N = length(mesh)
  P = length(refquad.weights)
  weights, points = Vector{Float32}(undef, N * P), Matrix{Float32}(undef, 1, N * P)
  i = 1
  for cell in mesh
    w, p = transport(refquad, cell)
    for j in 1:P
      weights[i] = w[j]
      points[1, i] = p[1, j]
      i += 1
    end
  end

  MeshQuadrature{T,1,E,typeof(mesh)}(mesh, weights, points)
end

function MeshQuadrature(mesh::ReferenceMesh{T,1,E}, order::Int) where {T,E}
  refquad = ReferenceQuadrature(Segment{T,E}, order)

  N = length(mesh)
  P = length(refquad.weights)
  weights, points = Vector{Float32}(undef, N * P), Matrix{Float32}(undef, 1, N * P)
  i = 1
  for cell in mesh
    w, p = transport(refquad, cell)
    for j in 1:P
      weights[i] = w[j]
      points[1, i] = p[1, j]
      i += 1
    end
  end

  MeshQuadrature{T,1,E,typeof(mesh)}(mesh, weights, points)
end

function MeshQuadrature(mesh::ReferenceMesh{T,2,E}, order::Int) where {T,E}
  refquad = ReferenceQuadrature(Quadrangle{T,E}, order)

  N = length(mesh)
  P = length(refquad.weights)
  weights, points = Vector{Float32}(undef, N * P), Matrix{Float32}(undef, 2, N * P)
  i = 1
  for cell in mesh
    w, p = transport(refquad, cell)
    for j in 1:P
      weights[i] = w[j]
      points[1, i] = p[1, j]
      points[2, i] = p[2, j]
      i += 1
    end
  end

  MeshQuadrature{T,2,E,typeof(mesh)}(mesh, weights, points)
end

##############
# initialise #
##############
initialise!(mq::MeshQuadrature) = nothing

#############
# integrate #
#############
function integrate(integrand::DomainIntegrand, mq::MeshQuadrature)
  F = integrand.f(mq.points)
  sum(mq.weights' .* F)
end

function integrate(integrand::BoundaryIntegrand, mq::MeshQuadrature)
  normal = getNormal(getPolytope(mq.mesh))
  sum(mq.weights' .* integrand.f(normal, mq.points))
end
