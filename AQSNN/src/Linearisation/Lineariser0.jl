###############
# Dimension 0 #
###############
mutable struct Lineariser0{T,E,C} <: AbstractLineariser{T,0,E,C}
  const mesh::C

  const roots::Vector{T}
  const slopes::Vector{T}
  const offsets::Vector{T}
  const tolerance::T
  const depth::Int

  points::Vector{Vector{T}}
  images::Vector{Vector{T}}
end

function Lineariser(
  mesh::C, activation::AbstractActivation, P::Int; depth::Int=-1
) where {T,E,C<:ConvexMesh{T,0,E}}
  # Approximate activation function
  roots, slopes, offsets, tolerance = CPWLise(activation, P)

  # Prepare representation of the elements
  p̄ = getVertex(getPolytope(mesh), 1)
  points = [copy(p̄)]
  images = [copy(p̄)]

  Lineariser0{T,E,C}(mesh, roots, slopes, offsets, tolerance, depth, points, images)
end

######################
# AbstractLineariser #
######################
getMesh(lin::Lineariser0) = lin.mesh

length(::Lineariser0) = 1

function getindex(lin::Lineariser0, i::Int)
  polytope = getPolytope(getMesh(lin))
  Point(getContext(polytope), lin.points, i:i)
end

function firstindex(lin::Lineariser0)
  polytope = getPolytope(getMesh(lin))
  Point(getContext(polytope), lin.points, 1:1)
end

function lastindex(lin::Lineariser0)
  polytope = getPolytope(getMesh(lin))
  Point(getContext(polytope), lin.points, 1:1)
end

#################
# Linearisation #
#################
function linearise!(
  u::Sequential, lin::AbstractLineariser{T,0};
  layer::Int=1
) where {T}
  #########################
  # Reinitialise elements #
  #########################
  if layer == 1
    @. lin.images[1] = lin.points[1]
  end

  ####################
  # Apply linear map #
  ####################
  W, b = weight(u, layer), bias(u, layer)
  for k in eachindex(lin.images)
    lin.images[k] = W * lin.images[k] .+ b
  end

  ######################
  # Stop at last layer #
  ######################
  if (layer == lin.depth) || (lin.depth == -1 && layer == length(u.architecture) - 1)
    return
  end

  ####################
  # Apply activation #
  ####################
  D = u.architecture[layer+1]
  for k in eachindex(lin.images)
    for d in 1:D
      # Select piece
      x = lin.images[k][d]
      i = searchsortedfirst(lin.roots, x)

      # Retrieve slope and offset
      a = lin.slopes[i]
      b = lin.offsets[i]

      # Apply linear map
      lin.images[k][d] = a * x + b
    end
  end

  ######################
  # Move to next layer #
  ######################
  linearise!(u, lin; layer=layer + 1)
end
