###############
# Dimension 1 #
###############
mutable struct Lineariser1{T,E,C} <: AbstractLineariser{T,1,E,C}
  const mesh::C
  const p̄::Vector{T}
  const Δp::Vector{T}

  const roots::Vector{T}
  const slopes::Vector{T}
  const offsets::Vector{T}
  const tolerance::T
  const depth::Int

  abscissae::Vector{T}
  points::Vector{Vector{T}}
  images::Vector{Vector{T}}
end

function Lineariser(
  mesh::C, activation::AbstractActivation, P::Int; depth::Int=-1
) where {T,E,C<:ConvexMesh{T,1,E}}
  # Build direction vector and barycenter
  p₋, p₊ = getVertex(getPolytope(mesh), 1), getVertex(getPolytope(mesh), 2)
  p̄, Δp = (p₊ .+ p₋) ./ 2, (p₊ .- p₋) ./ 2

  # Approximate activation function
  roots, slopes, offsets, tolerance = CPWLise(activation, P)

  # Prepare representation of the elements
  abscissae = Vector{T}()
  points = Vector{Vector{T}}()
  images = Vector{Vector{T}}()

  Lineariser1{T,E,C}(
    mesh, p̄, Δp,
    roots, slopes, offsets, tolerance, depth,
    abscissae, points, images
  )
end

######################
# AbstractLineariser #
######################
getMesh(lin::Lineariser1) = lin.mesh

length(lin::Lineariser1) = length(lin.points) - 1

function getindex(lin::Lineariser1, i::Int)
  polytope = getPolytope(getMesh(lin))
  Segment(getContext(polytope), lin.points, i:i+1)
end

function firstindex(lin::Lineariser1)
  polytope = getPolytope(getMesh(lin))
  Segment(getContext(polytope), lin.points, 1:2)
end

function lastindex(lin::Lineariser1)
  polytope = getPolytope(getMesh(lin))
  l = length(lin)
  Segment(getContext(polytope), lin.points, l:l+1)
end

#################
# Linearisation #
#################
function linearise!(
  u::Sequential, lin::AbstractLineariser{T,1};
  layer::Int=1
) where {T}
  ε = 10 * eps(T)

  #########################
  # Reinitialise elements #
  #########################
  if layer == 1
    empty!(lin.points)
    empty!(lin.images)

    push!(lin.points, lin.p̄ .- lin.Δp, lin.p̄ .+ lin.Δp)
    push!(lin.images, lin.p̄ .- lin.Δp, lin.p̄ .+ lin.Δp)

    empty!(lin.abscissae)
    push!(lin.abscissae, -1, +1)
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

  #################################
  # Intersection with hyperplanes #
  #################################
  D = u.architecture[layer+1]
  p̄, Δp = lin.p̄, lin.Δp

  # Look at [ts[i₋], ts[i₊]] = [t₋, t₊] = {t̄ + α Δt; -1 ≤ α ≤ +1}
  i₋, i₊ = 1, 2
  t₋, t₊ = lin.abscissae[i₋], lin.abscissae[i₊]
  t̄, Δt = (t₊ + t₋) / 2, (t₊ - t₋) / 2

  # Loop through each element
  while i₊ <= length(lin.abscissae)
    # Select images
    u₋, u₊ = lin.images[i₋], lin.images[i₊]

    # Loop through each output dimension
    for d in 1:D
      # Parameterise u[d] on [t₋, t₊] as
      # u(t̄ + α Δt)[d] = ū[d] + α Δu[d]
      ū, Δu = (u₊[d] + u₋[d]) / 2, (u₊[d] - u₋[d]) / 2

      # Skip if output is (almost) constant
      (abs(Δu) < ε) && continue

      # Filter roots in range [u₋[d], u₊[d]]
      rs = intersectSorted(lin.roots, u₋[d], u₊[d])
      for r in rs
        # Solve u(t̄ + α Δt)[d] = ξ
        α = (lin.roots[r] - ū) / Δu
        t = t̄ + α * Δt

        # Find index of t in ts between t₋ and t₊
        idx = searchsorted(lin.abscissae, t)

        # Skip if the point already exists
        if isempty(idx)
          # Compute point and image
          point = p̄ .+ t .* Δp
          image = ((1 - α) .* u₋ .+ (1 + α) .* u₊) ./ 2

          # Move away from the ends: they cannot change
          idx = min(max(idx.start, i₋), i₊)

          # Add to elements
          insert!(lin.points, idx, point)
          insert!(lin.images, idx, image)
          insert!(lin.abscissae, idx, t)
        end
        i₊ += 1
      end
    end

    # Move to next element
    i₋, i₊ = i₊, i₊ + 1
    (i₊ > length(lin.abscissae)) && break

    t₋, t₊ = t₊, lin.abscissae[i₊]
    t̄, Δt = (t₊ + t₋) / 2, (t₊ - t₋) / 2
  end

  ####################
  # Apply activation #
  ####################
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
