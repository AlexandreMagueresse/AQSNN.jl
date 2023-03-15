###############
# Dimension 2 #
###############
mutable struct Lineariser2{T,E,C} <: AbstractLineariser{T,2,E,C}
  const mesh::C

  const roots::Vector{T}
  const slopes::Vector{T}
  const offsets::Vector{T}
  const tolerance::T
  const depth::Int

  points::Vector{Vector{T}}
  images::Vector{Vector{T}}
  cycles::Vector{Vector{Int}}
  adjacency::Dict{Tuple{Int,Int},Tuple{Bool,Int,Int}}
end

function Lineariser(
  mesh::C, activation::AbstractActivation, P::Int; depth::Int=-1
) where {T,E,C<:ConvexMesh{T,2,E}}
  # Approximate activation function
  roots, slopes, offsets, tolerance = CPWLise(activation, P)

  # Prepare representation of the elements
  points = Vector{Vector{T}}()
  images = Vector{Vector{T}}()
  cycles = Vector{Vector{Int}}()
  adjacency = Dict{Tuple{Int,Int},Tuple{Bool,Int,Int}}()

  Lineariser2{T,E,C}(
    mesh,
    roots, slopes, offsets, tolerance, depth,
    points, images, cycles, adjacency
  )
end

######################
# AbstractLineariser #
######################
getMesh(lin::Lineariser2) = lin.mesh

length(lin::Lineariser2) = length(lin.cycles)

function getindex(lin::Lineariser2, i::Int)
  context = getContext(getPolytope(getMesh(lin)))
  cycle = getindex(lin.cycles, i)
  ConvexPolytope(context, lin.points, cycle)
end

function firstindex(lin::Lineariser2)
  context = getContext(getPolytope(getMesh(lin)))
  cycle = firstindex(lin.cycles)
  ConvexPolytope(context, lin.points, cycle)
end

function lastindex(lin::Lineariser2)
  context = getContext(getPolytope(getMesh(lin)))
  cycle = lastindex(lin.cycles)
  ConvexPolytope(context, lin.points, cycle)
end

#################
# Linearisation #
#################
function linearise!(
  u::Sequential, lin::AbstractLineariser{T,2,E};
  layer::Int=1
) where {T,E}
  ε = 10 * eps(T)
  ε₂ = 10 * eps(T)

  #########################
  # Reinitialise elements #
  #########################
  if layer == 1
    polytope = getPolytope(getMesh(lin))
    L = length(polytope)

    empty!(lin.points)
    empty!(lin.images)
    for i in 1:length(polytope)
      vertex = getVertex(polytope, i)
      push!(lin.points, vertex)
      push!(lin.images, vertex)
    end

    empty!(lin.cycles)
    empty!(lin.adjacency)
    for (c, convex) in enumerate(getMesh(lin).indices)
      # Add original convex decomposition
      if isa(convex, Vector)
        push!(lin.cycles, copy(convex))
      else
        push!(lin.cycles, [copy(cvx) for cvx in convex])
      end

      # Mark boundary edges
      for (k₋, k₊) in CircularIterator(convex)
        k₋, k₊ = minmax(k₋, k₊)
        isBoundary = (k₊ == k₋ + 1) || (k₋ == 1 && k₊ == L)
        lin.adjacency[(k₋, k₊)] = (isBoundary, c, -1)
      end
    end
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
  # Initialise graph #
  ####################
  N = length(lin.points)
  graph = Graph{T}()

  nNodes = 0
  for _ in 1:N
    nNodes += 1
    addNode!(graph, nNodes)
  end

  nLines = 0
  edges = Dict{Tuple{Int,Int},Line{T}}()
  for (n₋, n₊) in keys(lin.adjacency)
    p₋, p₊ = lin.points[n₋], lin.points[n₊]
    p̄, Δp = (p₊ .+ p₋) / 2, (p₊ .- p₋) / 2

    nLines += 1
    edges[minmax(n₋, n₊)] = Line{T}(nLines, p̄, Δp, T[-1, +1], [n₋, n₊])
  end

  ###################
  # Intersect edges #
  ###################
  D = u.architecture[layer+1]
  R = length(lin.roots)
  α, β = getBasis(getPolytope(getMesh(lin)))

  # When (n₋, n₊) is an edge, diag2ends[(n₋, n₊)] is a dictionary
  # keys: hyperplane (output dimension and root) and
  # values: indiex of the point at the intersection of hyperplane and (n₋, n₊)
  diag2ends = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Int}}()

  # Loop through each edge
  for (n₋, n₊) in keys(lin.adjacency)
    edge = edges[(n₋, n₊)]
    p̄, Δp = edge.p̄, edge.Δp
    nΔp = norm(Δp)

    # This will be diag2ends[(n₋, n₊)]
    dict = Dict{Tuple{Int,Int},Int}()

    # Loop through each output dimension
    for d in 1:D
      # Parameterise u[d] on [p₋, p₊] as
      # u(p̄ + t Δp)[d] = ū[d] + t Δu[d]
      u₋, u₊ = lin.images[n₋][d], lin.images[n₊][d]
      ū, Δu = (u₊ + u₋) / 2, (u₊ - u₋) / 2
      (n₋ > n₊) && (Δu *= -1)

      # Skip if output is (almost) constant
      (abs(Δu) < ε) && continue

      # Filter roots in range [u₋[d], u₊[d]]
      rs = intersectSorted(lin.roots, u₋, u₊)
      for r in rs
        # Solve u(p̄ + t Δp)[d] = ξ
        t = (lin.roots[r] - ū) / Δu

        # Find index of t in ts between t₋ and t₊
        ts, ps = edge.abscissae, edge.points
        rg = searchsorted(ts, t)

        # Create point or retrieve index if the point already exists
        if !isempty(rg)
          nNode = ps[rg.start]
        elseif (rg.start >= 1) && nΔp * abs(ts[rg.start] - t) < ε₂
          nNode = ps[rg.start]
        elseif (rg.stop <= length(ts)) && nΔp * abs(ts[rg.stop] - t) < ε₂
          nNode = ps[rg.stop]
        else
          # Compute point and image
          point = p̄ + t * Δp
          image = ((1 - t) * lin.images[n₋] + (1 + t) * lin.images[n₊]) ./ 2

          push!(lin.points, point)
          push!(lin.images, image)
          nNodes += 1

          nNode = nNodes
          addNode!(graph, nNode)

          # Move away from the ends: they cannot change
          idx = min(max(2, rg.start), length(ts))

          # Add to elements
          insert!(ts, idx, t)
          insert!(ps, idx, nNode)
        end

        dict[(d, r)] = nNode
      end
    end

    diag2ends[(n₋, n₊)] = dict
  end

  ######################
  # Add edges to graph #
  ######################
  adjacency = Dict{Tuple{Int,Int},Tuple{Bool,Int,Int}}()

  for line in keys(edges)
    # Propagate the belonging to the boundary
    isBoundary = lin.adjacency[line][1]
    for (p₋, p₊) in partition(edges[line].points, 2, 1)
      addEdge!(graph, p₋, p₊, edges[line].number, isBoundary)
      adjacency[minmax(p₋, p₊)] = (isBoundary, -1, -1)
    end
  end

  #######################
  # Intersect diagonals #
  #######################
  # Loop through each element
  for cycle in lin.cycles
    L = length(cycle)

    # Initialise diagonals for the element
    nDiagonals = 0
    diagonals = Vector{Line{T}}()
    groups = Vector{Vector{Int}}()

    ######################################
    # Build groups of parallel diagonals #
    ######################################
    # Loop through each output dimension
    for d in 1:D
      group = Vector{Int}()
      # Loop through each root
      for r in 1:R
        # Find the two indices where the hyperplane (d, r) sliced the element
        n₋, n₊ = -1, -1
        for (k₋, k₊) in CircularIterator(cycle)
          k = minmax(k₋, k₊)
          n = get(diag2ends[k], (d, r), -1)
          (n == -1) && continue
          if n₋ == -1
            n₋ = n
          else
            n₊ = n
          end
        end
        # Skip if not found
        (n₋ == -1 || n₊ == -1) && continue

        # Parameterise diagonal
        p₋, p₊ = lin.points[n₋], lin.points[n₊]
        p̄, Δp = (p₊ .+ p₋) / 2, (p₊ .- p₋) / 2

        # Add diagonal
        nLines += 1
        nDiagonals += 1
        push!(diagonals, Line{T}(nLines, p̄, Δp, T[-1, +1], [n₋, n₊]))
        push!(group, nDiagonals)
      end
      push!(groups, group)
    end

    #################################
    # Intersect group against group #
    #################################
    # Loop through each pair of diagonals with different direction
    # They can still be parallel (corner case)
    for d₁ in 1:D, d₂ in (d₁+1):D
      for l₁ in groups[d₁], l₂ in groups[d₂]
        # Retrieve direction vectors for both diagonals
        p̄₁, Δp₁ = diagonals[l₁].p̄, diagonals[l₁].Δp
        p̄₂, Δp₂ = diagonals[l₂].p̄, diagonals[l₂].Δp
        u₁₋, u₁₊ = lin.images[diagonals[l₁].points[1]], lin.images[diagonals[l₁].points[end]]
        u₂₋, u₂₊ = lin.images[diagonals[l₂].points[1]], lin.images[diagonals[l₂].points[end]]
        nΔp₁, nΔp₂ = norm(Δp₁), norm(Δp₂)

        # Build system
        m₁₁, m₁₂, f₁ = dot(Δp₁, α), -dot(Δp₂, α), dot₋(p̄₂, p̄₁, α)
        m₂₁, m₂₂, f₂ = dot(Δp₁, β), -dot(Δp₂, β), dot₋(p̄₂, p̄₁, β)

        # Solve system
        t₁, t₂, χ = sol2(m₁₁, m₁₂, m₂₁, m₂₂, f₁, f₂)

        # t₁, t₂ = -1, -1
        # try
        #   t₁, t₂ = [m₁₁ m₁₂; m₂₁ m₂₂] \ [f₁, f₂]
        # catch
        #   continue
        # end

        # Skip if parallel or intersection is not within both lines
        χ && continue
        (abs(t₁) > 1) && continue
        (abs(t₂) > 1) && continue

        # Check whether a close point already exists on either diagonal
        ts₁ = diagonals[l₁].abscissae
        rg₁ = searchsorted(ts₁, t₁)
        id₁ = -1
        if !isempty(rg₁)
          id₁ = rg₁.start
        elseif ((rg₁.start >= 1) && nΔp₁ * abs(ts₁[rg₁.start] - t₁) < ε₂)
          id₁ = rg₁.start
        elseif ((rg₁.stop <= length(ts₁)) && nΔp₁ * abs(ts₁[rg₁.stop] - t₁) < ε₂)
          id₁ = rg₁.stop
        end

        ts₂ = diagonals[l₂].abscissae
        rg₂ = searchsorted(ts₂, t₂)
        id₂ = -1
        if !isempty(rg₂)
          id₂ = rg₂.start
        elseif ((rg₂.start >= 1) && nΔp₂ * abs(ts₂[rg₂.start] - t₂) < ε₂)
          id₂ = rg₂.start
        elseif ((rg₂.stop <= length(ts₂)) && nΔp₂ * abs(ts₂[rg₂.stop] - t₂) < ε₂)
          id₂ = rg₂.stop
        end

        if (id₁ != -1) && (id₂ != -1)
          nNode₁ = diagonals[l₁].points[id₁]
          nNode₂ = diagonals[l₂].points[id₂]

          # The indices on both diagonals have to be the same
          if nNode₁ != nNode₂
            println(ts₁, " ", t₁)
            println(ts₂, " ", t₂)
            throw("Confusion")
          end
        elseif id₁ != -1
          # Use number on diagonal 1 and reuse rg₂, preserve ends
          nNode = diagonals[l₁].points[id₁]
          idx = min(max(2, rg₂.start), length(diagonals[l₂].abscissae))
          insert!(diagonals[l₂].abscissae, idx, t₂)
          insert!(diagonals[l₂].points, idx, nNode)
        elseif id₂ != -1
          # Use number on diagonal 2 and reuse rg₁, preserve ends
          nNode = diagonals[l₂].points[id₂]
          idx = min(max(2, rg₁.start), length(diagonals[l₁].abscissae))
          insert!(diagonals[l₁].abscissae, idx, t₁)
          insert!(diagonals[l₁].points, idx, nNode)
        else
          # Create point and image
          # Take average on both lines for stability and symmetry
          point = @. ((p̄₁ + t₁ * Δp₁) + (p̄₂ + t₂ * Δp₂)) / 2
          image = ((1 - t₁) * u₁₋ + (1 + t₁) * u₁₊) ./ 4
          image .+= ((1 - t₂) * u₂₋ + (1 + t₂) * u₂₊) ./ 4

          push!(lin.points, point)
          push!(lin.images, image)
          nNodes += 1
          addNode!(graph, nNodes)

          # Add on diagonals (reuse rg₁ and rg₂), preserve ends
          abscissae, points = diagonals[l₁].abscissae, diagonals[l₁].points
          id₁ = min(max(2, rg₁.start), length(abscissae))
          insert!(abscissae, id₁, t₁)
          insert!(points, id₁, nNodes)

          abscissae, points = diagonals[l₂].abscissae, diagonals[l₂].points
          id₂ = min(max(2, rg₂.start), length(abscissae))
          insert!(abscissae, id₂, t₂)
          insert!(points, id₂, nNodes)
        end
      end
    end

    ##########################
    # Add diagonals to graph #
    ##########################
    for line in diagonals
      # Diagonals are never on the boundary
      for (p₋, p₊) in partition(line.points, 2, 1)
        addEdge!(graph, p₋, p₊, line.number, false)
        adjacency[minmax(p₋, p₊)] = (false, -1, -1)
      end
    end
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

  # dict = Dict{Symbol,Any}()
  # dict[:x] = Vector{Union{Nothing,T}}()
  # dict[:y] = Vector{Union{Nothing,T}}()
  # if E == 3
  #   dict[:z] = Vector{Union{Nothing,T}}()
  #   dict[:type] = "scatter3d"
  # end
  # dict[:mode] = "markers+lines"
  # dict[:marker] = attr(size=2, color="firebrick")
  # dict[:line] = attr(width=1, color="royalblue")

  # for (n₋, n₊) in getEdges(graph)
  #   p₋, p₊ = lin.points[n₋], lin.points[n₊]
  #   # graph.marks[(n₋, n₊)] != 1 && continue
  #   push!(dict[:x], p₋[1], p₊[1], nothing)
  #   push!(dict[:y], p₋[2], p₊[2], nothing)
  #   if E == 3
  #     push!(dict[:z], p₋[3], p₊[3], nothing)
  #   end
  # end
  # display(PlotlyJS.plot([
  #   PlotlyJS.scatter(; dict...),
  # ]))

  ##################
  # Extract cycles #
  ##################
  cycles = Vector{Vector{Int}}()

  # Strategy
  # * Start a cycle from a point with only two neighbours
  # * Force orientation to be +1 in (α, β)
  # * Add points until the cycle is closed
  # * The next point is always in matching orientation and minimises the angle
  # * Marked edges as visited one more time

  # Edges visited twice are removed
  # Loop until there is no more active edge

  while true
    # Find point with two neighbours
    # Such that the three points are not aligned
    # Alignment has to be checked in two ways:
    # * Points belong to same parent edge/diagonal
    # * The two edges previously (at earlier layer) belonged to same parent
    #   Compute sin² to check this condition

    cycle = Vector{Int}()

    n₋, n₀, n₊ = -1, -1, -1
    for n in 1:nNodes
      nn = getNeighbours(graph, n)
      if length(nn) == 2
        n₀ = n
        n₋, n₊ = nn

        aligned = (getLine(graph, n₋, n₀) == getLine(graph, n₀, n₊))
        if !aligned
          p₋, p₊, p₀ = lin.points[n₋], lin.points[n₀], lin.points[n₊]
          sin² = 1 - dot₋₋(p₊, p₀, p₋, p₀)^2 / norm2₋(p₊, p₀) / norm2₋(p₋, p₀)
          aligned = (sin² < ε)
        end

        if !aligned
          break
        else
          n₋, n₀, n₊ = -1, -1, -1
        end
      end
    end

    if n₀ != -1
      # Found a vertex at which we can force orientation
      # Compute orientation and to be +1 according to (α, β) by flipping
      p₋, p₀, p₊ = lin.points[n₋], lin.points[n₀], lin.points[n₊]

      s₋, t₋ = dot₋(p₋, p₀, α), dot₋(p₋, p₀, β)
      s₊, t₊ = dot₋(p₊, p₀, α), dot₋(p₊, p₀, β)

      d = sqrt(s₋^2 + t₋^2)
      s₋, t₋ = s₋ / d, t₋ / d
      d = sqrt(s₊^2 + t₊^2)
      s₊, t₊ = s₊ / d, t₊ / d
      o = det2(s₋, t₋, s₊, t₊)

      if o > ε
        cycle = [n₋, n₀, n₊]
        addMark!(graph, n₋, n₀)
        addMark!(graph, n₀, n₊)
      else
        cycle = [n₊, n₀, n₋]
        addMark!(graph, n₊, n₀)
        addMark!(graph, n₀, n₋)
      end
    elseif !isempty(graph.marks)
      # If there remain edges, take one that has not been visited
      # So we can force orientation
      n₋, n₊ = -1, -1
      for k in getEdges(graph)
        if graph.marks[k] == 0
          n₋, n₊ = k
          break
        end
      end
      cycle = [n₋, n₊]
      addMark!(graph, n₋, n₊)
    else
      # Otherwise the extraction is complete
      break
    end

    # Close cycle
    while true
      # Retrieve last edge
      n₋, n₀ = cycle[end-1], cycle[end]
      p₋, p₀ = lin.points[n₋], lin.points[n₀]
      l = getLine(graph, n₋, n₀)

      # Compute coordinates in (α, β)
      s₋, t₋ = dot₋(p₋, p₀, α), dot₋(p₋, p₀, β)
      d = sqrt(s₋^2 + t₋^2)
      s₋, t₋ = s₋ / d, t₋ / d

      n₊, o₊, s₊, t₊ = -1, -2, T(0), T(0)
      if isempty(getNeighbours(graph, n₀))
        throw("EMPTY")
      end

      # Look for edge with smallest angle oriented positively
      for n in getNeighbours(graph, n₀)
        (n == n₋) && continue
        p = lin.points[n]

        # Check alignment
        aligned = (getLine(graph, n₀, n) == l)
        sin² = -3
        if !aligned
          sin² = 1 - dot₋₋(p, p₀, p₋, p₀)^2 / norm2₋(p, p₀) / norm2₋(p₋, p₀)
          aligned = (sin² < ε)
        end

        if aligned
          s, t = T(0), T(0)
          o = T(0)
        else
          # If not aligned, compute orientation
          s, t = dot₋(p, p₀, α), dot₋(p, p₀, β)
          d = sqrt(s^2 + t^2)
          s, t = s / d, t / d

          o = sign(det2(s₋, t₋, s, t))
          (o < 0) && continue
        end

        if n₊ == -1
          n₊, o₊, s₊, t₊ = n, o, s, t
        elseif (o == 1) && (o₊ == 0 || sign(det2(s₊, t₊, s, t)) == T(-1))
          # If current best is straight and candidate is left
          # Or if "more on the left" (smaller angle)
          n₊, o₊, s₊, t₊ = n, o, s, t
        end
      end

      # Add point to cycle and mark edge
      if n₊ == -1
        for k in cycle
          println(lin.points[k])
        end

        for (k₋, k₊) in CircularIterator(cycle)
          addEdge!(graph, k₋, k₊, 1, false)
        end

        # dict = Dict{Symbol,Any}()
        # dict[:x] = Vector{Union{Nothing,T}}()
        # dict[:y] = Vector{Union{Nothing,T}}()
        # if E == 3
        #   dict[:z] = Vector{Union{Nothing,T}}()
        #   dict[:type] = "scatter3d"
        # end
        # dict[:mode] = "markers+lines"
        # dict[:marker] = attr(size=2, color="firebrick")
        # dict[:line] = attr(width=1, color="royalblue")

        # for (n₋, n₊) in getEdges(graph)
        #   p₋, p₊ = lin.points[n₋], lin.points[n₊]
        #   push!(dict[:x], p₋[1], p₊[1], nothing)
        #   push!(dict[:y], p₋[2], p₊[2], nothing)
        #   if E == 3
        #     push!(dict[:z], p₋[3], p₊[3], nothing)
        #   end
        # end
        # display(PlotlyJS.plot([
        #   PlotlyJS.scatter(; dict...),
        # ]))

        throw(AssertionError("Found -1"))
      end

      addMark!(graph, n₀, n₊)
      if n₊ != cycle[1]
        push!(cycle, n₊)
      else
        break
      end
    end
    reverse!(cycle)
    push!(cycles, cycle)

    # Tag all edges as boundaries of this new cycle
    c = length(cycles)
    for (k₋, k₊) in CircularIterator(cycle)
      k = minmax(k₋, k₊)
      b, f, _ = adjacency[k]
      if f == -1
        adjacency[k] = (b, c, -1)
      else
        adjacency[k] = (b, f, c)
      end
    end
  end

  lin.cycles = cycles
  lin.adjacency = adjacency

  # dict = Dict{Symbol,Any}()
  # dict[:x] = Vector{Union{Nothing,T}}()
  # dict[:y] = Vector{Union{Nothing,T}}()
  # if E == 3
  #   dict[:z] = Vector{Union{Nothing,T}}()
  #   dict[:type] = "scatter3d"
  # end
  # dict[:mode] = "markers+lines"
  # dict[:marker] = attr(size=2, color="firebrick")
  # dict[:line] = attr(width=1, color="royalblue")

  # for (n₋, n₊) in getEdges(graph)
  #   p₋, p₊ = lin.points[n₋], lin.points[n₊]
  #   push!(dict[:x], p₋[1], p₊[1], nothing)
  #   push!(dict[:y], p₋[2], p₊[2], nothing)
  #   if E == 3
  #     push!(dict[:z], p₋[3], p₊[3], nothing)
  #   end
  # end
  # display(PlotlyJS.plot([
  #   PlotlyJS.scatter(; dict...),
  # ]))

  ######################
  # Move to next layer #
  ######################
  linearise!(u, lin; layer=layer + 1)
end
