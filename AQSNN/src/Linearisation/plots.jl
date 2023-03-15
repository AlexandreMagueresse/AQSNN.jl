function plotLineariser(::AbstractVector{<:AbstractLineariser})
  @notimplemented
end

function plotLineariser(lins::AbstractVector{<:AbstractLineariser{T,0,E}}) where {T,E}
  dict = Dict{Symbol,Any}()
  dict[:x] = Vector{Union{T,Nothing}}()
  dict[:y] = Vector{Union{T,Nothing}}()
  if E == 3
    dict[:z] = Vector{Union{T,Nothing}}()
    dict[:type] = "scatter3d"
  end

  for lin in lins
    p = lin.points[1]
    push!(dict[:x], p[1], nothing)
    if E >= 2
      push!(dict[:y], p[2], nothing)
    else
      push!(dict[:y], 0, nothing)
    end
    if E == 3
      push!(dict[:z], p[3], nothing)
    end
  end

  display(PlotlyJS.plot(
    PlotlyJS.scatter(;
      dict...,
      mode="markers",
      marker=attr(
        size=5,
        color="firebrick"
      )
    )
  ))
end

function plotLineariser(lins::AbstractVector{<:AbstractLineariser{T,1,E}}) where {T,E}
  dict = Dict{Symbol,Any}()
  dict[:x] = Vector{Union{T,Nothing}}()
  dict[:y] = Vector{Union{T,Nothing}}()
  if E == 3
    dict[:z] = Vector{Union{T,Nothing}}()
    dict[:type] = "scatter3d"
  end

  N = 0
  for lin in lins
    for point in lin.points
      push!(dict[:x], point[1])
      if E >= 2
        push!(dict[:y], point[2])
      else
        push!(dict[:y], 0)
      end
      if E == 3
        push!(dict[:z], point[3])
      end
    end

    N += length(lin.points)

    push!(dict[:x], nothing)
    push!(dict[:y], nothing)
    if E == 3
      push!(dict[:z], nothing)
    end
  end

  display(PlotlyJS.plot(
    PlotlyJS.scatter(;
      dict...,
      mode="markers+lines",
      marker=attr(
        size=5,
        color="firebrick"
      ),
      line=attr(
        color="royalblue"
      )
    ),
    PlotlyJS.Layout(
      title="Linear regions of u ($N)",
    )
  ))
end

function plotLineariser(lins::AbstractVector{<:AbstractLineariser{T,2,E}}) where {T,E}
  dict = Dict{Symbol,Any}()
  dict[:x] = Vector{Union{T,Nothing}}()
  dict[:y] = Vector{Union{T,Nothing}}()
  if E == 3
    dict[:z] = Vector{Union{T,Nothing}}()
    dict[:type] = "scatter3d"
  end

  N = 0
  for lin in lins
    for cycle in lin.cycles
      for (k₋, k₊) in CircularIterator(cycle)
        p₋, p₊ = lin.points[k₋], lin.points[k₊]
        push!(dict[:x], p₋[1], p₊[1], nothing)
        if E >= 2
          push!(dict[:y], p₋[2], p₊[2], nothing)
        else
          push!(dict[:y], 0, 0, nothing)
        end
        if E == 3
          push!(dict[:z], p₋[3], p₊[3], nothing)
        end
      end
    end

    N += length(lin.cycles)
  end

  display(PlotlyJS.plot(
    PlotlyJS.scatter(;
      dict...,
      mode="markers+lines",
      marker=attr(
        size=5,
        color="firebrick"
      ),
      line=attr(
        color="royalblue"
      )
    ),
    PlotlyJS.Layout(
      title="Linear regions of u ($N)",
    )
  ))
end

function plotLineariser(lin::AbstractLineariser)
  plotLineariser([lin])
end

##################
# plotProjection #
##################
function plotProjection(f::Function, meshes::AbstractVector{<:SimplexMesh}, normal, lins::AbstractVector{<:AbstractLineariser{T,M,E}}; N::Int=10^E) where {T,M,E}
  if length(normal) == 1
    normal = [0, normal[1]]
  end
  B = length(normal)

  # Compute barycenter
  p̄ = zeros(T, E)
  ms = 0
  for mesh in meshes
    for cell in mesh
      m = measure(cell)
      p̄ += m * sum(i -> getVertex(cell, i), 1:length(cell)) / length(cell)
      ms += m
    end
  end
  p̄ /= ms

  # Initialise dictionaries
  dicts = Vector{Dict{Symbol,Any}}()

  # Domain, normal, function, projection
  for _ in 1:4
    dict = Dict{Symbol,Any}()
    dict[:x] = Vector{Union{T,Nothing}}()
    dict[:y] = Vector{Union{T,Nothing}}()
    if B == 3
      dict[:z] = Vector{Union{T,Nothing}}()
      dict[:type] = "scatter3d"
    end
    push!(dicts, dict)
  end

  dicts[2][:x] = [p̄[1], p̄[1] + normal[1]]
  if B >= 2
    p = length(p̄) >= 2 ? p̄[2] : zero(T)
    dicts[2][:y] = [p, p + normal[2]]
  end
  if B >= 3
    p = length(p̄) >= 3 ? p̄[3] : zero(T)
    dicts[2][:z] = [p, p + normal[3]]
  end

  if M == 2
    dicts[4][:i] = Vector{Int}()
    dicts[4][:j] = Vector{Int}()
    dicts[4][:k] = Vector{Int}()
  end

  # Fill dictionaries
  for mesh in meshes
    sampler = PolytopeSampler(mesh, N)
    points = getSamples(sampler)
    _plotInput(Val(B), Val(E), points, dicts[1])

    F = f(points)
    images = view(F, 1, :)
    _plotOutput(Val(B), Val(E), points, images, normal, dicts[3])
  end


  for lin in lins
    offset = length(dicts[4][:x])

    for k in axes(lin.points, 1)
      point, image = lin.points[k], lin.images[k][1]
      if E == 1
        push!(dicts[4][:x], point[1] + image * normal[1])
        push!(dicts[4][:y], image * normal[2])
      elseif E == 2
        push!(dicts[4][:x], point[1] + image * normal[1])
        push!(dicts[4][:y], point[2] + image * normal[2])
        if B == 3
          push!(dicts[4][:z], image * normal[3])
        end
      elseif E == 3
        push!(dicts[4][:x], point[1] + image * normal[1])
        push!(dicts[4][:y], point[2] + image * normal[2])
        push!(dicts[4][:z], point[3] + image * normal[3])
      end
    end

    if M == 2
      for cycle in lin.cycles
        for k in 2:length(cycle)-1
          # Probably from a Python convention, indices start at 0
          # with the mesh3d function
          push!(dicts[4][:i], offset + cycle[1] - 1)
          push!(dicts[4][:j], offset + cycle[k] - 1)
          push!(dicts[4][:k], offset + cycle[k+1] - 1)
        end
      end
    end

    push!(dicts[4][:x], nothing)
    push!(dicts[4][:y], nothing)
    if B == 3
      push!(dicts[4][:z], nothing)
    end
  end

  # Formatting
  dicts[1][:name] = "Ω"
  dicts[2][:name] = "dir"
  dicts[3][:name] = "F"
  dicts[4][:name] = "Fproj"

  for dict in dicts
    dict[:mode] = "markers"
    dict[:marker] = attr(size=(B == 1) ? 3 : 1)
  end

  dicts[2][:mode] = "markers+lines"
  dicts[2][:marker] = attr(
    size=5,
    symbol=["square", "circle"]
  )

  plots = [PlotlyJS.scatter(; dicts[k]...) for k in 1:3]
  if M == 2
    delete!(dicts[4], :type)
    push!(plots, PlotlyJS.mesh3d(; dicts[4]...))
  else
    dicts[4][:mode] = "markers+lines"
    dicts[4][:marker] = attr(
      size=5, color="firebrick"
    )
    dicts[4][:line] = attr(
      width=2, color="royalblue"
    )
    push!(plots, PlotlyJS.scatter(; dicts[4]...))
  end

  # Plot
  display(PlotlyJS.plot(
    plots,
    PlotlyJS.Layout(
      showlegend=true,
      yaxis=attr(
        scaleanchor="x",
        scaleratio=1
      ),
      zaxis=attr(
        scaleanchor="x",
        scaleratio=1
      ),
    )
  ))
end

function plotProjection(f::Function, Ωτ::SimplexMesh, Γτs::AbstractVector{<:SimplexMesh}, lins::AbstractVector{<:AbstractLineariser}; N::Int=1000)
  normal = getNormal(getPolytope(Ωτ))
  plotProjection(f, Γτs, normal, lins; N=N)
end

function plotProjection(f::Function, Ωτ::SimplexMesh, lin::AbstractLineariser; N::Int=1000)
  normal = getNormal(getPolytope(Ωτ))
  plotProjection(f, [Ωτ], normal, [lin]; N=N)
end
