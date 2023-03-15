#############
# plotModel #
#############
#####
# 1 #
#####
function _plotInput(::Val{2}, ::Val{1}, points, dict) where {M}
  T = eltype(points)
  N = size(points, 2)
  append!(dict[:x], view(points, 1, :))
  append!(dict[:y], zeros(T, N))
end

function _plotOutput(::Val{2}, ::Val{1}, points, images, normal, dict) where {M}
  append!(dict[:x], view(points, 1, :) .+ images .* normal[1])
  append!(dict[:y], images .* normal[2])
end

#####
# 2 #
#####
function _plotInput(::Val{2}, ::Val{2}, points, dict) where {M}
  append!(dict[:x], view(points, 1, :))
  append!(dict[:y], view(points, 2, :))
end

function _plotOutput(::Val{2}, ::Val{2}, points, images, normal, dict) where {M}
  append!(dict[:x], view(points, 1, :) .+ images .* normal[1])
  append!(dict[:y], view(points, 2, :) .+ images .* normal[2])
end

function _plotInput(::Val{3}, ::Val{2}, points, dict) where {M}
  T = eltype(points)
  N = size(points, 2)
  append!(dict[:x], view(points, 1, :))
  append!(dict[:y], view(points, 2, :))
  append!(dict[:z], zeros(T, N))
end

function _plotOutput(::Val{3}, ::Val{2}, points, images, normal, dict) where {M}
  append!(dict[:x], view(points, 1, :) .+ images .* normal[1])
  append!(dict[:y], view(points, 2, :) .+ images .* normal[2])
  append!(dict[:z], images .* normal[3])
end

#####
# 3 #
#####
function _plotInput(::Val{3}, ::Val{3}, points, dict) where {M}
  append!(dict[:x], view(points, 1, :))
  append!(dict[:y], view(points, 2, :))
  append!(dict[:z], view(points, 3, :))
end

function _plotOutput(::Val{3}, ::Val{3}, points, images, normal, dict) where {M}
  append!(dict[:x], view(points, 1, :) .+ images .* normal[1])
  append!(dict[:y], view(points, 2, :) .+ images .* normal[2])
  append!(dict[:z], view(points, 3, :) .+ images .* normal[3])
end

##############
# plotModels #
##############
function _plotModels(
  models::AbstractVector, meshes::AbstractVector{<:SimplexMesh{T,M,E,P}}, normal;
  N::Int=10^E, showGrad::Bool=false
) where {T,M,E,P}
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

  # Domain and normal
  for _ in 1:2
    dict = Dict{Symbol,Any}()
    dict[:x] = Vector{T}()
    dict[:y] = Vector{T}()
    if B == 3
      dict[:z] = Vector{T}()
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

  # Functions
  nmodels = length(models)
  for _ in 1:nmodels
    dict = Dict{Symbol,Any}()
    dict[:x] = Vector{T}()
    dict[:y] = Vector{T}()
    if B == 3
      dict[:z] = Vector{T}()
      dict[:type] = "scatter3d"
    end
    push!(dicts, dict)
  end

  # Gradients
  if showGrad
    for _ in 1:nmodels
      for _ in 1:E
        dict = Dict{Symbol,Any}()
        dict[:x] = Vector{T}()
        dict[:y] = Vector{T}()
        if B == 3
          dict[:z] = Vector{T}()
          dict[:type] = "scatter3d"
        end
        push!(dicts, dict)
      end
    end
  end

  # Fill dictionaries
  for mesh in meshes
    sampler = PolytopeSampler(mesh, N)
    points = getSamples(sampler)
    _plotInput(Val(B), Val(E), points, dicts[1])

    for (i, (_, u)) in enumerate(models)
      ∇⁰U = isa(u, Sequential) ? ∇⁰(u, points) : u(points)
      images = view(∇⁰U, 1, :)
      _plotOutput(Val(B), Val(E), points, images, normal, dicts[i+2])
    end

    if showGrad
      for (i, (_, u)) in enumerate(models)
        ∇¹U = ∇¹(u, points)
        for e in 1:E
          idx = 2 + nmodels + (i - 1) * E + e
          images = view(∇¹U, e, :)
          _plotOutput(Val(B), Val(E), points, images, normal, dicts[idx])
        end
      end
    end
  end

  # Formatting
  dicts[1][:name] = "Ω"
  dicts[2][:name] = "dir"
  for (i, (name, _)) in enumerate(models)
    dicts[i+2][:name] = name
  end
  if showGrad
    subscripts = ["₁", "₂", "₃"]
    for (i, (name, _)) in enumerate(models)
      for e in 1:E
        idx = 2 + nmodels + (i - 1) * E + e
        dicts[idx][:name] = "∂$(subscripts[e])" * name
      end
    end
  end

  for dict in dicts
    dict[:mode] = "markers"
    dict[:marker] = attr(size=1)
  end

  dicts[2][:mode] = "markers+lines"
  dicts[2][:marker] = attr(
    size=5,
    symbol=["square", "circle"]
  )

  # Plot
  display(PlotlyJS.plot(
    [PlotlyJS.scatter(; dict...) for dict in dicts],
    PlotlyJS.Layout(
      showlegend=true,
      yaxis=attr(scaleanchor="x", scaleratio=1),
      zaxis=attr(scaleanchor="x", scaleratio=1),
    )
  ))
end

#############
# plotModel #
#############
function plotModel(
  u::Sequential, Ωτ::SimplexMesh, Γτs::AbstractVector{<:SimplexMesh};
  N::Int=1000, showGrad::Bool=false
)
  normal = getNormal(Ωτ[1])
  _plotModels([("U", u)], Γτs, normal; N=N, showGrad=showGrad)
end

function plotModel(
  u::Sequential, Ωτ::SimplexMesh;
  N::Int=1000, showGrad::Bool=false
)
  normal = getNormal(Ωτ[1])
  _plotModels([("U", u)], [Ωτ], normal; N=N, showGrad=showGrad)
end

#############
# plotBasis #
#############
function plotBasis(
  u::Sequential, Ωτ::SimplexMesh, Γτs::AbstractVector{<:SimplexMesh};
  N::Int=1000, showGrad::Bool=false, bases::AbstractVector{Int}=Vector{Int}()
)
  if isempty(bases)
    bases = collect(1:u.architecture[end-1])
  end
  normal = getNormal(getPolytope(Ωτ))
  _plotModels(
    [("U[$i]", basisFunction(u, i)) for i in bases], Γτs, normal;
    N=N, showGrad=showGrad
  )
end

function plotBasis(
  u::Sequential, Ωτ::SimplexMesh;
  N::Int=1000, showGrad::Bool=false, bases::Vector{Int}=Vector{Int}()
)
  if isempty(bases)
    bases = collect(1:u.architecture[end-1])
  end
  normal = getNormal(getPolytope(Ωτ))
  _plotModels(
    [("U[$i]", basisFunction(u, i)) for i in bases], [Ωτ], normal;
    N=N, showGrad=showGrad
  )
end
