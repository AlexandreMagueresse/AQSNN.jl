function _plotDomain(polytope::AbstractPolytope{T,M,E}) where {T,M,E}
  dict = Dict{Symbol,Any}()

  dict[:x] = [getVertex(polytope, i)[1] for i in 1:length(polytope)]
  if E == 1
    dict[:y] = zero.(dict[:x])
  else
    dict[:y] = [getVertex(polytope, i)[2] for i in 1:length(polytope)]
  end

  if E == 3
    dict[:type] = "scatter3d"
    dict[:z] = [getVertex(polytope, i)[3] for i in 1:length(polytope)]
  end

  if M == 2
    for k in (:x, :y, :z)
      !haskey(dict, k) && continue
      push!(dict[k], dict[k][1])
    end
  end

  dict
end

function _plotBoundary(polytope::AbstractPolytope{T,M,E}) where {T,M,E}
  c = sum(i -> getVertex(polytope, i), 1:length(polytope)) ./ length(polytope)
  n = getNormal(polytope)

  dict = Dict{Symbol,Any}()

  dict[:x] = [c[1], c[1] + n[1]]
  if E == 1
    dict[:y] = zero.(dict[:x])
  else
    dict[:y] = [c[2], c[2] + n[2]]
  end

  if E == 3
    dict[:z] = [c[3], c[3] + n[3]]
    dict[:type] = "scatter3d"
  end

  dict
end

function plotPolytope(
  polytopes::AbstractVector{<:AbstractPolytope},
  boundaries::AbstractVector{<:AbstractPolytope}=Vector{AbstractPolytope}()
)
  plots = Vector{PlotlyBase.AbstractTrace}()
  for polytope in polytopes
    dict = _plotDomain(polytope)
    dict[:mode] = "markers+lines"
    dict[:marker] = attr(
      size=5,
      color="firebrick"
    )
    dict[:line] = attr(
      width=2,
      color="royalblue"
    )

    push!(plots, PlotlyJS.scatter(; dict...))
  end

  for boundary in boundaries
    dict = _plotBoundary(boundary)
    dict[:mode] = "markers+lines"
    dict[:marker] = attr(
      size=5,
      color="seagreen",
      symbol=["square", "circle"]
    )
    dict[:line] = attr(
      width=2,
      color="royalblue"
    )

    push!(plots, PlotlyJS.scatter(; dict...))
  end

  display(PlotlyJS.plot(
    plots,
    Layout(showlegend=false, yaxis=attr(scaleanchor="x", scaleratio=1))
  ))
end

function plotPolytope(
  polytope::AbstractPolytope,
  boundaries::AbstractVector{<:AbstractPolytope}=Vector{AbstractPolytope}()
)
  plotPolytope([polytope], boundaries)
end
