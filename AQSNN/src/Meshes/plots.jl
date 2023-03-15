function _plotMesh(
  ::Val{M}, ::Val{E},
  polytope::AbstractPolytope,
) where {M,E}
  dict = Dict{Symbol,Any}()

  dict[:x] = [getVertex(polytope, i)[1] for i in 1:length(polytope)]
  if E == 1
    dict[:y] = zero.(dict[:x])
  else
    dict[:y] = [getVertex(polytope, i)[2] for i in 1:length(polytope)]
  end

  if E == 3
    dict[:z] = [getVertex(polytope, i)[3] for i in 1:length(polytope)]
    dict[:type] = "scatter3d"
  end

  if M == 2
    for k in (:x, :y, :z)
      !haskey(dict, k) && continue
      push!(dict[k], dict[k][1])
    end
  end

  dict
end

function plotMesh(mesh::AbstractMesh)
  plots = Vector{PlotlyBase.AbstractTrace}()
  M = mandim(mesh)
  E = embdim(mesh)
  for cell in mesh
    dict = _plotMesh(Val(M), Val(E), cell)
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

  display(PlotlyJS.plot(
    plots,
    Layout(
      showlegend=false,
      yaxis=attr(scaleanchor="x", scaleratio=1)
    )
  ))
end
