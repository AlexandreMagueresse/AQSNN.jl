function plotTrain(historyTuple; logx::Bool=false, logy::Bool=false)
  epochs = [h[1] for h in historyTuple]
  history = [h[2] for h in historyTuple]
  if logx
    epochs = log10.(epochs)
  end
  if logy && (minimum(history) > 0)
    history = log10.(history)
  end

  display(
    PlotlyJS.plot(
      PlotlyJS.scatter(
        x=epochs, y=history,
        mode="markers",
        marker=attr(
          size=2,
          color="royalblue"
        )
      ),
      PlotlyJS.Layout(
        title="Learning curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        showlegend=false
      )
    )
  )
end

function plotComp(
  u::Sequential, û::Function, Ωτ::SimplexMesh, Γτs::AbstractVector{<:SimplexMesh};
  N::Int=1000
)
  normal = getNormal(Ωτ[1])
  _plotModels([("U", u), ("Û", û)], Γτs, normal; N=N, showGrad=false)
end

function plotComp(
  u::Sequential, û::Function, Ωτ::SimplexMesh;
  N::Int=1000
)
  normal = getNormal(Ωτ[1])
  _plotModels([("U", u), ("Û", û)], [Ωτ], normal; N=N, showGrad=false)
end

function plotDiff(
  u::Sequential, û::Function, Ωτ::SimplexMesh, Γτs::AbstractVector{<:SimplexMesh};
  N::Int=1000
)
  normal = getNormal(Ωτ[1])
  _plotModels([("ΔU", x -> u(x) .- û(x))], Γτs, normal; N=N, showGrad=false)
end

function plotDiff(
  u::Sequential, û::Function, Ωτ::SimplexMesh;
  N::Int=1000
)
  normal = getNormal(Ωτ[1])
  _plotModels([("ΔU", x -> u(x) .- û(x))], [Ωτ], normal; N=N, showGrad=false)
end
