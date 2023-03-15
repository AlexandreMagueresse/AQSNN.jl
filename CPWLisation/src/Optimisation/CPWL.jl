function objective_cpwl(ρ, bc₋, bc₊, xs, ys)
  N = length(xs)
  ∫ = zero(eltype(ys))

  # Left boundary condition
  if !isfinite(bc₋)
    ∫ += ∫ρϕ₋²(ρ, xs[1])
  end

  # Integral in the middle
  for i in 1:N-1
    # 1. Define (x₋, y₋) and (x₊, y₊)
    x₋, y₋ = getPoint(ρ, bc₋, bc₊, xs, ys, i)
    x₊, y₊ = getPoint(ρ, bc₋, bc₊, xs, ys, i + 1)

    # 2. Define line
    Δx = x₊ - x₋
    Δy = y₊ - y₋
    Δxy = y₋ * x₊ - y₊ * x₋
    slope = Δy / Δx
    origin = Δxy / Δx

    # 3. ∫( ρ(x)^2 )
    ∫ += ∫ρ²(ρ, x₋, x₊)

    # 4. -2 * ∫( ρ(x) (slope * x + origin) )
    ∫ -= 2 * slope * ∫ρx(ρ, x₋, x₊)
    ∫ -= 2 * origin * ∫ρ(ρ, x₋, x₊)

    # 5. ∫( (slope * x + origin)^2 )
    # = ∫ (Δy / Δx (x - x₋) + y₋)^2
    # = Δx / Δy [(Δy / Δx (x - x₋) + y₋)^3] / 3
    # = Δx / Δy (y₊^3 - y₋^3) / 3
    # = Δx (y₋^2 + y₋ y₊ + y₊^2) / 3
    ∫ += Δx * (y₊^2 + y₊ * y₋ + y₋^2) / 3
  end

  # Right boundary condition
  if !isfinite(bc₊)
    ∫ += ∫ρϕ₊²(ρ, xs[end])
  end

  ∫
end

function project_cpwl!(ρ, bc₋, bc₊, xs, ys)
  if !isfinite(bc₋)
    α, β = ϕ₋(ρ)
    ys[1] = α * xs[1] + β
  elseif isa(bc₋, PointSlope)
    xx, yy = getPoint(ρ, bc₋, bc₊, xs, ys, 2)
    ys[1] = yy - bc₋.slope * (xx - bc₋.x)
  end

  if !isfinite(bc₊)
    α, β = ϕ₊(ρ)
    ys[end] = α * xs[end] + β
  elseif isa(bc₊, PointSlope)
    xx, yy = getPoint(ρ, bc₋, bc₊, xs, ys, length(xs) - 1)
    ys[end] = yy - bc₊.slope * (xx - bc₊.x)
  end
  nothing
end

function cpwl_ADAM(
  ρ::RegisteredFunction{T}, bc₋::BoundaryCondition, bc₊::BoundaryCondition, N::Int;
  lr::T=T(0.001), epochs::Int=1000, ε::T=T(0.001), xs=nothing, ys=nothing
) where {T}
  x₋ = isfinite(bc₋) ? bc₋.x : T(-10)
  x₊ = isfinite(bc₊) ? bc₊.x : T(+10)
  h = (x₊ - x₋) / (N + 1)

  if isnothing(xs) || isnothing(ys)
    xs = Param(T.(x₋:h:x₊), Adam(lr=lr))
    ys = Param(ρ.(xs.value), Adam(lr=lr))
  else
    xs = Param(xs, Adam(lr=lr))
    ys = Param(ys, Adam(lr=lr))
  end

  if !isfinite(bc₋)
    α, β = ϕ₋(ρ)
    ys[1] = α * xs[1] + β
  end
  if !isfinite(bc₊)
    α, β = ϕ₊(ρ)
    ys[end] = α * xs[end] + β
  end

  xbest, ybest = copy(xs), copy(ys)
  Lbest = objective_cpwl(ρ, bc₋, bc₊, xbest, ybest)
  Lbest0 = Lbest

  epoch = 0
  while true
    epoch += 1
    if epoch > epochs
      println("Maximum number of epochs exceeded")
      break
    end

    if !issorted(xs)
      throw(AssertionError("Invalid partition: not sorted"))
    elseif isfinite(bc₋) && any(xs .< x₋)
      throw(AssertionError("Invalid partition: left overflow"))
    elseif isfinite(bc₊) && any(xs .> x₊)
      throw(AssertionError("Invalid partition: right overflow"))
    end

    L = @diff objective_cpwl(ρ, bc₋, bc₊, xs, ys)
    ∇x = AutoGrad.full(AutoGrad.grad(L, xs))
    Knet.update!(xs, ∇x)
    project_cpwl!(ρ, bc₋, bc₊, xs, ys)

    L = @diff objective_cpwl(ρ, bc₋, bc₊, xs, ys)
    ∇y = AutoGrad.full(AutoGrad.grad(L, ys))
    Knet.update!(ys, ∇y)
    project_cpwl!(ρ, bc₋, bc₊, xs, ys)

    L² = value(L)
    ∇² = norm(∇x)^2 + norm(∇y)^2
    println(@sprintf("Epoch %5.0i, L²=%.5E, |∇|²=%.5E", epoch, L², ∇²))

    if L² < Lbest
      xbest, ybest = copy(xs), copy(ys)
      Lbest = L²
    end

    if ∇² < ε
      println("Reached convergence")
      break
    end
  end

  if Lbest < Lbest0
    println(@sprintf("Found new best. Reduced by %.3E %%.", 100 * (Lbest0 - Lbest) / Lbest0))
  end

  xbest, ybest, Lbest
end

function gradfun_cpwl(ρ, bc₋, bc₊, xs, ys, gs)
  xs = Param(xs)
  ys = Param(ys)
  L = @diff objective_cpwl(ρ, bc₋, bc₊, xs, ys)
  ∇x = AutoGrad.full(AutoGrad.grad(L, xs))
  ∇y = AutoGrad.full(AutoGrad.grad(L, ys))
  gs[1:div(end, 2)] .= ∇x
  gs[div(end, 2)+1:end] .= ∇y
  nothing
end

function cpwl_BFGS(
  ρ::RegisteredFunction{T}, bc₋::BoundaryCondition, bc₊::BoundaryCondition, N::Int;
  epochs::Int=1000, ε::T=T(0.001), xs=nothing, ys=nothing
) where {T}
  x₋ = isfinite(bc₋) ? bc₋.x : T(-10)
  x₊ = isfinite(bc₊) ? bc₊.x : T(+10)
  h = (x₊ - x₋) / (N + 1)

  if isnothing(xs) || isnothing(ys)
    xs = T.(x₋:h:x₊)
    ys = ρ.(xs)
  end

  if !isfinite(bc₋)
    α, β = ϕ₋(ρ)
    ys[1] = α * xs[1] + β
  end
  if !isfinite(bc₊)
    α, β = ϕ₊(ρ)
    ys[end] = α * xs[end] + β
  end

  res = Optim.optimize(
    (ξs) -> begin
      xs = ξs[1:div(end, 2)]
      ys = ξs[div(end, 2)+1:end]
      if !issorted(xs)
        throw(AssertionError("Invalid partition: not sorted"))
      elseif isfinite(bc₋) && any(xs .< x₋)
        throw(AssertionError("Invalid partition: left overflow"))
      elseif isfinite(bc₊) && any(xs .> x₊)
        throw(AssertionError("Invalid partition: right overflow"))
      end

      project_cpwl!(ρ, bc₋, bc₊, xs, ys)
      objective_cpwl(ρ, bc₋, bc₊, xs, ys)
    end,
    (gs, ξs) -> begin
      xs = ξs[1:div(end, 2)]
      ys = ξs[div(end, 2)+1:end]
      if !issorted(xs)
        throw(AssertionError("Invalid partition: not sorted"))
      elseif isfinite(bc₋) && any(xs .< x₋)
        throw(AssertionError("Invalid partition: left overflow"))
      elseif isfinite(bc₊) && any(xs .> x₊)
        throw(AssertionError("Invalid partition: right overflow"))
      end

      project_cpwl!(ρ, bc₋, bc₊, xs, ys)
      gradfun_cpwl(ρ, bc₋, bc₊, xs, ys, gs)
    end,
    [xs; ys],
    Optim.Options(
      iterations=epochs,
      g_tol=ε,
      allow_f_increases=true,
      show_trace=true
    )
  )
  xs = res.minimizer[1:div(end, 2)]
  ys = res.minimizer[div(end, 2)+1:end]
  L² = res.minimum

  xs, ys, L²
end
