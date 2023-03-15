function objective_tangent(ρ, bc₋, bc₊, ξs; verbose::Bool=false)
  T = eltype(ξs)
  ∫ = zero(T)

  verbose && println("Definition of the CPWL function")
  verbose && println("(x₋, x₊, slope, origin)")

  # Left boundary
  ξ₋ = isfinite(bc₋) ? bc₋.x : -T(Inf)
  s₋, y₋ = isfinite(ξ₋) ? (∇¹(ρ, ξ₋), ∇⁰(ρ, ξ₋)) : ϕ₋(ρ)
  ξ₊ = isempty(ξs) ? (isfinite(bc₊) ? bc₊.x : +T(Inf)) : ξs[1]
  s₊, y₊ = isfinite(ξ₊) ? (∇¹(ρ, ξ₊), ∇⁰(ρ, ξ₊)) : ϕ₊(ρ)

  b₋ = isfinite(ξ₋) ? y₋ - s₋ * ξ₋ : y₋
  b₊ = isfinite(ξ₊) ? y₊ - s₊ * ξ₊ : y₊
  x₀ = -(b₊ - b₋) / (s₊ - s₋)

  if isfinite(ξ₋)
    ∫ += ∫ρϕ²(ρ, ξ₋, x₀, s₋, b₋)
  else
    ∫ += ∫ρϕ₋²(ρ, x₀)
  end
  verbose && println((ξ₋, x₀, s₋, b₋))

  # Middle
  x₋, x₊ = x₀, x₀
  for i in 1:length(ξs)-1
    ξ₋ = ξs[i]
    y₋, s₋ = ∇⁰(ρ, ξ₋), ∇¹(ρ, ξ₋)
    b₋ = y₋ - s₋ * ξ₋

    ξ₊ = ξs[i+1]
    y₊, s₊ = ∇⁰(ρ, ξ₊), ∇¹(ρ, ξ₊)
    b₊ = y₊ - s₊ * ξ₊

    x₊ = -(b₊ - b₋) / (s₊ - s₋)
    ∫ += ∫ρϕ²(ρ, x₋, x₊, s₋, b₋)
    verbose && println((x₋, x₊, s₋, b₋))
    x₋ = x₊
  end

  # Right boundary
  ξ₋ = isempty(ξs) ? (isfinite(bc₋) ? bc₋.x : -T(Inf)) : ξs[end]
  s₋, y₋ = isfinite(ξ₋) ? (∇¹(ρ, ξ₋), ∇⁰(ρ, ξ₋)) : ϕ₋(ρ)
  ξ₊ = isfinite(bc₊) ? bc₊.x : +T(Inf)
  s₊, y₊ = isfinite(ξ₊) ? (∇¹(ρ, ξ₊), ∇⁰(ρ, ξ₊)) : ϕ₊(ρ)

  b₋ = isfinite(ξ₋) ? y₋ - s₋ * ξ₋ : y₋
  b₊ = isfinite(ξ₊) ? y₊ - s₊ * ξ₊ : y₊
  x₀ = -(b₊ - b₋) / (s₊ - s₋)

  if isfinite(ξ₊)
    ∫ += ∫ρϕ²(ρ, x₀, ξ₊, s₊, b₊)
  else
    ∫ += ∫ρϕ₊²(ρ, x₀)
  end

  # Last middle
  if x₊ < x₀
    ∫ += ∫ρϕ²(ρ, x₊, x₀, s₋, b₋)
    verbose && println((x₊, x₀, s₋, b₋))
  end
  verbose && println((x₀, ξ₊, s₊, b₊))

  ∫
end

function tangent_ADAM(
  ρ::RegisteredFunction{T}, bc₋::BoundaryCondition, bc₊::BoundaryCondition, N::Int;
  lr::T=T(0.001), epochs::Int=1000, ε::T=T(0.001), ξs=nothing
) where {T}
  # We assume that ∇²ρ vanishes at bc₋ and bc₊
  if isnothing(ξs)
    if isfinite(bc₋) && isfinite(bc₊)
      x₋, x₊ = bc₋.x, bc₊.x
    elseif isfinite(bc₋)
      x₋, x₊ = bc₋.x, bc₋.x + T(10)
    elseif isfinite(bc₊)
      x₋, x₊ = bc₊.x - T(10), bc₊.x
    else
      x₋, x₊ = T(-10), T(+10)
    end
    h = (x₊ - x₋) / (N + 1)
    ξs = T.(x₋:h:x₊)[2:end-1]
  end
  ξs = Param(ξs, Adam(lr=lr))

  ξbest = copy(ξs)
  Lbest = objective_tangent(ρ, bc₋, bc₊, ξbest)
  Lbest0 = Lbest

  epoch = 0
  while true
    epoch += 1
    if epoch > epochs
      println("Maximum number of epochs exceeded")
      break
    end

    if !issorted(ξs)
      println(ξs.value)
      throw(AssertionError("Invalid partition: not sorted"))
    elseif isfinite(bc₋) && any(ξs .< bc₋.x)
      throw(AssertionError("Invalid partition: left overflow"))
    elseif isfinite(bc₊) && any(ξs .> bc₊.x)
      throw(AssertionError("Invalid partition: right overflow"))
    end

    L = @diff objective_tangent(ρ, bc₋, bc₊, ξs)
    ∇ξ = AutoGrad.full(AutoGrad.grad(L, ξs))
    Knet.update!(ξs, ∇ξ)

    L² = value(L)
    ∇² = norm(∇ξ)
    println(@sprintf("Epoch %5.0i, L=%.5E, |∇|=%.5E", epoch, L², ∇²))

    if L² < Lbest
      ξbest = copy(ξs)
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

  ξbest, Lbest
end

function gradfun_tangent(ρ, bc₋, bc₊, ξs, gs)
  ws = Param(ξs)
  L = @diff objective_tangent(ρ, bc₋, bc₊, ws)
  ∇L = AutoGrad.full(AutoGrad.grad(L, ws))
  gs .= ∇L
  nothing
end

function tangent_BFGS(
  ρ::RegisteredFunction{T}, bc₋::BoundaryCondition, bc₊::BoundaryCondition, N::Int;
  epochs::Int=1000, ε::T=T(0.001), ξs=nothing
) where {T}
  # We assume that ∇²ρ vanishes at bc₋ and bc₊
  if isnothing(ξs)
    if isfinite(bc₋) && isfinite(bc₊)
      x₋, x₊ = bc₋.x, bc₊.x
    elseif isfinite(bc₋)
      x₋, x₊ = bc₋.x, bc₋.x + T(10)
    elseif isfinite(bc₊)
      x₋, x₊ = bc₊.x - T(10), bc₊.x
    else
      x₋, x₊ = T(-10), T(+10)
    end
    h = (x₊ - x₋) / (N + 1)
    ξs = T.(x₋:h:x₊)[2:end-1]
  end

  res = Optim.optimize(
    (ξs) -> objective_tangent(ρ, bc₋, bc₊, ξs),
    (gs, ξs) -> gradfun_tangent(ρ, bc₋, bc₊, ξs, gs),
    ξs,
    Optim.Options(
      iterations=epochs,
      g_tol=ε,
      allow_f_increases=true,
      show_trace=true
    )
  )
  ξs = res.minimizer
  L² = res.minimum

  ξs, L²
end
