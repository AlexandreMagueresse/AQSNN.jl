function Domain(::Val{M}, ::AbstractVector) where {M}
  @notimplemented
end

function Domain(
  ::Val{0}, store::AbstractVector{<:AbstractVector{T}};
  kwargs...
) where {T}
  # Check length
  msg = "A 0-dimensional domain is defined by one and only one point."
  (length(store) != 1) && throw(AssertionError(msg))

  p, = store

  # Define normal
  E = length(p)
  n = SVector{E,T}(i == E ? 1 : 0 for i in 1:E)

  context = Context0D{T,E}(n)
  Point(context, store[1:1], 1:1)
end

function Domain(
  ::Val{1}, store::AbstractVector{<:AbstractVector{T}};
  kwargs...
) where {T}
  ε = 10 * eps(T)

  # Check length
  msg = "A 1-dimensional domain is defined by two and only two different points."
  (length(store) != 2) && throw(AssertionError(msg))

  p₋, p₊ = store

  # Make sure that the domain is not too small
  msg = "Vertices are too close for this type (> 10 ε($T) = $ε)."
  (norm₋(p₊, p₋) < ε) && throw(AssertionError(msg))

  # Define direction
  d = normalize(p₊ .- p₋)

  # Define normal
  E = length(d)
  if E == 1
    n = SVector{2,T}(0, 1)
  elseif E == 2
    n = SVector{2,T}(-d[2], d[1])
  else
    # Sample random vector and orthogonalise
    n = SVector{E,T}(randn(T, E))
    n = n .- dot(n, d) .* d
    # If we have very bad luck, n was sampled colinear to d
    while norm(n) < ε
      n = SVector{E,T}(randn(T, E))
      n = n .- dot(n, d) .* d
    end
    # Normalise and orthogonalise again to reduce roundoff error
    n = normalize(n)
    n = normalize(n .- dot(n, d) .* d)
  end
  # Force last component to be positive
  if n[end] < 0
    n *= -1
  end

  context = Context1D{T,E}(d, n)
  Segment(context, store[1:2], 1:2)
end

function Domain(
  ::Val{2}, store::AbstractVector{<:AbstractVector{T}};
  checkIntersection::Bool=true, checkConvexity::Bool=true, kwargs...
) where {T}
  ε = 10 * eps(T)

  # Check length
  msg = "A 2-dimensional domain is defined by at least three points."
  L = length(store)
  (L < 3) && throw(AssertionError(msg))

  # Find two different points
  p₀, = store
  E = length(p₀)

  i = 2
  while (i <= L) && norm₋(store[i], p₀) < ε
    i += 1
  end
  msg = "Vertices are too close for this type (> 10 ε($T) = $ε)."
  (i > L) && throw(AssertionError(msg))

  # Define first direction vector
  u = normalize(store[i] .- p₀)

  # Find third point not aligned
  i += 1
  while i <= L
    p = store[i]
    # Find coordinates in (p₀, u)
    s = dot₋(p, p₀, u)
    # Compute distance between p and p₀ + α u
    dist² = T(0)
    @inbounds @simd for j in 1:E
      dist² += abs2(p[j] - p₀[j] - s * u[j])
    end
    (dist² > ε) && break
    i += 1
  end
  msg = "Vertices are too close for this type (> 10 ε($T) = $ε)."
  (i > L) && throw(AssertionError(msg))

  # Define second direction vector, compute scaling and orthonormalise basis
  v = store[i] .- p₀
  v = normalize(v .- dot(u, v) .* u)
  v = normalize(v .- dot(u, v) .* u)

  # Check coplanarity and compute coordinates of (p - p₀) in (u, v)
  λs = Vector{SVector{2,T}}()
  sizehint!(λs, L)
  for p in store
    # Find coordinates in (u, v)
    s, t = dot₋(p, p₀, u), dot₋(p, p₀, v)
    # Compute distance between v and p₀ + s * u + t * v
    dist² = T(0)
    @inbounds @simd for i in 1:E
      dist² += abs2(p[i] - p₀[i] - s * u[i] - t * v[i])
    end
    msg = "All the points are not coplanar."
    (dist² > ε) && throw(AssertionError(msg))

    push!(λs, SVector{2,T}(s, t))
  end

  # Check non self-intersection (extremely slow)
  if checkIntersection
    for i in 1:L-2
      j = (i == L) ? 1 : i + 1
      (s₁₋, t₁₋), (s₁₊, t₁₊) = λs[i], λs[j]
      s̄₁, Δs₁ = middif(s₁₋, s₁₊)
      t̄₁, Δt₁ = middif(t₁₋, t₁₊)

      # We always have i < j < k < l but we can have l = i
      # iff i = 1, k = L: (1, 2) with (L, 1)
      # We need to skip this case because it is not a problem
      for k in i+2:L
        l = (k == L) ? 1 : k + 1
        (l == i) && continue

        (s₂₋, t₂₋), (s₂₊, t₂₊) = λs[k], λs[l]
        s̄₂, Δs₂ = middif(s₂₋, s₂₊)
        t̄₂, Δt₂ = middif(t₂₋, t₂₊)

        # (s̄, t̄)₁ + α (Δs, Δt)₁ = (s̄, t̄)₂ + β (Δs, Δt)₂
        # Project on (u, v)
        # s̄₁ + α Δs₁ = s̄₂ + β Δs₂
        # t̄₁ + α Δt₁ = t̄₂ + β Δt₂

        m₁₁, m₁₂, f₁ = Δs₁, -Δs₂, s̄₂ - s̄₁
        m₂₁, m₂₂, f₂ = Δt₁, -Δt₂, t̄₂ - t̄₁
        α, β, χ = sol2(m₁₁, m₁₂, m₂₁, m₂₂, f₁, f₂)
        χ && continue

        if abs(α) < 1 && abs(β) < 1
          msg = "A 2-dimensional domain cannot be self-intersecting."
          throw(AssertionError(msg))
        end
      end
    end
  end

  # Remove aligned points
  inds = [1]

  i₋, (s₋, t₋) = 1, λs[1]
  i₀, (s₀, t₀) = 2, λs[2]
  for i in 3:L+1
    (i == L + 1) && (i = 1)
    i₊, (s₊, t₊) = i, λs[i]

    # Check whether (p₋ - p₀) and (p₊ - p₀) are collinear
    # (s, t)₀ = (s̄, t̄) + α (Δs, Δt)
    # Project on (u, v)
    # Δs (t₀ - t̆) - Δt (s₀ - s̄) = 0
    s̄, Δs = middif(s₋, s₊)
    t̄, Δt = middif(t₋, t₊)

    if abs(det2(Δs, Δt, s₀ - s̄, t₀ - t̄)) < ε
      i₀, s₀, t₀ = i₊, s₊, t₊
    else
      push!(inds, i₀)
      i₋, s₋, t₋ = i₀, s₀, t₀
      i₀, s₀, t₀ = i₊, s₊, t₊
    end
  end

  # Check first point
  if length(inds) > 3
    (s₋, t₋), (s₀, t₀), (s₊, t₊) = λs[inds[end]], λs[inds[1]], λs[inds[2]]
    s̄, Δs = middif(s₋, s₊)
    t̄, Δt = middif(t₋, t₊)
    if abs(det2(Δs, Δt, s₀ - s̄, t₀ - t̄)) < ε
      popfirst!(inds)
    end
  end

  # Convexity (somewhat slow)
  if checkConvexity
    convex = true
    L = length(inds)
    for i in 1:L
      j = (i == L) ? 1 : i + 1
      k = (j == L) ? 1 : j + 1
      (s₋, t₋), (s₀, t₀), (s₊, t₊) = λs[inds[i]], λs[inds[j]], λs[inds[k]]
      Δs₋, Δt₋ = s₋ - s₀, t₋ - t₀
      Δs₊, Δt₊ = s₊ - s₀, t₊ - t₀
      if det2(Δs₊, Δt₊, Δs₋, Δt₋) < 0
        convex = false
        break
      end
    end
  else
    convex = false
  end

  # Make basis direct: swap basis if the sign(det) does not match convexity
  if !convex
    (s₋, t₋), (s₀, t₀), (s₊, t₊) = λs[inds[end]], λs[inds[1]], λs[inds[2]]
    Δs₀, Δt₀ = (s₊ + s₋) / 2 - s₀, (t₊ + t₋) / 2 - s₀

    # Take direction as the bisector
    # [(p₊ - p₀) + (p₋ - p₀)] / 2 = (Δs₀, Δt₀)
    # Half-Line: (s₀, t₀) + α (Δs₀, Δt₀), α >= 0
    # Edge: (s̄, t̄) + β (Δs, Δt), -1 <= β <= 1
    # Project in (u, v)
    # Δs₀ α - Δs β = s̄ - s₀
    # Δt₀ α - Δt β = t̄ - t₀

    # Count the parity of edges that intersect with the half-line
    i = 2
    isEven = true
    while i <= L - 1
      (s₋, t₋), (s₊, t₊) = λs[inds[i]], λs[inds[i+1]]
      s̄, Δs = middif(s₋, s₊)
      t̄, Δt = middif(t₋, t₊)

      m₁₁, m₁₂, f₁ = Δs₀, -Δs, s̄ - s₀
      m₂₁, m₂₂, f₂ = Δt₀, -Δt, t̄ - t₀
      α, β, χ = sol2(m₁₁, m₁₂, m₂₁, m₂₂, f₁, f₂)

      # Two cases:
      # * Edge coincides with half-line and is located "after" p₀
      #   sign(s̄ - s₀) = sign(Δs₀) or equivalently sign(t̄ - t₀) = sign(Δt₀)
      # * Edge intersects with half-line once
      coincide = χ && ((sign(f₁) == sign(m₁₁)) || (sign(f₂) == sign(m₂₁)))
      intersect = (α >= 0) && (abs(β) <= 1)
      if coincide || intersect
        isEven = !isEven
        # Skip next edge if intersected at vertex (has to be at p₋ by induction)
        if χ || (abs(β + 1) < ε)
          i += 1
        end
      end
      i += 1
    end

    # Compute sign of angle
    (s₋, t₋), (s₀, t₀), (s₊, t₊) = λs[inds[end]], λs[inds[1]], λs[inds[2]]
    Δs₋, Δt₋ = s₋ - s₀, t₋ - t₀
    Δs₊, Δt₊ = s₊ - s₀, t₊ - t₀
    s = det2(Δs₊, Δt₊, Δs₋, Δt₋)

    # Intersections odd =>   convex, s has to be > 0
    # Intersections even => concave, s has to be < 0
    # (!isEven && (s < 0)) || (isEven && (s > 0)) => swap
    # isEven != (s < 0)

    if isEven != (s < 0)
      u, v = v, u
    end
  end

  if E == 2
    n = SVector{3,T}(0, 0, 1)
  else
    # Sample random vector and orthogonalise
    n = SVector{E,T}(rand(T, E))
    n = n .- dot(n, u) .* u .- dot(n, v) .* v
    # If we have very bad luck, n was sampled in span(u, v)
    while norm(n) < ε
      n = SVector{E,T}(randn(T, E))
      n = n .- dot(n, u) .* u .- dot(n, v) .* v
    end
    # Normalise and orthogonalise again to reduce roundoff error
    n = normalize(n)
    n = normalize(n .- dot(n, u) .* u .- dot(n, v) .* v)
  end
  # Force last component to be positive
  if n[end] < 0
    n *= -1
  end

  context = Context2D{T,E}(u, v, n)

  # Classify polygon
  if convex
    if L == 3
      Triangle(context, store[inds], 1:L)
    elseif L == 4
      Quadrangle(context, store[inds], 1:L)
    else
      ConvexPolytope(context, store[inds], 1:L)
    end
  else
    GeneralPolytope(context, store[inds], 1:L)
  end
end

function Domain(store; kwargs...)
  E = length(store[1])
  Domain(Val(E), store; kwargs...)
end

###################
# CartesianDomain #
###################
function CartesianDomain(x₋::T1, x₊::T2) where {T1<:Real,T2<:Real}
  T = promote_type(T1, T2, Float32)
  SV1 = SVector{1,T}
  Domain([SV1(x₋), SV1(x₊)])
end

function CartesianDomain(x₋::T1, x₊::T2, y₋::T3, y₊::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
  T = promote_type(T1, T2, T3, T4, Float32)
  SV2 = SVector{2,T}
  Domain([SV2(x₋, y₋), SV2(x₊, y₋), SV2(x₊, y₊), SV2(x₋, y₊)])
end

#################
# RegularDomain #
#################
function RegularDomain(r::T1, N::Int) where {T1<:Real}
  T = promote_type(T1, Float32)
  points = Vector{SVector{2,T}}()
  sizehint!(points, N)
  θ = T(2 / N)
  for n in 0:N-1
    s, c = sincospi(n * θ)
    push!(points, SVector{2,T}(r * c, r * s))
  end
  Domain(points)
end
