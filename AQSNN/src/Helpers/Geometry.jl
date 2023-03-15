function middif(x, y)
  (x .+ y) / 2, (y .- x) / 2
end

function inTriangle(p, a, b, c)
  uu = norm2₋(b, a)
  uv = dot₋₋(b, a, c, a)
  vv = norm2₋(c, a)

  pu = dot₋₋(p, a, b, a)
  pv = dot₋₋(p, a, c, a)

  s, t, χ = sol2(uu, uv, uv, vv, pu, pv)
  if !χ
    (0 <= s + t <= 1) && (0 <= s <= 1) && (0 <= t <= 1)
  else
    # Degenerate cases: (a, b, c) are aligned
    L = length(p)

    # p ∈ [a, b], α = pu / uu
    all(i -> uu * (p[i] - a[i]) == pu * (b[i] - a[i]), 1:L) && return true

    # p ∈ [a, c], α = pv / vv
    all(i -> vv * (p[i] - a[i]) == pv * (c[i] - a[i]), 1:L) && return true

    # p ∈ [b, c],  α = pw / ww
    ww = norm2₋(c, b)
    pw = dot₋₋(p, b, c, b)
    all(i -> ww * (p[i] - b[i]) == pw * (c[i] - b[i]), 1:L) && return true

    return false
  end
end

function areaTriangle(p, q, r, u, v)
  if length(p) == 2
    xp, yp = p
    xq, yq = q
    xr, yr = r
    m = det2(xp, yp, xq, yq)
    m += det2(xq, yq, xr, yr)
    m += det2(xr, yr, xp, yp)
  else
    s₋, t₋ = dot₋(q, p, u), dot₋(q, p, v)
    s₊, t₊ = dot₋(r, p, u), dot₋(r, p, v)
    m = det2(s₋, t₋, s₊, t₊)
  end
  abs(m) / 2
end

function convexAngle(p₋, p₀, p₊, u, v)
  dot₊u = dot₋(p₊, p₀, u)
  dot₊v = dot₋(p₊, p₀, v)
  dot₋u = dot₋(p₋, p₀, u)
  dot₋v = dot₋(p₋, p₀, v)
  det2(dot₊u, dot₊v, dot₋u, dot₋v) > 0
end

function cosAngle(p₋, p₀, p₊)
  T = promote_type(eltype(p₋), eltype(p₀), eltype(p₊))
  ε = 10 * eps(T)
  nΔp₊, nΔp₋ = norm₋(p₊, p₀), norm₋(p₋, p₀)
  if nΔp₊ < ε || nΔp₋ < ε
    zero(eltype(p₊))
  else
    dot₋₋(p₊, p₀, p₋, p₀) / nΔp₊ / nΔp₋
  end
end
