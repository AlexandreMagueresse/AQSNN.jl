function dot₋(a, b, u)
  r = zero(promote_type(eltype(a), eltype(b), eltype(u)))
  @inbounds @simd for i in eachindex(a, b, u)
    r += (a[i] - b[i]) * u[i]
  end
  r
end

function dot₊(a, b, u)
  r = zero(promote_type(eltype(a), eltype(b), eltype(u)))
  @inbounds @simd for i in eachindex(a, b, u)
    r += (a[i] + b[i]) * u[i]
  end
  r
end

function dot₋₋(a, b, c, d)
  r = zero(promote_type(eltype(a), eltype(b), eltype(c), eltype(d)))
  @inbounds @simd for i in eachindex(a, b, c, d)
    r += (a[i] - b[i]) * (c[i] - d[i])
  end
  r
end

function dot₋₊(a, b, c, d)
  r = zero(promote_type(eltype(a), eltype(b), eltype(c), eltype(d)))
  @inbounds @simd for i in eachindex(a, b, c, d)
    r += (a[i] - b[i]) * (c[i] + d[i])
  end
  r
end

function norm2₋(a, b)
  r = zero(promote_type(eltype(a), eltype(b)))
  @inbounds @simd for i in eachindex(a, b)
    r += abs2(a[i] - b[i])
  end
  r
end

function norm2₊(a, b)
  r = zero(promote_type(eltype(a), eltype(b)))
  @inbounds @simd for i in eachindex(a, b)
    r += abs2(a[i] + b[i])
  end
  r
end

function norm₋(a, b)
  sqrt(norm2₋(a, b))
end

# Non-allocating inverse of 2x2 systems
function det2(a, b, c, d)
  a * d - b * c
end

function sol2(a, b, c, d, e, f)
  T = promote_type(typeof(a), typeof(b), typeof(c), typeof(d), typeof(e), typeof(f))
  if a == 0 && c == 0
    x, y, error = T(NaN), T(NaN), true
  elseif abs(a) < abs(c)
    sol2(c, d, a, b, f, e)
  else
    r = c / a
    s = d - b * r
    if s == 0
      x, y, error = T(NaN), T(NaN), true
    else
      t = f - r * e
      y = t / s
      x = (e - b * y) / a
      x, y, false
    end
  end
end
