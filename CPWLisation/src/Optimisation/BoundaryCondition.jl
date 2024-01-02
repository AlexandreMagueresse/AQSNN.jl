abstract type BoundaryCondition end

"""
Force the approximation to coincide with the tangent at infinity.
"""
struct Asymptote <: BoundaryCondition
end
Base.isfinite(::Asymptote) = false

"""
Force r(x) = y, where r approximates ρ and usually y = ρ(x).
"""
struct PointValue{T} <: BoundaryCondition
  x::T
  y::T
end
Base.isfinite(::PointValue) = true

"""
Leave r(x) free, where r approximates ρ.
"""
struct PointFree{T} <: BoundaryCondition
  x::T
end
Base.isfinite(::PointFree) = true

"""
Force r'(x) = slope, where r approximates ρ and usually slope = ρ'(x).
"""
struct PointSlope{T} <: BoundaryCondition
  x::T
  slope::T
end
Base.isfinite(::PointSlope) = true

"""
Force r(x) = y and r'(x) = slope, where r approximates ρ and usually y = ρ(x)
and slope = ρ'(x).
"""
struct PointTangent{T} <: BoundaryCondition
  x::T
  y::T
  slope::T
end
Base.isfinite(::PointTangent) = true

"""
Return the value of r at the i-th free point, where r approximates ρ, bc₋ and
c₊ are the boundary conditions, xs are the coordinates of the free points and
ys are the values at the free points.
"""
function getPoint(ρ, bc₋, bc₊, xs, ys, i)
  N = length(xs)
  if i == 1
    if isfinite(bc₋)
      x = bc₋.x
      if isa(bc₋, PointFree)
        y = ys[i]
      elseif isa(bc₋, PointValue) || isa(bc₋, PointTangent)
        y = bc₋.y
      elseif isa(bc₋, PointSlope)
        xx, yy = getPoint(ρ, bc₋, bc₊, xs, ys, i + 1)
        y = yy - bc₋.slope * (xx - bc₋.x)
      else
        @notreachable
      end
    else
      α, β = ϕ₋(ρ)
      x = xs[i]
      y = α * x + β
    end
  elseif i == 2 && isa(bc₋, PointTangent)
    x = xs[i]
    y = bc₋.y + bc₋.slope * (x - bc₋.x)
  elseif i == N - 1 && isa(bc₊, PointTangent)
    x = xs[i]
    y = bc₊.y - bc₊.slope * (bc₊.x - x)
  elseif i == N
    if isfinite(bc₊)
      x = bc₊.x
      if isa(bc₊, PointFree)
        y = ys[i]
      elseif isa(bc₊, PointValue) || isa(bc₊, PointTangent)
        y = bc₊.y
      elseif isa(bc₊, PointSlope)
        xx, yy = getPoint(ρ, bc₋, bc₊, xs, ys, i - 1)
        y = yy + bc₊.slope * (bc₊.x - xx)
      else
        @notreachable
      end
    else
      α, β = ϕ₊(ρ)
      x = xs[i]
      y = α * x + β
    end
  else
    x, y = xs[i], ys[i]
  end
  x, y
end
