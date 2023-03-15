abstract type BoundaryCondition end

struct Asymptote <: BoundaryCondition
end
Base.isfinite(::Asymptote) = false

struct PointValue{T} <: BoundaryCondition
  x::T
  y::T
end
Base.isfinite(::PointValue) = true

struct PointFree{T} <: BoundaryCondition
  x::T
end
Base.isfinite(::PointFree) = true

struct PointSlope{T} <: BoundaryCondition
  x::T
  slope::T
end
Base.isfinite(::PointSlope) = true

struct PointTangent{T} <: BoundaryCondition
  x::T
  y::T
  slope::T
end
Base.isfinite(::PointTangent) = true

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
