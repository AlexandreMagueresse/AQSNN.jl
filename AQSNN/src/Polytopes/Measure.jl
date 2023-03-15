function measure(::AbstractPolytope)
  @abstractmethod
end

function measure(::AbstractPolytope{T,0}) where {T}
  one(T)
end

function measure(polytope::AbstractPolytope{T,1}) where {T}
  p₋, p₊ = getVertex(polytope, 1), getVertex(polytope, 2)
  norm₋(p₊, p₋)
end

function measure(polytope::AbstractPolytope{T,1,1}) where {T}
  p₋, p₊ = getVertex(polytope, 1), getVertex(polytope, 2)
  abs(p₊[1] - p₋[1])
end

function measure(polytope::AbstractPolytope{T,2}) where {T}
  p₀ = getVertex(polytope, 1)
  u, v = getBasis(polytope)

  m = zero(T)
  L = length(polytope)

  p = getVertex(polytope, 2)
  s₋, t₋ = dot₋(p, p₀, u), dot₋(p, p₀, v)
  for i in 3:L
    p = getVertex(polytope, i)
    s₊, t₊ = dot₋(p, p₀, u), dot₋(p, p₀, v)
    m += det2(s₋, t₋, s₊, t₊)
    s₋, t₋ = s₊, t₊
  end

  abs(m) / 2
end

function measure(polytope::AbstractPolytope{T,2,2}) where {T}
  m = zero(T)
  L = length(polytope)
  s₋, t₋ = getVertex(polytope, 1)
  for i in 2:L
    s₊, t₊ = getVertex(polytope, i)
    m += det2(s₋, t₋, s₊, t₊)
    s₋, t₋ = s₊, t₊
  end
  s₊, t₊ = getVertex(polytope, 1)
  m += det2(s₋, t₋, s₊, t₊)

  abs(m) / 2
end
