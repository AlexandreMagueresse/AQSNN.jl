using Test
using StaticArrays
using LinearAlgebra

using AQSNN

T = BigFloat
SV2 = SVector{2,T}
ε = 1.0e-36

function polynomial(ps, i, j)
  X = view(ps, 1:1, :)
  Y = view(ps, 2:2, :)
  X .^ i .* Y .^ j
end

tri = Domain(Val(2), [SV2(0, 0), SV2(1, 0), SV2(0, 1)])
for o in 0:20
  quadrature = ReferenceQuadrature(tri, o)
  w, p = transport(quadrature, tri)
  maxerror = zero(T)
  for i in 0:o
    for j in 0:o-i
      ∫numerical = sum(w' .* polynomial(p, i, j))
      ∫exact = factorial(big(i)) / factorial(big(i + j + 2)) * factorial(big(j))
      maxerror = max(maxerror, abs(∫numerical - ∫exact))
    end
  end
  println("tri $o $maxerror")
  @test maxerror <= ε
end

quad = Domain(Val(2), [SV2(-1, -1), SV2(+1, -1), SV2(+1, +1), SV2(-1, +1)])
for o in 0:20
  quadrature = ReferenceQuadrature(quad, o)
  w, p = transport(quadrature, quad)
  maxerror = zero(T)
  for i in 0:o
    for j in 0:o-i
      ∫numerical = sum(w' .* polynomial(p, i, j))
      ∫exact = (mod(i, 2) == 1 || mod(j, 2) == 1) ? big(0) : big(4) / big(i + 1) / big(j + 1)
      maxerror = max(maxerror, abs(∫numerical - ∫exact))
    end
  end
  println("quad $o $maxerror")
  @test maxerror <= ε
end
