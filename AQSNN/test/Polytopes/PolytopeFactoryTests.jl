module GeneralDomainTests

using Test
using StaticArrays
using LinearAlgebra

using AQSNN

SV1 = SVector{1,Float32}
SV2 = SVector{2,Float32}
SV3 = SVector{3,Float32}
ε = 10 * eps(Float32)
ε² = eps(Float32)

###############
# Dimension 0 #
###############
for ps in ([SV1(1)], [SV2(1, 2)], [SV3(1, 2, 3)])
  Ω = Domain(Val(0), ps)

  @test eltype(Ω) == Float32
  @test mandim(Ω) == 0
  @test embdim(Ω) == length(ps[1])
  @test isempty(getBasis(Ω))
  normal = getNormal(Ω)
  @test all(i -> (i == lastindex(normal)) ? (normal[i] == 1) : (normal[i] == 0), eachindex(normal))
  @test_throws Exception Boundary(Ω)
end

###############
# Dimension 1 #
###############
for ps in ([SV1(-1), SV1(+1)], [SV2(1, 3), SV2(2, 4)], [SV3(-1, 0, +1), SV3(0, 0, 0)])
  Ω = Domain(Val(1), ps)

  @test eltype(Ω) == Float32
  @test mandim(Ω) == 1
  @test embdim(Ω) == length(ps[1])
  @test length(getBasis(Ω)) == 1
  v, = getBasis(Ω)
  n = getNormal(Ω)
  @test abs(norm(v) - 1) <= ε
  @test abs(norm(n) - 1) <= ε
  if embdim(Ω) == 1
    @test abs(n[1]) <= ε
  else
    @test abs(dot(n, v)) <= ε
  end
  Γ = Boundary(Ω)
  for γ in Γ
    nγ = getNormal(γ)
    @test abs(norm(nγ) - 1) <= ε
    if embdim(Ω) == 1
      @test abs(nγ[1] * n[1]) <= ε
    else
      @test abs(dot(nγ, n)) <= ε
    end
  end
end

@test_throws AssertionError Domain(Val(1), [SV1(ε²), SV1(ε²)])
@test_throws AssertionError Domain(Val(1), [SV2(ε², ε²), SV2(ε², ε²)])
@test_throws AssertionError Domain(Val(1), [SV3(ε², ε², ε²), SV3(ε², ε², ε²)])

###############
# Dimension 2 #
###############
for ps in (
  [SV2(0, 0), SV2(1, 0), SV2(0, 1)],
  [SV3(1, 0, 0), SV3(0, 1, 0), SV3(0, 0, 1)],
  [SV2(-1, -1), SV2(+1, -1), SV2(+1, +1), SV2(-1, +1)],
  [SV3(-1, -1, +1), SV3(+1, -1, +1), SV3(+1, +1, -1), SV3(-1, +1, -1)],
  [SV2(-1, 0), SV2(0, 2), SV2(+1, 0), SV2(0, 1)]
)
  Ω = Domain(Val(2), ps)

  @test eltype(Ω) == Float32
  @test mandim(Ω) == 2
  @test embdim(Ω) == length(ps[1])
  @test length(getBasis(Ω)) == 2
  v₁, v₂ = getBasis(Ω)
  n = getNormal(Ω)
  @test abs(norm(v₁) - 1) <= ε
  @test abs(norm(v₂) - 1) <= ε
  @test abs(norm(n) - 1) <= ε
  @test abs(dot(view(n, 1:embdim(Ω)), v₁)) <= ε
  @test abs(dot(view(n, 1:embdim(Ω)), v₂)) <= ε
  @test abs(dot(v₁, v₂)) <= ε

  Γ = Boundary(Ω)
  for γ in Γ
    nγ = getNormal(γ)
    @test abs(norm(nγ) - 1) <= ε
    @test abs(dot(nγ, view(n, 1:embdim(Ω)))) <= ε
  end
end

@test_throws AssertionError Domain(Val(2), [SV2(-1, -1), SV2(+1, +1), SV2(+1, -1), SV2(-1, +1)])
@test_throws AssertionError Domain(Val(2), [SV3(1, 0, 0), SV3(0, 1, 0), SV3(0, 0, 1), SV3(0, 0, 0)])

end
