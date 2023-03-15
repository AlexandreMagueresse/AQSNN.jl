function ReLi₂(x)
  cnum = (
    0.9999999999999999502e+0,
    -2.6883926818565423430e+0,
    2.6477222699473109692e+0,
    -1.1538559607887416355e+0,
    2.0886077795020607837e-1,
    -1.0859777134152463084e-2
  )
  cden = (
    1.0000000000000000000e+0,
    -2.9383926818565635485e+0,
    3.2712093293018635389e+0,
    -1.7076702173954289421e+0,
    4.1596017228400603836e-1,
    -3.9801343754084482956e-2,
    8.2743668974466659035e-4
  )
  ζ₂ = abs2(pi) / 6

  # transform to [0, 0.5]
  (y¹, rest, sgn) = if x < -1
    l = log(1 - x)
    (1 / (1 - x), -ζ₂ + l * (l / 2 - log(-x)), 1)
  elseif x == -1
    return -ζ₂ / 2
  elseif x < 0
    (x / (x - 1), -abs2(log1p(-x)) / 2, -1)
  elseif x == 0
    return 0
  elseif 2 * x < 1
    (x, 0, 1)
  elseif x < 1
    (1 - x, ζ₂ - log(x) * log(1 - x), -1)
  elseif x == 1
    return ζ₂
  elseif x < 2
    l = log(x)
    (1 - 1 / x, ζ₂ - l * (log(1 - 1 / x) + l / 2), 1)
  else
    (1 / x, 2 * ζ₂ - abs2(log(x)) / 2, -1)
  end

  y² = abs2(y¹)
  y⁴ = abs2(y²)

  num = cnum[1] + y¹ * cnum[2] + y² * (cnum[3] + y¹ * cnum[4]) + y⁴ * (cnum[5] + y¹ * cnum[6])
  den = cden[1] + y¹ * cden[2] + y² * (cden[3] + y¹ * cden[4]) + y⁴ * (cden[5] + y¹ * cden[6] + y² * cden[7])

  rest + sgn * y¹ * num / den
end

function ∂ReLi₂(x)
  -log(1 - x) / x
end

@primitive ReLi₂(x), dy (dy .* ∂ReLi₂.(x))
