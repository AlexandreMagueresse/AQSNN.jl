using Plots
using LaTeXStrings

∫ReLU = [
  0.001333333333333, # 2
  0.000513395215721, # 3
  0.0001401571238538778, # 5 - 1
  5.489573634218358e-5,
  2.6071079825542862e-5,
  1.4009442321515431e-5,
  8.212050258324167e-6, # 13 - 5
  5.13671407959217e-6,
  3.3795258591412744e-6,
  2.3153245245429715e-6,
  1.639832352974955e-6,
  1.1941360291721246e-6, # 23 - 10
  8.903357647501994e-7,
  6.774319974945819e-7,
  5.246135804229523e-7,
  4.126086858200497e-7,
  3.2899433977666097e-7, # 33 - 15
  2.6554795141166566e-7,
  2.166985185308995e-7,
  1.7859190214703415e-7,
  1.4851238632393295e-7,
  1.2451047183213306e-7, # 43 - 20
  1.0537002544399083e-7,
  8.987707985463734e-8,
  7.70441405444124e-8,
  6.76794275417208e-8, # not converged
  5.880100614360232e-8, # 53 - 25, not converged
]
∫ReLU .= sqrt.(2 .* ∫ReLU)
nsReLU = collect(3:2:(2*length(∫ReLU)-1))
pushfirst!(nsReLU, 2)

∫Tanh = [
  0.024787882043, # 3
  0.0027884387470134137, # 5 - 1
  0.000697059865459683,
  0.0002506291330785348,
  0.00011125515976029721,
  5.670430863809585e-5, # 13 - 5
  3.186897403130158e-5,
  1.926513621899298e-5,
  1.2322397051573906e-5,
  8.24475233153531e-6,
  5.723074636829634e-6, # 23 - 10
  4.09607319393054e-6,
  3.008391675874665e-6,
  2.2589976111899944e-6,
  1.7291122235592519e-6,
  1.345894417862851e-6, # 33 - 15
  1.0632101254352433e-6,
  8.510037201561233e-7,
  6.892004528044429e-7,
  5.640915843889527e-7,
  4.6612807825430054e-7, # 43 - 20
  3.8853851045453557e-7,
  3.264428306683934e-7,
  2.7627194408250914e-7,
  2.3538021474409443e-7,
  2.017822001026148e-7 # 53 - 25
]
∫Tanh .= sqrt.(2 .* ∫Tanh)
nsTanh = 3:2:(2*length(∫Tanh)+1)

plot(log10.(nsTanh), log10.(∫Tanh), label=L"\tanh")
plot!(log10.(nsReLU), log10.(∫ReLU), label=L"\mathrm{ReLU}_\varepsilon")
plot!(log10.(nsReLU), -2 .* log10.(nsReLU), label="Slope -2")
plot!(legend=:topright, legendfontsize=10)
plot!(xlabel=L"\log\ n")
plot!(ylabel=L"\log\ \|f - \pi_n[f]\|_{L^2(\mathbb{R})}")