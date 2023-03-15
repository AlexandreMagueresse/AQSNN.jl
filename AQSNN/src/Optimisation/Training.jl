##########
# Struct #
##########
struct Training{QΩ,QΓ,O<:AbstractOptimiser}
  epochs::Int
  β::Function
  dΩ::QΩ
  dΓ::QΓ
  refreshRate::Int
  optimisers::Vector{O}

  function Training(
    epochs::Int,
    β::Function,
    dΩ::QΩ,
    dΓ::QΓ,
    refreshRate::Int,
    optimisers::Vector{O}
  ) where {QΩ,QΓ,O<:AbstractOptimiser}
    new{QΩ,QΓ,O}(epochs, β, dΩ, dΓ, refreshRate, optimisers)
  end
end

############
# Training #
############
function train!(
  u::Sequential,
  training::Training,
  objective::Objective,
  callbacks::Vector{F}=Vector{Function}();
  globalEpoch::Int=0
) where {F<:Function}
  # Initialise optimisers and quadratures
  for optimiser in training.optimisers
    initialise!(optimiser)
  end

  if globalEpoch == 0 || training.refreshRate != -1
    initialise!(training.dΩ)
    initialise!(training.dΓ)
  end

  for epoch in 1:training.epochs
    globalEpoch += 1
    # Refresh quadratures
    if (training.refreshRate != -1) && (mod(epoch, training.refreshRate) == 0)
      initialise!(training.dΩ)
      initialise!(training.dΓ)
    end

    # Run optimisers
    for optimiser in training.optimisers
      step!(optimiser, u, objective, training.β(epoch), training.dΩ, training.dΓ)
    end

    # Run callbacks
    for callback in callbacks
      stop = callback(globalEpoch, u, objective, training.β(epoch), training.dΩ, training.dΓ)
      isa(stop, Bool) && stop && return
    end
  end
end

function train!(
  u::Sequential,
  trainings::Vector{TR},
  objective::Objective,
  callbacks::Vector{F}=Vector{Function}()
) where {TR<:Training,F<:Function}
  globalEpoch = 0
  for (i, training) in enumerate(trainings)
    println("Phase #$i")
    train!(u, training, objective, callbacks, globalEpoch=globalEpoch)
    globalEpoch += training.epochs
  end
end
