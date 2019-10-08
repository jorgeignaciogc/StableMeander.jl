using Distributions

struct StableMeander <: ContinuousUnivariateDistribution
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  StableMeander(α,β) = ( β<-1 || β>1 || (β==-1 && α<=1) || α<=0 || α>2 ) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") :
    β==1 && α<=1 ? Stable(α,1) :
    new(α,β,β*(α<=1 ? 1 : (α-2)/α),(1+β*(α<=1 ? 1 : (α-2)/α))/2)
end

import Distributions.params
function params(d::StableMeander)
  return (d.α,d.β,d.θ,d.ρ)
end

import Distributions.minimum
function minimum(d::StableMeander)
  return 0
end

import Distributions.maximum
function maximum(d::StableMeander)
  return Inf
end

import Distributions.insupport
function insupport(d::StableMeander,x::Real)
  return x>=0
end

import Distributions.mean
function mean(d::StableMeander)
  return pi/(sin(pi/d.α) * gamma(d.ρ + 1/d.α) * gamma(1 - d.ρ))
end

struct PreciseStableMeander <: Sampleable{Multivariate,Continuous}
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  d::Float64
  η::Float64
  δ::Float64
  γ::Float64
  κ::Float64
  Δ::Int64
  mAst::Int64
  ε::Float64
  PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int,ε::Float64) =
    (β < -1 || β > 1 || (β == -1 && α <= 1) || α <= 0 || α > 2) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") : (0 >= γ || γ >= α) ?
    error("Parameters' requirements unmet: α>γ>0") : (0 >= δ || δ >= d || d >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(αρ)>d>δ>0") : κ < 0 ?
    error("Parameters' requirements unmet: κ≥0") : Δ < 0 ?
    error("Parameters' requirements unmet: Δ≥0") : mAst < 0 ?
    error("Parameters' requirements unmet: m*≥0") : ε <= 0 ?
    error("Parameters' requirements unmet: ε>0") : new(α, β, β*(α <= 1 ? 1 : (α-2)/α),(1+β*(α <= 1 ? 1 : (α-2)/α))/2 , d,
      etaF(d*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2 , δ, γ, κ + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),
      log(2)*2/(3*etaF(d*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))), Δ, mAst, ε)
  PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int) = PreciseStableMeander(α,β,d,δ,γ,κ,Δ,mAst,.5^32)
  PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int) = PreciseStableMeander(α,β,d,δ,γ,κ,Δ,12)
  PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real) = PreciseStableMeander(α,β,d,δ,γ,κ,
    Int(ceil(33*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))))
  PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real) = PreciseStableMeander(α,β,d,δ,γ,4)
  PreciseStableMeander(α::Real,β::Real) = PreciseStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α)
  PreciseStableMeander(α::Real,β::Real,Δ::Integer,ε::Real) = PreciseStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,4,Δ,12,ε)
  PreciseStableMeander(α::Real,β::Real,ε::Real) = PreciseStableMeander(α,β,
    Int(ceil(abs(log(ε/2))/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),ε)
end

function precise_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int,ε::Real)
  return PreciseStableMeander(d.α,d.β,dr,δ,γ,κ,Δ,mAst,ε)
end

function precise_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int)
  return PreciseStableMeander(d.α,d.β,dr,δ,γ,κ,Δ,mAst)
end

function precise_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real,Δ::Int)
  return PreciseStableMeander(d.α,d.β,dr,δ,γ,κ,Δ)
end

function precise_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real)
  return PreciseStableMeander(d.α,d.β,dr,δ,γ,κ)
end

function precise_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real)
  return PreciseStableMeander(d.α,d.β,dr,δ,γ)
end

function precise_sampler(d::StableMeander,Δ::Integer,ε::Real)
  return PreciseStableMeander(d.α,d.β,Δ,ε)
end

function precise_sampler(d::StableMeander,ε::Real)
  return PreciseStableMeander(d.α,d.β,ε)
end

function precise_sampler(d::StableMeander)
  return PreciseStableMeander(d.α,d.β)
end

import Base.length
function length(d::PreciseStableMeander)
  return 3
end

using LambertW
# @Input: normalised drift d'=αρd∈(0,1)
# @Returns: normalised η'=η/(αρ)>0
function etaF(d::Real)
  return -lambertw(-d*exp(-d),-1)/d-1
end

# @Input: drift d, η, 1/(αρ+η), and a bound M we want to know if we exceed
# @Returns: a sample of 1{R_0 <= M}
function reflectedProcess(d::Real,et::Real,et1::Real,M::Real)
  T = rand(Exponential())/et
  if T <= M
    return true
  end
  Sn = [0.]
  while Sn[end] < T
    push!(Sn,Sn[end]-rand(Exponential())*et1+d)
  end
  return maximum(Sn[1:(end-1)]) <= M
end

# @Input: drift d, η, 1/(αρ+η), a bound M that it should not exceed,
# and another bound M1 we want to know if we exceed
# @Returns: a sample of 1{R_0 <= M1} given R_0<M
function reflectedProcess(d::Real,et::Real,et1::Real,M::Real,M1::Real)
  m = exp(-M-d)
  Sn = [0.]
  while true
    T = -log(rand(Uniform(m,1)))/et # Exponential conditioned on <= M+d, since T>M+d always yields a rejection
    if T <= M1
      return true
    end
    while Sn[end] < T
      push!(Sn,Sn[end]-rand(Exponential())*et1+d)
    end
    max = maximum(Sn[1:(end-1)])
    if max <= M
      return max <= M1
    else
      Sn = [0.]
    end
  end
end

# @Input: drift d, η, 1/(αρ+η), κ and an upper bound x
# @Returns: A simulated run of the random walk up to the hitting time of the barrier
# -2κ conditioned on never exceeding x
function downRW(d::Real,rar::Real,et::Real,et1::Real,κ::Real,x::Real)
  Sn = [0.]
  INCn = [0.]
  if isfinite(x)
    while true
      while Sn[end] > -2 *κ
        push!(INCn,rand(Exponential())*rar)
        push!(Sn,Sn[end]-INCn[end]+d)
      end
      if maximum(Sn) < x && reflectedProcess(d,et,et1,x-Sn[end])
        return (Sn[2:end],INCn[2:end])
      else
        Sn = [0.]
        INCn = [0.]
      end
    end
  else
    while Sn[end] > -2 *κ
      push!(INCn,rand(Exponential())*rar)
      push!(Sn,Sn[end]-INCn[end]+d)
    end
    return (Sn[2:end],INCn[2:end])
  end
end

# @Input: drift d, η, 1/(αρ+η), κ, an upper bound x that will be attained
# @Returns: A simulated run of the random walk up to the hitting time of the barrier
# κ conditioned on never exceeding x
function upRW(d::Real,et::Real,et1::Real,κ::Real,x::Real)
  Sn = [0.]
  INCn = [0.]
  if x == Inf
    while true
      while Sn[end] < κ
        push!(INCn,rand(Exponential())*et1)
        push!(Sn,Sn[end]-INCn[end]+d)
      end
      if rand(Uniform()) <= exp(-et*Sn[end])
        return (Sn[2:end],INCn[2:end])
      else
        Sn = [0.]
        INCn = [0.]
      end
    end
  else
    while true
      while Sn[end] < κ
        push!(INCn,rand(Exponential())*et1)
        push!(Sn,Sn[end]-INCn[end]+d)
      end
      if rand(Uniform()) <= exp(-et*Sn[end]) && reflectedProcess(d,et,et1,x-Sn[end]) # The condition maximum(Sn) < x1 is trivially satisfied by construction of κ
        return (Sn[2:end],INCn[2:end])
      else
        Sn = [0.]
        INCn = [0.]
      end
    end
  end
end

# Blanchet-Sigman algorithm
# @Input: drift d, η, 1/(αρ+η), κ and an upper bound x, last value of random walk s,
# @Returns: A simulated run of the random walk up to the first time we know an upper bound for the walk,
# the increments of the RW, the new upper bound, and the last time at which we may compute D_n
function BSAlgorithm(d::Real,et::Real,et1::Real,κ::Real,x::Real,s::Real)
  Sn = [Float64(s)]
  INCn = [0.]
  x1 = x
  t = 0

  while true
    # Step 1
    (Sn1,INCn1) = downRW(d,et,et1,κ,x1-Sn[end])
    # Step 2
    append!(Sn,Sn[end] .+ Sn1)
    append!(INCn,INCn1)
    t = t+length(Sn1)
    # Step 3 and 4
    if reflectedProcess(d,et,et1,x1-Sn[end],κ)
      x1 = Sn[end]+κ
      t = t-length(Sn1)
      break
    else
      (Sn1,INCn1) = upRW(d,et,et1,κ,x1-Sn[end])
      append!(Sn,Sn[end] .+ Sn1)
      append!(INCn,INCn1)
      t = t+length(Sn1)
    end
  end

  return (Sn[2:end],INCn[2:end],x1,t)
end

# Algorithm 3 - 1 (unconditional probabilities)
# @Input: S^+(α,ρ), δ, dega = δ*γ, dega2 = -1/(1-exp(-dega)), mom = E(S_1^γ), shift = m-k >= m*-1
function chiStable1(d::PositiveStable,δ::Real,dega::Real,dega2::Real,mom::Real,shift::Integer)
  U = rand(Uniform())
  n = shift
  S = Float64[]
  while true
    n = n+1
    aux = exp(δ*n)
    p = cdf(d,aux) # 1-pn
    aux1 = exp(-(n+1)*dega)*mom
    q = p*exp(dega2*aux1/(1-aux1)) # qn
    if U > p
      push!(S,cond_rand(d,aux,Inf))
      U = (U-p)/(1-p)
    elseif U < q
      push!(S,cond_rand(d,0,aux))
      return S
    else
      push!(S,cond_rand(d,0,aux))
      U = U/p
    end
  end
end

# Algorithm 3 - 2 (conditional probabilities)
# @Input: S^+(α,ρ), δ, dega = δ*γ, dega2 = -1/(1-exp(-dega)), mom = E(S_1^γ)
function chiStable2(d::PositiveStable,δ::Real,dega::Real,dega2::Real,mom::Real,shift::Integer)
  U = rand(Uniform())
  n = shift
  S = Float64[]
  while true
    n = n+1
    aux = exp(δ*n)
    aux0 = exp(δ*(n+1))
    p = cdf(d,aux)/cdf(d,aux0) # 1-pn
    aux1 = exp(-(n+1)*dega)*mom
    q = p*exp(dega2*aux1/(1 -aux1)) # qn
    if U > p
      push!(S,cond_rand(d,aux,aux0))
      U = (U-p)/(1 -p)
    elseif U < q
      push!(S,cond_rand(d,0,aux))
      return S
    else
      push!(S,cond_rand(d,0,aux))
      U = U/p
    end
  end
end

import Distributions.rand

# Algorithm 9
# @Input: PreciseStableMeander
# @Output: A random sample from a the law d, σ
function rand(d::PreciseStableMeander)
  ar = d.α*d.ρ
  et1 = 1 /(ar+d.η)
  ra = 1 /d.α
  rar = 1 /ar
  # Conditionally positive stable distribution
  dPos = PositiveStable(d.α,d.β)
  sep = d.d - d.δ
  e2 = 1 /(1 -exp(-sep))
  mom = mellin(dPos,d.γ)
  dega = d.δ*d.γ
  dega2 = -1 /(1 -exp(-dega))
  mAst = d.mAst + Int(ceil( max(0,log(mom)/dega) + rar ))
  # θ sequence
  U = Float64[]
  # Line 1
  x = Inf
  σ = 0
  # Line 2
  lagU = rand(Beta(d.ρ,1),d.Δ)
  lagS = rand(dPos,d.Δ)
  eps = d.ε / prod(lagU)^ra

  # Line 3 (first iteration)
  # Line 4 (Algorithm 3 with unconditional probabilities)
  S = rand(dPos,mAst)
  append!(S,chiStable1(dPos,d.δ,dega,dega2,mom,mAst-1))

  R = Float64[]
  C = [0.]
  F = Float64[]
  # Last value of C
  endC = 0.
  # First iteration
  # Line 5
  while (length(R) == σ) || (length(S) >= length(C))
    t = length(C)
    # Lines 6 and 7
    (C1,F1) = downRW(d.d,rar,d.η,et1,d.κ,x-endC)
    append!(F,F1)
    append!(C,endC .+ C1)
    endC = C[end]
    # Lines 8 and 9
    if reflectedProcess(d.d,d.η,et1,d.κ,x-endC)
      # Line 10
      x = endC + d.κ
      R1 = [maximum(C[t:end])]
      for i = (t-1):-1:(length(R)+1)
        push!(R1,max(R1[end],C[i]))
      end
      append!(R,R1[end:-1:1] .- C[(length(R)+1):t])
    else # Line 11
      # Lines 12 and 13
      (C1,F1) = upRW(d.d,d.η,et1,d.κ,x-endC)
      append!(F,F1)
      append!(C,endC .+ C1)
      endC = C[end]
    end # Line 14
  end # Line 15
  # Line 16
  while length(U) < length(S)
    push!(U,exp(-d.α*F[length(U)+1]))
  end
  # Lines 17, 18 and 19
  σ = 1
  D = exp(R[σ])*( e2*exp(-sep*(length(S)-σ-1)) + sum(S[σ:end] .*
    exp.(-d.d .* (0:(length(S)-σ))) .* (1 .- U[σ:end]) .^ ra) )
  eps /= U[σ]^ra
  # Line 20
  if D <= eps # Finished!
    # Line 21
    D = (prod(lagU)*U[1])^ra * D
    X = (1-U[1])^ra * S[1]
    for i = d.Δ:-1:1
      X = lagU[i]^ra * X + (1-lagU[i])^ra * lagS[i]
    end
    return (X,X+D,σ)
  end # Line 22
  # Line 3 (second to last iterations)
  while true
    # Line 4 (Algorithm 3 with conditional probabilities)
    append!(S,chiStable2(dPos,d.δ,dega,dega2,mom,length(S)-σ-1))
    # Line 5
    while (length(R) == σ) || (length(S) >= length(C))
      t = length(C)-1
      # Lines 6 and 7
      (C1,F1) = downRW(d.d,rar,d.η,et1,d.κ,x-endC)
      append!(F,F1)
      append!(C,endC .+ C1)
      endC = C[end]
      # Lines 8 and 9
      if reflectedProcess(d.d,d.η,et1,d.κ,x-endC)
        # Line 10
        x = endC + d.κ
        R1 = [maximum(C[t:end])]
        for i = (t-1):-1:(length(R)+1)
          push!(R1,max(R1[end],C[i]))
        end
        append!(R,R1[end:-1:1] .- C[(length(R)+1):t])
      else # Line 11
        # Lines 12 and 13
        (C1,F1) = upRW(d.d,d.η,et1,d.κ,x-endC)
        append!(F,F1)
        append!(C,endC .+ C1)
        endC = C[end]
      end # Line 14
    end # Line 15
    # Line 16
    while length(U) < length(S)
      push!(U,exp(-d.α*F[length(U)+1]))
    end
    # Lines 17, 18 and 19
    σ += 1
    D = exp(R[σ])*( e2*exp(-sep*(length(S)-σ-1)) + sum(S[σ:end] .*
      exp.(-d.d .* (0:(length(S)-σ))) .* (1 .- U[σ:end]) .^ ra) )
    eps /= U[σ]^ra
    # Line 24
    if D <= eps # Finished!
      D = (prod(U[1:σ])*prod(lagU))^ra * D
      X = 0.
      for i = σ:-1:1
        X = U[i]^ra * X + (1-U[i])^ra * S[i]
      end
      for i = d.Δ:-1:1
        X = lagU[i]^ra * X + (1-lagU[i])^ra * lagS[i]
      end
      # Line 25
      return (X,X+D,σ)
    end # Line 26
  end # Line 27
end

# @Input: StableMeander
# @Output: A random sample (lower estimate) from the law d
function rand(d::StableMeander)
  return rand(precise_sampler(d))[1]
end

function rand(d::StableMeander,N::Int)
  return [rand(d) for i =1:N]
end

function length(d::PreciseStableMeander)
  return 3
end

# Distribution created to sample ε-strongly and localise the function according
# to the partition induced by a discrete function f.
struct LocStableMeander <: Distribution{Multivariate,Continuous}
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  d::Float64
  η::Float64
  δ::Float64
  γ::Float64
  κ::Float64
  Δ::Int64
  mAst::Int64
  f::Function
  LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int,f::Function) =
    (β < -1 || β > 1 || (β == -1 && α <= 1) || α <= 0 || α > 2) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") : (0 >= γ || γ >= α) ?
    error("Parameters' requirements unmet: α>γ>0") : (0 >= δ || δ >= d || d >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(αρ)>d>δ>0") : κ < 0 ?
    error("Parameters' requirements unmet: κ≥0") : Δ <= 0 ?
    error("Parameters' requirements unmet: Δ≥0") : mAst < 0 ?
    error("Parameters' requirements unmet: m*≥0") : new(α, β, β*(α <= 1 ? 1 : (α-2)/α),(1+β*(α <= 1 ? 1 : (α-2)/α))/2 , d,
      etaF(d*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2 , δ, γ, κ + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),
      log(2)*2/(3*etaF(d*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))), Δ, mAst, f)
  LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,f::Function) = LocStableMeander(α,β,d,δ,γ,κ,Δ,12,f)
  LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,f::Function) = LocStableMeander(α,β,d,δ,γ,κ,
    Int(ceil(33*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),f)
  LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,f::Function) = LocStableMeander(α,β,d,δ,γ,4,f)
  LocStableMeander(α::Real,β::Real,Δ::Integer,f::Function) = LocStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,4,Δ,f)
  LocStableMeander(α::Real,β::Real,f::Function) = LocStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,f)
end

function local_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int,f::Function)
  return LocStableMeander(d.α,d.β,dr,δ,γ,κ,Δ,mAst,f)
end

function local_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real,Δ::Int,f::Function)
  return LocStableMeander(d.α,d.β,dr,δ,γ,κ,Δ,f)
end

function local_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,κ::Real,f::Function)
  return LocStableMeander(d.α,d.β,dr,δ,γ,κ,f)
end

function local_sampler(d::StableMeander,dr::Real,δ::Real,γ::Real,f::Function)
  return LocStableMeander(d.α,d.β,dr,δ,γ,f)
end

function local_sampler(d::StableMeander,Δ::Integer,f::Function)
  return LocStableMeander(d.α,d.β,Δ,f)
end

function local_sampler(d::StableMeander,f::Function)
  return LocStableMeander(d.α,d.β,f)
end

# Algorithm 9
# @Input: LocStableMeander
# @Output: A random sample from (f(StableMeander),UE,LE),
# where UE and LE are the upper and lower estimates of StableMeander.
function rand(d::LocStableMeander)
  ar = d.α*d.ρ
  et1 = 1 /(ar+d.η)
  ra = 1 /d.α
  rar = 1 /ar
  # Conditionally positive stable distribution
  dPos = PositiveStable(d.α,d.β)
  sep = d.d - d.δ
  e2 = 1 /(1 -exp(-sep))
  mom = mellin(dPos,d.γ)
  dega = d.δ*d.γ
  dega2 = -1 /(1 -exp(-dega))
  mAst = d.mAst + Int(ceil( max(0,log(mom)/dega) + rar ))
  # θ sequence
  U = Float64[]
  # Line 1
  x = Inf
  σ = 0
  # Line 2
  lagU = rand(Beta(d.ρ,1),d.Δ)
  lagS = rand(dPos,d.Δ)
  eps = 1 / prod(lagU)^ra

  # Line 3 (first iteration)
  # Line 4 (Algorithm 3 with unconditional probabilities)
  S = rand(dPos,mAst)
  append!(S,chiStable1(dPos,d.δ,dega,dega2,mom,mAst-1))

  R = Float64[]
  C = [0.]
  F = Float64[]
  # Last value of C
  endC = 0.
  # First iteration
  # Line 5
  while (length(R) == σ) || (length(S) >= length(C))
    t = length(C)
    # Lines 6 and 7
    (C1,F1) = downRW(d.d,rar,d.η,et1,d.κ,x-endC)
    append!(F,F1)
    append!(C,endC .+ C1)
    endC = C[end]
    # Lines 8 and 9
    if reflectedProcess(d.d,d.η,et1,d.κ,x-endC)
      # Line 10
      x = endC + d.κ
      R1 = [maximum(C[t:end])]
      for i = (t-1):-1:(length(R)+1)
        push!(R1,max(R1[end],C[i]))
      end
      append!(R,R1[end:-1:1] .- C[(length(R)+1):t])
    else # Line 11
      # Lines 12 and 13
      (C1,F1) = upRW(d.d,d.η,et1,d.κ,x-endC)
      append!(F,F1)
      append!(C,endC .+ C1)
      endC = C[end]
    end # Line 14
  end # Line 15
  # Line 16
  while length(U) < length(S)
    push!(U,exp(-d.α*F[length(U)+1]))
  end
  # Lines 17, 18 and 19
  σ = 1
  D = exp(R[σ])*( e2*exp(-sep*(length(S)-σ-1)) + sum(S[σ:end] .*
    exp.(-d.d .* (0:(length(S)-σ))) .* (1 .- U[σ:end]) .^ ra) )
  eps /= U[σ]^ra
  # Line 20
  # Line 21
  X = (1-U[1])^ra * S[1]
  for i = d.Δ:-1:1
    X = lagU[i]^ra * X + (1-lagU[i])^ra * lagS[i]
  end
  aux = d.f(X)
  if aux == d.f(X+D/eps)
    return (aux,X,X+D/eps,σ)
  end # Line 22
  # Line 3 (second to last iterations)
  while true
    # Line 4 (Algorithm 3 with conditional probabilities)
    append!(S,chiStable2(dPos,d.δ,dega,dega2,mom,length(S)-σ-1))
    # Line 5
    while (length(R) == σ) || (length(S) >= length(C))
      t = length(C)-1
      # Lines 6 and 7
      (C1,F1) = downRW(d.d,rar,d.η,et1,d.κ,x-endC)
      append!(F,F1)
      append!(C,endC .+ C1)
      endC = C[end]
      # Lines 8 and 9
      if reflectedProcess(d.d,d.η,et1,d.κ,x-endC)
        # Line 10
        x = endC + d.κ
        R1 = [maximum(C[t:end])]
        for i = (t-1):-1:(length(R)+1)
          push!(R1,max(R1[end],C[i]))
        end
        append!(R,R1[end:-1:1] .- C[(length(R)+1):t])
      else # Line 11
        # Lines 12 and 13
        (C1,F1) = upRW(d.d,d.η,et1,d.κ,x-endC)
        append!(F,F1)
        append!(C,endC .+ C1)
        endC = C[end]
      end # Line 14
    end # Line 15
    # Line 16
    while length(U) < length(S)
      push!(U,exp(-d.α*F[length(U)+1]))
    end
    # Lines 17, 18 and 19
    σ += 1
    D = exp(R[σ])*( e2*exp(-sep*(length(S)-σ-1)) + sum(S[σ:end] .*
      exp.(-d.d .* (0:(length(S)-σ))) .* (1 .- U[σ:end]) .^ ra) )
    X += (1-U[σ])^ra * S[σ] / eps
    eps /= U[σ]^ra
    # Line 24
    aux = d.f(X)
    if aux == d.f(X+D/eps)
      return (aux,X,X+D/eps,σ)
    end # Line 26
  end # Line 27
end

function length(d::LocStableMeander)
  return 4
end

# Finite-dimensional stable meander

struct MvStableMeander <: ContinuousMultivariateDistribution
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  t::Array{Float64,1}
  MvStableMeander(α,β,t) = ( β<-1 || β>1 || (β==-1 && α<=1) || α<=0 || α>2 ) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") :
    !( 0 < t[1] && prod([ t[i-1] < t[i] for i = 2:length(t)]) ) ? error("Parameters' requirements unmet: 0 < t[1] < ... < t[end]") :
    new(α,β,β*(α<=1 ? 1 : (α-2)/α),(1+β*(α<=1 ? 1 : (α-2)/α))/2,t)
end

import Distributions.params
function params(d::MvStableMeander)
  return (d.α,d.β,d.θ,d.ρ,d.t)
end

import Distributions.insupport
function insupport(d::MvStableMeander,x::Array{Float64,1})
  return prod(x .>= 0)
end

import Base.length
function length(d::MvStableMeander)
  return length(d.t)
end

import Distributions.rand
function rand(d::MvStableMeander)
  if d.α <= 1 && d.β == 1
    X = rand(Stable(d.α,1),length(d.t))
    X[1] *= d.t[1]^(1/d.α)
    for i=2:length(d.t)
      X[i] *= (d.t[i]-d.t[i-1])^(1/d.α)
    end
    return cumsum(X)
  else
    return rand(precise_sampler(d))[1]
  end
end

# Does not really support α≤1 and β=1 (produces an exact MvStableMeander)
struct PreciseMvStableMeander <: Sampleable{Multivariate,Continuous}
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  t::Array{Float64,1}
  # For the rest of the parameters, we have left/right versions
  d_l::Float64
  d_r::Float64
  η_l::Float64
  η_r::Float64
  δ_l::Float64
  δ_r::Float64
  γ_l::Float64
  γ_r::Float64
  κ_l::Float64
  κ_r::Float64
  Δ_l::Int64
  Δ_r::Int64
  mAst_l::Int64
  mAst_r::Int64
  ε::Float64
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,ε::Real) =
    ( β<-1 || β>1 || (β==-1 && α<=1) || α<=0 || α>2 ) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") :
    (β == 1 && α <= 1) ? MvStableMeander(α,1,t) :
    (0 >= γ_l || γ_l >= α || 0 >= γ_r || γ_r >= α) ?
    error("Parameters' requirements unmet: α>γ_l>0 and α>γ_r>0") : (0 >= δ_l || δ_l >= d_l || 0 >= δ_r || δ_r >= d_r || d_l >= 2/(α - β*(α <= 1 ? α : α-2)) || d_r >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(α(1-ρ))>d_l>δ_l>0 and 1/(αρ)>dr>δr>0") : (κ_l < 0 || κ_r < 0) ?
    error("Parameters' requirements unmet: κ_l,κ_r≥0") : (Δ_l <= 0 || Δ_r <= 0) ?
    error("Parameters' requirements unmet: Δ_l,Δr≥0") : (mAst_l < 0 || mAst_r < 0) ?
    error("Parameters' requirements unmet: m_l*,m_r*≥0") : ε <= 0 ?
    error("Parameters' requirements unmet: ε>0") : !( 0 < t[1] && prod([ t[i-1] < t[i] for i = 2:length(t)]) ) ?
    error("Parameters' requirements unmet: 0 < t[1] < ... < t[end]") :
    new(α, β, β*(α <= 1 ? 1 : (α-2)/α), (1+β*(α <= 1 ? 1 : (α-2)/α))/2, t, d_l, d_r,
      etaF(d_l*(α - β*(α <= 1 ? α : α-2))/2)*(α - β*(α <= 1 ? α : α-2))/2, etaF(d_r*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2,
      δ_l, δ_r, γ_l, γ_r,
      κ_l + Int(ceil(max(2/(α-β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_l*(α-β*(α <= 1 ? α : α-2))/2)*(α-β*(α <= 1 ? α : α-2)) ) ))),
      κ_r + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_r*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))),
      Δ_l, Δ_r, mAst_l, mAst_r, ε)
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int) = PreciseMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,mAst_l,mAst_r,.5^16)
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int) = PreciseMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,12,12)
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real) = PreciseMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))))
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real) = PreciseMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,4,4)
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1}) = PreciseMvStableMeander(α,β,t,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α)
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},Δ_l::Integer,Δ_r::Integer,ε::Real) = PreciseMvStableMeander(α,β,t,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α,4,4,Δ_l,Δ_r,12,12,ε)
  PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},ε::Real) = PreciseMvStableMeander(α,β,t,
    Int(ceil(abs(log(ε/2))/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(abs(log(ε/2))/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),ε)
end

function precise_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,ε::Real)
  return PreciseMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,mAst_l,mAst_r,ε)
end

function precise_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int)
  return PreciseMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,mAst_l,mAst_r)
end

function precise_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int)
  return PreciseMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r)
end

function precise_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real)
  return PreciseMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r)
end

function precise_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real)
  return PreciseMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r)
end

function precise_sampler(d::MvStableMeander,Δ_l::Integer,Δ_r::Integer,ε::Real)
  return PreciseMvStableMeander(d.α,d.β,d.t,Δ_l,Δ_r,ε)
end

function precise_sampler(d::MvStableMeander,ε::Real)
  return PreciseMvStableMeander(d.α,d.β,d.t,ε)
end

function precise_sampler(d::MvStableMeander)
  return PreciseMvStableMeander(d.α,d.β,d.t)
end

import Base.length
function length(d::PreciseMvStableMeander)
  return Int(2*length(d.t) + 1)
end

function rand(d::PreciseMvStableMeander)
  ra = 1 /d.α

  if length(d.t) == 1
    return rand(PreciseStableMeander(d.α,d.β,d.d_r,δ_r,γ_r,κ_r,Δ_r,mAst_r,ε)) * t[1]^ra
  end

  m_r = length(d.t)
  m_l = Int(m_r-1)

  while true
    τ = rand(Beta(1-d.ρ,d.ρ),m_l)
    T_l = [d.t[i]-d.t[i-1] for i = 2:length(d.t)] .* τ
    T_r = [dt[1]]
    append!(T_r, [d.t[i]-d.t[i-1] for i = 2:length(d.t)] .* (1 .- τ))

    T_l = T_l .^ ra
    T_r = T_r .^ ra

    ar_l, ar_r = d.α*(1-d.ρ), d.α*d.ρ
    et1_l, et1_r = 1/(ar_l+d.η_l), 1/(ar_r+d.η_r)
    rar_l, rar_r = 1/ar_l, 1/ar_r

    # Conditionally positive stable distribution
    dPos_l, dPos_r = PositiveStable(d.α,-d.β), PositiveStable(d.α,d.β)
    sep_l, sep_r = d.d_l - d.δ_l, d.d_r - d.δ_r
    e2_l, e2_r = 1 /(1 -exp(-sep_l)), 1 /(1 -exp(-sep_r))
    mom_l, mom_r = mellin(dPos_l,d.γ_l), mellin(dPos_r,d.γ_r)
    dega_l, dega_r = d.δ_l*d.γ_l, d.δ_r*d.γ_r
    dega2_l, dega2_r = -1 /(1 -exp(-dega_l)), -1 /(1 -exp(-dega_r))
    mAst_l, mAst_r = d.mAst_l + Int(ceil( max(0,log(mom_l)/dega_l) + rar_l )), d.mAst_r + Int(ceil( max(0,log(mom_r)/dega_r) + rar_r ))
    # θ sequence
    U_l, U_r = [Float64[] for i = 1:m_l], [Float64[] for i = 1:m_r]
    # Line 1
    x_l, x_r = [Inf for i = 1:m_l], [Inf for i = 1:m_r]
    σ = 0
    # Line 2
    lagU_l, lagU_r = [rand(Beta(1-d.ρ,1),d.Δ_l) for i = 1:m_l], [rand(Beta(d.ρ,1),d.Δ_r) for i = 1:m_r]
    lagS_l, lagS_r = [rand(dPos_l,d.Δ_l) for i = 1:m_l], [rand(dPos_r,d.Δ_r) for i = 1:m_r]
    prod_l, prod_r = [prod(lagU_l[i]) for i = 1:m_l], [prod(lagU_r[i]) for i = 1:m_r]
    # Line 3 (first iteration)
    # Line 4 (Algorithm 3 with unconditional probabilities)
    S_l, S_r = [rand(dPos_l,mAst_l) for i = 1:m_l], [rand(dPos_r,mAst_r) for i = 1:m_r]
    for i = 1:m_l
      append!(S_l[i],chiStable1(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,mAst_l-1))
    end
    for i = 1:m_r
      append!(S_r[i],chiStable1(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,mAst_r-1))
    end

    R_l, R_r = [Float64[] for i = 1:m_l], [Float64[] for i = 1:m_r]
    C_l, C_r = [[0.] for i = 1:m_l], [[0.] for i = 1:m_r]
    F_l, F_r = [Float64[] for i = 1:m_l], [Float64[] for i = 1:m_r]
    # Last value of C
    endC_l, endC_r = [0. for i = 1:m_l], [0. for i = 1:m_r]
    # First iteration
    # Line 5
    for i = 1:m_l
      while (length(R_l[i]) == σ) || (length(S_l[i]) >= length(C_l[i]))
        t_l = length(C_l[i])
        # Lines 6 and 7
        (C1_l[i],F1_l[i]) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
        append!(F_l[i],F1_l[i])
        append!(C_l[i],endC_l[i] .+ C1_l[i])
        endC_l[i] = C_l[i][end]
        # Lines 8 and 9
        if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
          # Line 10
          x_l[i] = endC_l[i] + d.κ_l
          R1_l = [maximum(C_l[i][t_l:end])]
          for j = (t_l-1):-1:(length(R_l[i])+1)
            push!(R1_l,max(R1_l[end],C_l[i][j]))
          end
          append!(R_l,R1_l[end:-1:1] .- C_l[i][(length(R_l[i])+1):t_l])
        else # Line 11
          # Lines 12 and 13
          (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
          append!(F_l[i],F1_l)
          append!(C_l[i],endC_l .+ C1_l)
          endC_l[i] = C_l[i][end]
        end # Line 14
      end # Line 15 - left
    end
    for i = 1:m_r
      while (length(R_r[i]) == σ) || (length(S_r[i]) >= length(C_r[i]))
        t_r = length(C_r[i])
        # Lines 6 and 7
        (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
        append!(F_r[i],F1_r)
        append!(C_r[i],endC_r[i] .+ C1_r)
        endC_r[i] = C_r[i][end]
        # Lines 8 and 9
        if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
          # Line 10
          x_r[i] = endC_r[i] + d.κ_r
          R1_r = [maximum(C_r[i][t_r:end])]
          for j = (t_r-1):-1:(length(R_r[i])+1)
            push!(R1_r,max(R1_r[end],C_r[i][j]))
          end
          append!(R_r[i],R1_r[end:-1:1] .- C_r[i][(length(R_r[i])+1):t_r])
        else # Line 11
          # Lines 12 and 13
          (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
          append!(F_r[i],F1_r)
          append!(C_r[i],endC_r[i] .+ C1_r)
          endC_r[i] = C_r[i][end]
        end # Line 14
      end # Line 15 - right
    end
    # Line 16
    for i = 1:m_l
      while length(U_l[i]) < length(S_l[i])
        push!(U_l[i],exp(-d.α*F_l[i][length(U_l[i])+1]))
      end
    end
    for i = 1:m_r
      while length(U_r[i]) < length(S_r[i])
        push!(U_r[i],exp(-d.α*F_r[i][length(U_r[i])+1]))
      end
    end
    # Lines 17, 18 and 19
    σ = 1
    D_l = [exp(R_l[i][σ])*( e2_l*exp(-sep_l*(length(S_l[i])-σ-1)) + sum(S_l[i][σ:end] .*
        exp.(-d.d_l .* (0:(length(S_l[i])-σ))) .* (1 .- U_l[i][σ:end]) .^ ra) ) for i = 1:m_l]
    D_r = [exp(R_r[i][σ])*( e2_r*exp(-sep_r*(length(S_r[i])-σ-1)) + sum(S_r[i][σ:end] .*
        exp.(-d.d_r .* (0:(length(S_r[i])-σ))) .* (1 .- U_r[i][σ:end]) .^ ra) ) for i = 1:m_r]

    for i = 1:m_l
      push!(lagS_l[i],S_l[i][σ])
      push!(lagU_l[i],U_l[i][σ])
    end
    for i = 1:m_r
      push!(lagS_r[i],S_r[i][σ])
      push!(lagU_r[i],U_r[i][σ])
    end
    # Line 20
    # Line 21
    L_l = [(prod_l[i]*U_l[i][1])^ra for i = 1:m_l]
    L_r = [(prod_r[i]*U_r[i][1])^ra for i = 1:m_r]

    X_l = [(1-U_l[i][1])^ra * S_l[i][1] for i = 1:m_l]
    X_r = [(1-U_r[i][1])^ra * S_r[i][1] for i = 1:m_r]

    for i = 1:m_l
      for j = d.Δ_l:-1:1
        X_l[i] = lagU_l[i][j]^ra * X_l[i] + (1-lagU_l[i][j])^ra * lagS_l[i][j]
      end
    end
    for i = 1:m_r
      for j = d.Δ_r:-1:1
        X_r[i] = lagU_r[i][j]^ra * X_r[i] + (1-lagU_r[i][j])^ra * lagS_r[i][j]
      end
    end

    CX_lu = [0.]
    append!(CX_lu, cumsum(X_l .* T_l))
    CX_ld = [0.]
    append!(CX_ld, cumsum((X_l .+ D_l .* L_l) .* T_l))
    CX_ru = cumsum((X_r .+ D_r .* L_r) .* T_r)
    CX_rd = cumsum(X_r .* T_r)
    X_u = CX_ru .- CX_lu
    X_d = CX_rd .- CX_ld

    if !(prod(X_u .> 0))
      continue
    elseif X_u[end] - X_d[end] < d.ε && prod(X_d .> 0)# Precision reached!
      return (X_d,X_u,σ)
    end # Line 22
    # Line 3 (second to last iterations)
    while true
      # Line 4 (Algorithm 3 with conditional probabilities)
      for i = 1:m_l
        append!(S_l[i],chiStable2(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,length(S_l[i])-σ-1))
      end
      for i = 1:m_r
        append!(S_r[i],chiStable2(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,length(S_r[i])-σ-1))
      end
      # Line 5
      for i = 1:m_l
        while (length(R_l[i]) == σ) || (length(S_l[i]) >= length(C_l[i]))
          t_l = length(C_l[i])-1
          # Lines 6 and 7
          (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
          append!(F_l[i],F1_l)
          append!(C_l[i],endC_l[i] .+ C1_l)
          endC_l[i] = C_l[i][end]
          # Lines 8 and 9
          if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
            # Line 10
            x_l[i] = endC_l[i] + d.κ_l
            R1_l = [maximum(C_l[i][t_l:end])]
            for j = (t_l-1):-1:(length(R_l[i])+1)
              push!(R1_l,max(R1_l[end],C_l[i][j]))
            end
            append!(R_l[i],R1_l[end:-1:1] .- C_l[i][(length(R_l[i])+1):t_l])
          else # Line 11
            # Lines 12 and 13
            (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
            append!(F_l[i],F1_l)
            append!(C_l[i],endC_l .+ C1_l)
            endC_l[i] = C_l[i][end]
          end # Line 14
        end # Line 15
      end
      for i = 1:m_r
        while (length(R_r[i]) == σ) || (length(S_r[i]) >= length(C_r[i]))
          t_r = length(C_r[i])-1
          # Lines 6 and 7
          (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
          append!(F_r[i],F1_r)
          append!(C_r[i],endC_r[i] .+ C1_r)
          endC_r[i] = C_r[i][end]
          # Lines 8 and 9
          if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
            # Line 10
            x_r[i] = endC_r[i] + d.κ_r
            R1_r = [maximum(C_r[t_r:end])]
            for j = (t_r-1):-1:(length(R_r[i])+1)
              push!(R1_r,max(R1_r[end],C_r[i][j]))
            end
            append!(R_r[i],R1_r[end:-1:1] .- C_r[i][(length(R_r[i])+1):t_r])
          else # Line 11
            # Lines 12 and 13
            (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
            append!(F_r[i],F1_r)
            append!(C_r[i],endC_r[i] .+ C1_r)
            endC_r[i] = C_r[i][end]
          end # Line 14
        end # Line 15
      end
      # Line 16
      for i = 1:m_l
        while length(U_l[i]) < length(S_l[i])
          push!(U_l[i],exp(-d.α*F_l[i][length(U_l[i])+1]))
        end
      end
      for i = 1:m_r
        while length(U_r[i]) < length(S_r[i])
          push!(U_r[i],exp(-d.α*F_r[i][length(U_r[i])+1]))
        end
      end
      # Lines 17, 18 and 19
      σ += 1
      D_l = [exp(R_l[i][σ])*( e2_l*exp(-sep_l*(length(S_l[i])-σ-1)) + sum(S_l[i][σ:end] .*
          exp.(-d.d_l .* (0:(length(S_l[i])-σ))) .* (1 .- U_l[i][σ:end]) .^ ra) ) for i = 1:m_l]
      D_r = [exp(R_r[σ])*( e2_r*exp(-sep_r*(length(S_r[i])-σ-1)) + sum(S_r[i][σ:end] .*
          exp.(-d.d_r .* (0:(length(S_r[i])-σ))) .* (1 .- U_r[i][σ:end]) .^ ra) ) for i = 1:m_r]
      for i = 1:m_l
        push!(lagS_l[i],S_l[i][σ])
        push!(lagU_l[i],U_l[i][σ])
      end
      for i = 1:m_r
        push!(lagS_r[i],S_r[i][σ])
        push!(lagU_r[i],U_r[i][σ])
      end
      # Line 24
      X_l .+= [L_l[i]*(1-U_l[i][σ])^ra * S_l[i][σ] for i = 1:m_l]
      X_r .+= [L_r[i]*(1-U_r[i][σ])^ra * S_r[i][σ] for i = 1:m_r]

      L_l .*= [U_l[i][σ]^ra for i = 1:m_l]
      L_r .*= [U_r[i][σ]^ra for i = 1:m_r]

      CX_lu = [0.]
      append!(CX_lu, cumsum(X_l .* T_l))
      CX_ld = [0.]
      append!(CX_ld, cumsum((X_l .+ D_l .* L_l) .* T_l))
      CX_ru = cumsum((X_r .+ D_r .* L_r) .* T_r)
      CX_rd = cumsum(X_r .* T_r)
      X_u = CX_ru .- CX_lu
      X_d = CX_rd .- CX_ld

      if !(prod(X_u .> 0))
        continue
      elseif X_u[end] - X_d[end] < d.ε && prod(X_d .> 0)# Precision reached!
        return (X_d,X_u,σ)
      end # Line 26
    end # Line 27
  end
end

struct LocMvStableMeander <: Sampleable{Multivariate,Continuous}
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  t::Array{Float64,1}
  # For the rest of the parameters, we have left/right versions
  d_l::Float64
  d_r::Float64
  η_l::Float64
  η_r::Float64
  δ_l::Float64
  δ_r::Float64
  γ_l::Float64
  γ_r::Float64
  κ_l::Float64
  κ_r::Float64
  Δ_l::Int64
  Δ_r::Int64
  mAst_l::Int64
  mAst_r::Int64
  f::Function
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,f::Function) =
    ( β<-1 || β>1 || (β==-1 && α<=1) || α<=0 || α>2 ) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") :
    #(β == 1 && α <= 1) ? MvStableMeander(α,1,t) :
    (0 >= γ_l || γ_l >= α || 0 >= γ_r || γ_r >= α) ?
    error("Parameters' requirements unmet: α>γ_l>0 and α>γ_r>0") : (0 >= δ_l || δ_l >= d_l || 0 >= δ_r || δ_r >= d_r || d_l >= 2/(α - β*(α <= 1 ? α : α-2)) || d_r >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(α(1-ρ))>d_l>δ_l>0 and 1/(αρ)>dr>δr>0") : (κ_l < 0 || κ_r < 0) ?
    error("Parameters' requirements unmet: κ_l,κ_r≥0") : (Δ_l <= 0 || Δ_r <= 0) ?
    error("Parameters' requirements unmet: Δ_l,Δr≥0") : (mAst_l < 0 || mAst_r < 0) ?
    error("Parameters' requirements unmet: m_l*,m_r*≥0") : !( 0 < t[1] && prod([ t[i-1] < t[i] for i = 2:length(t)]) ) ?
    error("Parameters' requirements unmet: 0 < t[1] < ... < t[end]") :
    new(α, β, β*(α <= 1 ? 1 : (α-2)/α), (1+β*(α <= 1 ? 1 : (α-2)/α))/2, t, d_l, d_r,
      etaF(d_l*(α - β*(α <= 1 ? α : α-2))/2)*(α - β*(α <= 1 ? α : α-2))/2, etaF(d_r*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2,
      δ_l, δ_r, γ_l, γ_r,
      κ_l + Int(ceil(max(2/(α-β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_l*(α-β*(α <= 1 ? α : α-2))/2)*(α-β*(α <= 1 ? α : α-2)) ) ))),
      κ_r + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_r*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))),
      Δ_l, Δ_r, mAst_l, mAst_r, f)
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,f::Function) = LocMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,mAst_l,mAst_r,.5^16,f)
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,f::Function) = LocMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,12,12,f)
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,f::Function) = LocMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))))
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,f::Function) = LocMvStableMeander(α,β,t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,4,4,f)
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},f::Function) = LocMvStableMeander(α,β,t,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α,f)
  LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},Δ_l::Integer,Δ_r::Integer,f::Function) = LocMvStableMeander(α,β,t,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α,4,4,Δ_l,Δ_r,12,12,f)
end

function local_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,f::Function)
  return LocMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,mAst_l,mAst_r,f)
end

function local_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,f::Function)
  return LocMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,f)
end

function local_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,f::Function)
  return LocMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,f)
end

function local_sampler(d::MvStableMeander,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,f::Function)
  return LocMvStableMeander(d.α,d.β,d.t,d_l,d_r,δ_l,δ_r,γ_l,γ_r,f)
end

function local_sampler(d::MvStableMeander,Δ_l::Integer,Δ_r::Integer,f::Function)
  return LocMvStableMeander(d.α,d.β,d.t,Δ_l,Δ_r,f)
end

function local_sampler(d::MvStableMeander,f::Function)
  return LocMvStableMeander(d.α,d.β,d.t,f)
end

import Base.length
function length(d::LocMvStableMeander)
  return Int(2*length(d.t) + 2)
end

function rand(d::LocMvStableMeander)
  ra = 1 /d.α

  if d.β == 1 && d.α <= 1
    return f(rand(MvStableMeander(d.α,1,d.t)))
  end

  if length(d.t) == 1
    g(x) = d.f(x* t[1]^ra)
    return rand(LocStableMeander(d.α,d.β,d.d_r,δ_r,γ_r,κ_r,Δ_r,mAst_r,g))
  end

  m_r = length(d.t)
  m_l = Int(m_r-1)

  while true
    τ = rand(Beta(1-d.ρ,d.ρ),m_l)
    T_l = [d.t[i]-d.t[i-1] for i = 2:length(d.t)] .* τ
    T_r = [dt[1]]
    append!(T_r, [d.t[i]-d.t[i-1] for i = 2:length(d.t)] .* (1 .- τ))

    T_l = T_l .^ ra
    T_r = T_r .^ ra

    ar_l, ar_r = d.α*(1-d.ρ), d.α*d.ρ
    et1_l, et1_r = 1/(ar_l+d.η_l), 1/(ar_r+d.η_r)
    rar_l, rar_r = 1/ar_l, 1/ar_r

    # Conditionally positive stable distribution
    dPos_l, dPos_r = PositiveStable(d.α,-d.β), PositiveStable(d.α,d.β)
    sep_l, sep_r = d.d_l - d.δ_l, d.d_r - d.δ_r
    e2_l, e2_r = 1 /(1 -exp(-sep_l)), 1 /(1 -exp(-sep_r))
    mom_l, mom_r = mellin(dPos_l,d.γ_l), mellin(dPos_r,d.γ_r)
    dega_l, dega_r = d.δ_l*d.γ_l, d.δ_r*d.γ_r
    dega2_l, dega2_r = -1 /(1 -exp(-dega_l)), -1 /(1 -exp(-dega_r))
    mAst_l, mAst_r = d.mAst_l + Int(ceil( max(0,log(mom_l)/dega_l) + rar_l )), d.mAst_r + Int(ceil( max(0,log(mom_r)/dega_r) + rar_r ))
    # θ sequence
    U_l, U_r = [Float64[] for i = 1:m_l], [Float64[] for i = 1:m_r]
    # Line 1
    x_l, x_r = [Inf for i = 1:m_l], [Inf for i = 1:m_r]
    σ = 0
    # Line 2
    lagU_l, lagU_r = [rand(Beta(1-d.ρ,1),d.Δ_l) for i = 1:m_l], [rand(Beta(d.ρ,1),d.Δ_r) for i = 1:m_r]
    lagS_l, lagS_r = [rand(dPos_l,d.Δ_l) for i = 1:m_l], [rand(dPos_r,d.Δ_r) for i = 1:m_r]
    prod_l, prod_r = [prod(lagU_l[i]) for i = 1:m_l], [prod(lagU_r[i]) for i = 1:m_r]
    # Line 3 (first iteration)
    # Line 4 (Algorithm 3 with unconditional probabilities)
    S_l, S_r = [rand(dPos_l,mAst_l) for i = 1:m_l], [rand(dPos_r,mAst_r) for i = 1:m_r]
    for i = 1:m_l
      append!(S_l[i],chiStable1(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,mAst_l-1))
    end
    for i = 1:m_r
      append!(S_r[i],chiStable1(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,mAst_r-1))
    end

    R_l, R_r = [Float64[] for i = 1:m_l], [Float64[] for i = 1:m_r]
    C_l, C_r = [[0.] for i = 1:m_l], [[0.] for i = 1:m_r]
    F_l, F_r = [Float64[] for i = 1:m_l], [Float64[] for i = 1:m_r]
    # Last value of C
    endC_l, endC_r = [0. for i = 1:m_l], [0. for i = 1:m_r]
    # First iteration
    # Line 5
    for i = 1:m_l
      while (length(R_l[i]) == σ) || (length(S_l[i]) >= length(C_l[i]))
        t_l = length(C_l[i])
        # Lines 6 and 7
        (C1_l[i],F1_l[i]) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
        append!(F_l[i],F1_l[i])
        append!(C_l[i],endC_l[i] .+ C1_l[i])
        endC_l[i] = C_l[i][end]
        # Lines 8 and 9
        if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
          # Line 10
          x_l[i] = endC_l[i] + d.κ_l
          R1_l = [maximum(C_l[i][t_l:end])]
          for j = (t_l-1):-1:(length(R_l[i])+1)
            push!(R1_l,max(R1_l[end],C_l[i][j]))
          end
          append!(R_l,R1_l[end:-1:1] .- C_l[i][(length(R_l[i])+1):t_l])
        else # Line 11
          # Lines 12 and 13
          (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
          append!(F_l[i],F1_l)
          append!(C_l[i],endC_l .+ C1_l)
          endC_l[i] = C_l[i][end]
        end # Line 14
      end # Line 15 - left
    end
    for i = 1:m_r
      while (length(R_r[i]) == σ) || (length(S_r[i]) >= length(C_r[i]))
        t_r = length(C_r[i])
        # Lines 6 and 7
        (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
        append!(F_r[i],F1_r)
        append!(C_r[i],endC_r[i] .+ C1_r)
        endC_r[i] = C_r[i][end]
        # Lines 8 and 9
        if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
          # Line 10
          x_r[i] = endC_r[i] + d.κ_r
          R1_r = [maximum(C_r[i][t_r:end])]
          for j = (t_r-1):-1:(length(R_r[i])+1)
            push!(R1_r,max(R1_r[end],C_r[i][j]))
          end
          append!(R_r[i],R1_r[end:-1:1] .- C_r[i][(length(R_r[i])+1):t_r])
        else # Line 11
          # Lines 12 and 13
          (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
          append!(F_r[i],F1_r)
          append!(C_r[i],endC_r[i] .+ C1_r)
          endC_r[i] = C_r[i][end]
        end # Line 14
      end # Line 15 - right
    end
    # Line 16
    for i = 1:m_l
      while length(U_l[i]) < length(S_l[i])
        push!(U_l[i],exp(-d.α*F_l[i][length(U_l[i])+1]))
      end
    end
    for i = 1:m_r
      while length(U_r[i]) < length(S_r[i])
        push!(U_r[i],exp(-d.α*F_r[i][length(U_r[i])+1]))
      end
    end
    # Lines 17, 18 and 19
    σ = 1
    D_l = [exp(R_l[i][σ])*( e2_l*exp(-sep_l*(length(S_l[i])-σ-1)) + sum(S_l[i][σ:end] .*
        exp.(-d.d_l .* (0:(length(S_l[i])-σ))) .* (1 .- U_l[i][σ:end]) .^ ra) ) for i = 1:m_l]
    D_r = [exp(R_r[i][σ])*( e2_r*exp(-sep_r*(length(S_r[i])-σ-1)) + sum(S_r[i][σ:end] .*
        exp.(-d.d_r .* (0:(length(S_r[i])-σ))) .* (1 .- U_r[i][σ:end]) .^ ra) ) for i = 1:m_r]

    for i = 1:m_l
      push!(lagS_l[i],S_l[i][σ])
      push!(lagU_l[i],U_l[i][σ])
    end
    for i = 1:m_r
      push!(lagS_r[i],S_r[i][σ])
      push!(lagU_r[i],U_r[i][σ])
    end
    # Line 20
    # Line 21
    L_l = [(prod_l[i]*U_l[i][1])^ra for i = 1:m_l]
    L_r = [(prod_r[i]*U_r[i][1])^ra for i = 1:m_r]

    X_l = [(1-U_l[i][1])^ra * S_l[i][1] for i = 1:m_l]
    X_r = [(1-U_r[i][1])^ra * S_r[i][1] for i = 1:m_r]

    for i = 1:m_l
      for j = d.Δ_l:-1:1
        X_l[i] = lagU_l[i][j]^ra * X_l[i] + (1-lagU_l[i][j])^ra * lagS_l[i][j]
      end
    end
    for i = 1:m_r
      for j = d.Δ_r:-1:1
        X_r[i] = lagU_r[i][j]^ra * X_r[i] + (1-lagU_r[i][j])^ra * lagS_r[i][j]
      end
    end

    CX_lu = [0.]
    append!(CX_lu, cumsum(X_l .* T_l))
    CX_ld = [0.]
    append!(CX_ld, cumsum((X_l .+ D_l .* L_l) .* T_l))
    CX_ru = cumsum((X_r .+ D_r .* L_r) .* T_r)
    CX_rd = cumsum(X_r .* T_r)
    X_u = CX_ru .- CX_lu
    X_d = CX_rd .- CX_ld

    if !(prod(X_u .> 0))
      continue
    end
    aux = d.f(X_u)
    if aux == d.f(X_d) && prod(X_d .> 0)# Precision reached!
      return (aux,X_d,X_u,σ)
    end # Line 22
    # Line 3 (second to last iterations)
    while true
      # Line 4 (Algorithm 3 with conditional probabilities)
      for i = 1:m_l
        append!(S_l[i],chiStable2(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,length(S_l[i])-σ-1))
      end
      for i = 1:m_r
        append!(S_r[i],chiStable2(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,length(S_r[i])-σ-1))
      end
      # Line 5
      for i = 1:m_l
        while (length(R_l[i]) == σ) || (length(S_l[i]) >= length(C_l[i]))
          t_l = length(C_l[i])-1
          # Lines 6 and 7
          (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
          append!(F_l[i],F1_l)
          append!(C_l[i],endC_l[i] .+ C1_l)
          endC_l[i] = C_l[i][end]
          # Lines 8 and 9
          if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
            # Line 10
            x_l[i] = endC_l[i] + d.κ_l
            R1_l = [maximum(C_l[i][t_l:end])]
            for j = (t_l-1):-1:(length(R_l[i])+1)
              push!(R1_l,max(R1_l[end],C_l[i][j]))
            end
            append!(R_l[i],R1_l[end:-1:1] .- C_l[i][(length(R_l[i])+1):t_l])
          else # Line 11
            # Lines 12 and 13
            (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l[i]-endC_l[i])
            append!(F_l[i],F1_l)
            append!(C_l[i],endC_l .+ C1_l)
            endC_l[i] = C_l[i][end]
          end # Line 14
        end # Line 15
      end
      for i = 1:m_r
        while (length(R_r[i]) == σ) || (length(S_r[i]) >= length(C_r[i]))
          t_r = length(C_r[i])-1
          # Lines 6 and 7
          (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
          append!(F_r[i],F1_r)
          append!(C_r[i],endC_r[i] .+ C1_r)
          endC_r[i] = C_r[i][end]
          # Lines 8 and 9
          if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
            # Line 10
            x_r[i] = endC_r[i] + d.κ_r
            R1_r = [maximum(C_r[t_r:end])]
            for j = (t_r-1):-1:(length(R_r[i])+1)
              push!(R1_r,max(R1_r[end],C_r[i][j]))
            end
            append!(R_r[i],R1_r[end:-1:1] .- C_r[i][(length(R_r[i])+1):t_r])
          else # Line 11
            # Lines 12 and 13
            (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r[i]-endC_r[i])
            append!(F_r[i],F1_r)
            append!(C_r[i],endC_r[i] .+ C1_r)
            endC_r[i] = C_r[i][end]
          end # Line 14
        end # Line 15
      end
      # Line 16
      for i = 1:m_l
        while length(U_l[i]) < length(S_l[i])
          push!(U_l[i],exp(-d.α*F_l[i][length(U_l[i])+1]))
        end
      end
      for i = 1:m_r
        while length(U_r[i]) < length(S_r[i])
          push!(U_r[i],exp(-d.α*F_r[i][length(U_r[i])+1]))
        end
      end
      # Lines 17, 18 and 19
      σ += 1
      D_l = [exp(R_l[i][σ])*( e2_l*exp(-sep_l*(length(S_l[i])-σ-1)) + sum(S_l[i][σ:end] .*
          exp.(-d.d_l .* (0:(length(S_l[i])-σ))) .* (1 .- U_l[i][σ:end]) .^ ra) ) for i = 1:m_l]
      D_r = [exp(R_r[σ])*( e2_r*exp(-sep_r*(length(S_r[i])-σ-1)) + sum(S_r[i][σ:end] .*
          exp.(-d.d_r .* (0:(length(S_r[i])-σ))) .* (1 .- U_r[i][σ:end]) .^ ra) ) for i = 1:m_r]
      for i = 1:m_l
        push!(lagS_l[i],S_l[i][σ])
        push!(lagU_l[i],U_l[i][σ])
      end
      for i = 1:m_r
        push!(lagS_r[i],S_r[i][σ])
        push!(lagU_r[i],U_r[i][σ])
      end
      # Line 24
      X_l .+= [L_l[i]*(1-U_l[i][σ])^ra * S_l[i][σ] for i = 1:m_l]
      X_r .+= [L_r[i]*(1-U_r[i][σ])^ra * S_r[i][σ] for i = 1:m_r]

      L_l .*= [U_l[i][σ]^ra for i = 1:m_l]
      L_r .*= [U_r[i][σ]^ra for i = 1:m_r]

      CX_lu = [0.]
      append!(CX_lu, cumsum(X_l .* T_l))
      CX_ld = [0.]
      append!(CX_ld, cumsum((X_l .+ D_l .* L_l) .* T_l))
      CX_ru = cumsum((X_r .+ D_r .* L_r) .* T_r)
      CX_rd = cumsum(X_r .* T_r)
      X_u = CX_ru .- CX_lu
      X_d = CX_rd .- CX_ld

      if !(prod(X_u .> 0))
        continue
      end
      aux = d.f(X_u)
      if aux == d.f(X_d) && prod(X_d .> 0)# Precision reached!
        return (aux,X_d,X_u,σ)
      end # Line 26
    end # Line 27
  end
end

export StableMeander, LocStableMeander, PreciseStableMeander, MvStableMeander, LocMvStableMeander, PreciseMvStableMeander, rand, precise_sampler, local_sampler, minimum, maximum, insupport, mean, params, length
