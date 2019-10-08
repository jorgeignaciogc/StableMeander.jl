using Distributions, DataFrames, DataFramesMeta, Gadfly

# @Input: a function f
# @Output: a function F whose input is a dataframe with the lengths and
# heights of a piecewise linear function and whose output is 1 if the
# dataframe crossed f and 0 otherwise.
function cross_func(f::Function)
  function F(df::DataFrame)
    x = [0.]
    y = [0.]
    append!(x,cumsum(df[:length]))
    append!(y,cumsum(df[:height]))
    return 1 - prod(y .>= f.(x))
  end
  return F
end

function cross_func(f::Function,y0::Real)
  function F(df::DataFrame)
    x = [0.]
    y = [y0]
    append!(x,cumsum(df[:length]))
    append!(y,y0 .+ cumsum(df[:height]))
    return 1 - prod(y .>= f.(x))
  end
  return F
end

# Tupple({ℓ_n},{ξ_n}) using the notation of the paper
struct ConvexMinorantStableMeander <: Distribution{Multivariate,Continuous}
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
  ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int,ε::Float64) =
    (β < -1 || β > 1 || (β == -1 && α <= 1) || α <= 0 || α > 2) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") : (0 >= γ || γ >= α) ?
    error("Parameters' requirements unmet: α>γ>0") : (0 >= δ || δ >= d || d >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(αρ)>d>δ>0") : κ < 0 ?
    error("Parameters' requirements unmet: κ≥0") : Δ <= 0 ?
    error("Parameters' requirements unmet: Δ≥0") : mAst < 0 ?
    error("Parameters' requirements unmet: m*≥0") : ε <= 0 ?
    error("Parameters' requirements unmet: ε>0") : new(α, β, β*(α <= 1 ? 1 : (α-2)/α),(1+β*(α <= 1 ? 1 : (α-2)/α))/2 , d,
      etaF(d*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2 , δ, γ, κ + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),
      log(2)*2/(3*etaF(d*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))), Δ, mAst, ε)
  ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int) = ConvexMinorantStableMeander(α,β,d,δ,γ,κ,Δ,mAst,.5^16)
  ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int) = ConvexMinorantStableMeander(α,β,d,δ,γ,κ,Δ,12)
  ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real) = ConvexMinorantStableMeander(α,β,d,δ,γ,κ,
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))))
  ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real) = ConvexMinorantStableMeander(α,β,d,δ,γ,4)
  ConvexMinorantStableMeander(α::Real,β::Real) = ConvexMinorantStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α)
  ConvexMinorantStableMeander(α::Real,β::Real,Δ::Integer,ε::Real) = ConvexMinorantStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,4,Δ,12,ε)
  ConvexMinorantStableMeander(α::Real,β::Real,ε::Real) = ConvexMinorantStableMeander(α,β,
    Int(ceil(abs(log(ε/2))/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),ε)
end

import Distributions.params
function params(d::ConvexMinorantStableMeander)
  return (d.α,d.β,d.θ,d.ρ,
          d.d,d.η,d.δ,d.γ,d.κ,d.Δ,d.mAst,
          d.ε)
end

# Useful auxiliary function that reconstructs the faces from the sequences {S_n} and {U_n}
# @Input: Stability parameter and (almost) output of rand(ConvexMinorantStableMeander)
# @Output: DataFrame with heights, lengths, and slopes of the faces of the
# lower and upper convex minorants sorted by slopes and their distance
function faces(α::Real,D::Real,S::Array{Float64},U::Array{Float64})
  ℓ = cumprod(U)
  push!(ℓ,0.)
  for i = length(ℓ):-1:2
      ℓ[i] = ℓ[i-1]-ℓ[i]
  end
  ℓ[1] = 1 - ℓ[1]
  push!(ℓ,ℓ[end])

  H = vcat(S,[D,0.]) .* ℓ .^ (1/α)
  err = H[end-1]

  # Labels: 0 belongs to both estimates, 1 to upper estimate and -1 to lower estimate
  lab = Int.(zeros(length(ℓ)))
  lab[end-1] = 1
  lab[end] = -1
  S = H ./ ℓ

  df = DataFrame(length = ℓ, height = H, slope = S, label = lab)
  sort!(df, order(:slope, rev = false))

  return (df,err)
end

import Distributions.rand
# @Input: ConvexMinorantStableMeander
# @Output: (df,err)=(faces,error)
# Note: We follow the algorithms in the EpsStrongStable paper, with slight modifications
# The initial face's length should be replaced with 0 to get a lower estimate
function rand(d::ConvexMinorantStableMeander)
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
  push!(lagS,S[σ])
  push!(lagU,U[σ])
  # Line 20
  if D <= eps # Finished!
    # Line 21
    (df,err) = faces(d.α,D,lagS,lagU)
    return (df,err,σ)
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
    push!(lagS,S[σ])
    push!(lagU,U[σ])
    # Line 24
    if D <= eps # Finished!
      # Line 25
      (df,err) = faces(d.α,D,lagS,lagU)
      return (df,err,σ)
    end # Line 26
  end # Line 27
end

struct CrossConvexMinorantStableMeander <: Distribution{Multivariate,Continuous}
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
  CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,mAst::Int,f::Function) =
    (β < -1 || β > 1 || (β == -1 && α <= 1) || α <= 0 || α > 2) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") : (0 >= γ || γ >= α) ?
    error("Parameters' requirements unmet: α>γ>0") : (0 >= δ || δ >= d || d >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(αρ)>d>δ>0") : κ < 0 ?
    error("Parameters' requirements unmet: κ≥0") : Δ <= 0 ?
    error("Parameters' requirements unmet: Δ≥0") : mAst < 0 ?
    error("Parameters' requirements unmet: m*≥0") : new(α, β, β*(α <= 1 ? 1 : (α-2)/α),(1+β*(α <= 1 ? 1 : (α-2)/α))/2 , d,
      etaF(d*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2 , δ, γ, κ + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),
      log(2)*2/(3*etaF(d*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))), Δ, mAst, cross_func(f))
  CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Int,f::Function) = CrossConvexMinorantStableMeander(α,β,d,δ,γ,κ,Δ,12,f)
  CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,f::Function) = CrossConvexMinorantStableMeander(α,β,d,δ,γ,κ,
    Int(ceil(33*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),f)
  CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,f::Function) = CrossConvexMinorantStableMeander(α,β,d,δ,γ,4,f)
  CrossConvexMinorantStableMeander(α::Real,β::Real,Δ::Integer,mAst::Integer,f::Function) = CrossConvexMinorantStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,4,Δ,mAst,f)
  CrossConvexMinorantStableMeander(α::Real,β::Real,Δ::Integer,f::Function) = CrossConvexMinorantStableMeander(α,β,(2/3)*2/(α+β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,4,Δ,12,f)
  CrossConvexMinorantStableMeander(α::Real,β::Real,f::Function) = CrossConvexMinorantStableMeander(α,β,Int(ceil(33*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),f)
end

# @Input: CrossConvexMinorantStableMeander
# @Output: The triple (B,df,err) where (df,err) is the output of the function 'faces' and
# B is a boolean indicating a crossing of the function f or not
# Note: We follow the algorithms in the EpsStrongStable paper, with slight modifications
# The initial face's length should be replaced with 0 to get a lower estimate
function rand(d::CrossConvexMinorantStableMeander)
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
  # Line 20
  # Line 21
  push!(lagS,S[1])
  push!(lagU,U[1])
  # Line 25
  (df,err) = faces(d.α,D,lagS,lagU)
  dfL = @linq df |>
      where(:label .< 1) |>
      select(:length,:height)
  dfU = @linq df |>
      where(:label .> -1) |>
      select(:length,:height)
  aux = d.f(dfL)
  if aux == d.f(dfU) # Crossing / non-crossing detection!
    return (aux,df,err)
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
    # Line 24
    push!(lagS,S[σ])
    push!(lagU,U[σ])
    # Line 25
    (df,err) = faces(d.α,D,lagS,lagU)
    dfL = @linq df |>
        where(:label .< 1) |>
        select(:length,:height)
    dfU = @linq df |>
        where(:label .> -1) |>
        select(:length,:height)
    aux = d.f(dfL)
    if aux == d.f(dfU)
      return (aux,df,err)
    end # Line 26
  end # Line 27
end

struct ConvexMinorantWeaklyStable <: Distribution{Multivariate,Continuous}
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  # Scale, drift and time horizon
  δ::Float64
  μ::Float64
  T::Float64
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
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,ε::Float64) =
    (β < -1 || β > 1 || α <= 0 || α > 2) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]") :
    (δ <= 0) ?
    error("Parameters' requirements unmet: δ>0") :
    (T <= 0) ?
    error("Parameters' requirements unmet: T>0") :
    (β == -1 && α <= 1) ? (0 >= γ_l || γ_l >= α) ?
      error("Parameters' requirements unmet: α>γ_l>0") : (0 >= δ_l || δ_l >= d_l || d_l >= 1/α) ?
      error("Parameters' requirements unmet: 1/α>d_l>δ_l>0") : κ_l < 0 ?
      error("Parameters' requirements unmet: κ_l≥0") : Δ_l <= 0 ?
      error("Parameters' requirements unmet: Δ_l≥0") : mAst_l < 0 ?
      error("Parameters' requirements unmet: m_l*≥0") : ε <= 0 ?
      error("Parameters' requirements unmet: ε>0") : new(α,-1.,-1.,0.,δ,μ,T,d_l,d_r,etaF(d_l*α)*α,0.,
        δ_l,δ_r,γ_l,γ_r,κ_l + Int(ceil(max(1/α,log(2)*2/(3*etaF(d_l*α)*α)))),κ_r,Δ_l,Δ_r,mAst_l,mAst_r,ε) :
    (β == 1 && α <= 1) ? (0 >= γ_r || γ_r >= α) ?
      error("Parameters' requirements unmet: α>γ_r>0") : (0 >= δ_r || δ_r >= d_r || d_r >= 1/α) ?
      error("Parameters' requirements unmet: 1/α>d_r>δ_r>0") : κ_r < 0 ?
      error("Parameters' requirements unmet: κ_r≥0") : Δ_r <= 0 ?
      error("Parameters' requirements unmet: Δ_r≥0") : mAst_r < 0 ?
      error("Parameters' requirements unmet: m_r*≥0") : ε <= 0 ?
      error("Parameters' requirements unmet: ε>0") : new(α,1.,1.,1.,δ,μ,T,d_l,d_r,0.,etaF(d_r*α)*α,
        δ_l,δ_r,γ_l,γ_r,κ_l,κ_r + Int(ceil(max(1/α,log(2)*2/(3*etaF(d_r*α)*α)))),Δ_l,Δ_r,mAst_l,mAst_r,ε) :
    (0 >= γ_l || γ_l >= α || 0 >= γ_r || γ_r >= α) ?
    error("Parameters' requirements unmet: α>γ_l>0 and α>γ_r>0") : (0 >= δ_l || δ_l >= d_l || 0 >= δ_r || δ_r >= d_r || d_l >= 2/(α - β*(α <= 1 ? α : α-2)) || d_r >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(α(1-ρ))>d_l>δ_l>0 and 1/(αρ)>dr>δr>0") : (κ_l < 0 || κ_r < 0) ?
    error("Parameters' requirements unmet: κ_l,κ_r≥0") : (Δ_l <= 0 || Δ_r <= 0) ?
    error("Parameters' requirements unmet: Δ_l,Δr≥0") : (mAst_l < 0 || mAst_r < 0) ?
    error("Parameters' requirements unmet: m_l*,m_r*≥0") : ε <= 0 ?
    error("Parameters' requirements unmet: ε>0") : new(α, β, β*(α <= 1 ? 1 : (α-2)/α),(1+β*(α <= 1 ? 1 : (α-2)/α))/2 , δ, μ, T, d_l, d_r,
      etaF(d_l*(α - β*(α <= 1 ? α : α-2))/2)*(α - β*(α <= 1 ? α : α-2))/2, etaF(d_r*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2,
      δ_l, δ_r, γ_l, γ_r,
      κ_l + Int(ceil(max(2/(α-β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_l*(α-β*(α <= 1 ? α : α-2))/2)*(α-β*(α <= 1 ? α : α-2)) ) ))),
      κ_r + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_r*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))),
      Δ_l, Δ_r, mAst_l, mAst_r, ε)
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,mAst_l,mAst_r,.5^16)
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,12,12)
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))))
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,4,4)
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α)
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,Δ_l::Integer,Δ_r::Integer,ε::Real) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α,4,4,Δ_l,Δ_r,12,12,ε)
  ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,ε::Real) = ConvexMinorantWeaklyStable(α,β,δ,μ,T,
    Int(ceil(abs(log(ε/2))/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(abs(log(ε/2))/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),ε)
end

# @Input: ConvexMinorantStableMeander
# @Output: (df,err_left,err_right)=(faces,error) where err_left and err_right
# are the errors produced by the pre- and post-minumum convex minorants (before tilting)
function rand(d::ConvexMinorantWeaklyStable)
  ra = 1/d.α
  ϵ = d.ε / (d.δ * d.T^ra)

  if d.α <= 1 && d.β == -1
    (df,err) = rand(ConvexMinorantStableMeander(d.α,1,d.d_l,d.δ_l,d.γ_l,d.κ_l,d.Δ_l,d.mAst_l,ϵ))
    sort!(df, order(:slope, rev = true))
    df[:length] = df[:length] .* d.T
    df[:height] = df[:height] .* (-d.δ * d.T^ra) .+ df[:length] .* d.μ
    df[:slope] = df[:slope] .* (-d.δ * d.T^(ra-1)) .+ d.μ
    return (df,err * d.δ * d.T^ra,0.)
  elseif d.α <= 1 && d.β == 1
    (df,err) = rand(ConvexMinorantStableMeander(d.α,1,d.d_r,d.δ_r,d.γ_r,d.κ_r,d.Δ_r,d.mAst_r,ϵ))
    df[:length] = df[:length] .* d.T
    df[:height] = df[:height] .* (d.δ * d.T^ra) .+ df[:length] .* d.μ
    df[:slope] = df[:slope] .* (d.δ * d.T^(ra-1)) .+ d.μ
    return (df,0,err * d.δ * d.T^ra)
  end
  τ = rand(Beta(1-d.ρ,d.ρ))
  T_l = d.T * τ
  T_r = d.T * (1-τ)

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
  U_l, U_r = Float64[], Float64[]
  # Line 1
  x_l, x_r = Inf, Inf
  σ, σ = 0, 0
  # Line 2
  lagU_l, lagU_r = rand(Beta(1-d.ρ,1),d.Δ_l), rand(Beta(d.ρ,1),d.Δ_r)
  lagS_l, lagS_r = rand(dPos_l,d.Δ_l), rand(dPos_r,d.Δ_r)
  eps_l, eps_r = 1 / (τ * prod(lagU_l))^ra, 1 / ((1-τ) * prod(lagU_r))^ra

  # Line 3 (first iteration)
  # Line 4 (Algorithm 3 with unconditional probabilities)
  S_l, S_r = rand(dPos_l,mAst_l), rand(dPos_r,mAst_r)
  append!(S_l,chiStable1(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,mAst_l-1))
  append!(S_r,chiStable1(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,mAst_r-1))

  R_l, R_r = Float64[], Float64[]
  C_l, C_r = [0.], [0.]
  F_l, F_r = Float64[], Float64[]
  # Last value of C
  endC_l, endC_r = 0., 0.
  # First iteration
  # Line 5
  while (length(R_l) == σ) || (length(S_l) >= length(C_l))
    t_l = length(C_l)
    # Lines 6 and 7
    (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
    append!(F_l,F1_l)
    append!(C_l,endC_l .+ C1_l)
    endC_l = C_l[end]
    # Lines 8 and 9
    if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      # Line 10
      x_l = endC_l + d.κ_l
      R1_l = [maximum(C_l[t_l:end])]
      for i = (t_l-1):-1:(length(R_l)+1)
        push!(R1_l,max(R1_l[end],C_l[i]))
      end
      append!(R_l,R1_l[end:-1:1] .- C_l[(length(R_l)+1):t_l])
    else # Line 11
      # Lines 12 and 13
      (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      append!(F_l,F1_l)
      append!(C_l,endC_l .+ C1_l)
      endC_l = C_l[end]
    end # Line 14
  end # Line 15 - left
  while (length(R_r) == σ) || (length(S_r) >= length(C_r))
    t_r = length(C_r)
    # Lines 6 and 7
    (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
    append!(F_r,F1_r)
    append!(C_r,endC_r .+ C1_r)
    endC_r = C_r[end]
    # Lines 8 and 9
    if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
      # Line 10
      x_r = endC_r + d.κ_r
      R1_r = [maximum(C_r[t_r:end])]
      for i = (t_r-1):-1:(length(R_r)+1)
        push!(R1_r,max(R1_r[end],C_r[i]))
      end
      append!(R_r,R1_r[end:-1:1] .- C_r[(length(R_r)+1):t_r])
    else # Line 11
      # Lines 12 and 13
      (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
      append!(F_r,F1_r)
      append!(C_r,endC_r .+ C1_r)
      endC_r = C_r[end]
    end # Line 14
  end # Line 15 - right
  # Line 16
  while length(U_l) < length(S_l)
    push!(U_l,exp(-d.α*F_l[length(U_l)+1]))
  end
  while length(U_r) < length(S_r)
    push!(U_r,exp(-d.α*F_r[length(U_r)+1]))
  end
  # Lines 17, 18 and 19
  σ, σ = 1, 1
  D_l = exp(R_l[σ])*( e2_l*exp(-sep_l*(length(S_l)-σ-1)) + sum(S_l[σ:end] .*
      exp.(-d.d_l .* (0:(length(S_l)-σ))) .* (1 .- U_l[σ:end]) .^ ra) )
  D_r = exp(R_r[σ])*( e2_r*exp(-sep_r*(length(S_r)-σ-1)) + sum(S_r[σ:end] .*
      exp.(-d.d_r .* (0:(length(S_r)-σ))) .* (1 .- U_r[σ:end]) .^ ra) )
  eps_l /= U_l[σ]^ra
  eps_r /= U_r[σ]^ra

  push!(lagS_l,S_l[σ])
  push!(lagS_r,S_r[σ])
  push!(lagU_l,U_l[σ])
  push!(lagU_r,U_r[σ])
  # Line 20
  if D_l/eps_l + D_r/eps_r <= ϵ # Finished!
    # Line 21
    (df_l,err_l) = faces(d.α,D_l,lagS_l,lagU_l)
    (df_r,err_r) = faces(d.α,D_r,lagS_r,lagU_r)

    df_l = @linq df_l |>
         where(:label .> -1) |>
         select(:length,:height,:slope,:label)
    df_l[:label] = 0
    sort!(df_l, order(:slope, rev = true))
    df_l[:length] = df_l[:length] .* T_l
    df_l[:height] = df_l[:height] .* (-d.δ * T_l^ra) .+ df_l[:length] .* d.μ
    df_l[:slope] = df_l[:slope] .* (-d.δ * T_l^(ra-1)) .+ d.μ
    df_r[:length] = df_r[:length] .* T_r
    df_r[:height] = df_r[:height] .* (d.δ * T_r^ra) .+ df_r[:length] .* d.μ
    df_r[:slope] = df_r[:slope] .* (d.δ * T_r^(ra-1)) .+ d.μ

    append!(df_l,df_r)
    return (df_l,err_l * d.δ * T_l^ra,err_r * d.δ * T_r^ra)
  end # Line 22
  # Line 3 (second to last iterations)
  while true
    # Line 4 (Algorithm 3 with conditional probabilities)
    append!(S_l,chiStable2(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,length(S_l)-σ-1))
    append!(S_r,chiStable2(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,length(S_r)-σ-1))
    # Line 5
    while (length(R_l) == σ) || (length(S_l) >= length(C_l))
      t_l = length(C_l)-1
      # Lines 6 and 7
      (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      append!(F_l,F1_l)
      append!(C_l,endC_l .+ C1_l)
      endC_l = C_l[end]
      # Lines 8 and 9
      if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        # Line 10
        x_l = endC_l + d.κ_l
        R1_l = [maximum(C_l[t_l:end])]
        for i = (t_l-1):-1:(length(R_l)+1)
          push!(R1_l,max(R1_l[end],C_l[i]))
        end
        append!(R_l,R1_l[end:-1:1] .- C_l[(length(R_l)+1):t_l])
      else # Line 11
        # Lines 12 and 13
        (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        append!(F_l,F1_l)
        append!(C_l,endC_l .+ C1_l)
        endC_l = C_l[end]
      end # Line 14
    end # Line 15
    while (length(R_r) == σ) || (length(S_r) >= length(C_r))
      t_r = length(C_r)-1
      # Lines 6 and 7
      (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
      append!(F_r,F1_r)
      append!(C_r,endC_r .+ C1_r)
      endC_r = C_r[end]
      # Lines 8 and 9
      if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
        # Line 10
        x_r = endC_r + d.κ_r
        R1_r = [maximum(C_r[t_r:end])]
        for i = (t_r-1):-1:(length(R_r)+1)
          push!(R1_r,max(R1_r[end],C_r[i]))
        end
        append!(R_r,R1_r[end:-1:1] .- C_r[(length(R_r)+1):t_r])
      else # Line 11
        # Lines 12 and 13
        (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
        append!(F_r,F1_r)
        append!(C_r,endC_r .+ C1_r)
        endC_r = C_r[end]
      end # Line 14
    end # Line 15
    # Line 16
    while length(U_l) < length(S_l)
      push!(U_l,exp(-d.α*F_l[length(U_l)+1]))
    end
    while length(U_r) < length(S_r)
      push!(U_r,exp(-d.α*F_r[length(U_r)+1]))
    end
    # Lines 17, 18 and 19
    σ += 1
    σ += 1
    D_l = exp(R_l[σ])*( e2_l*exp(-sep_l*(length(S_l)-σ-1)) + sum(S_l[σ:end] .*
        exp.(-d.d_l .* (0:(length(S_l)-σ))) .* (1 .- U_l[σ:end]) .^ ra) )
    D_r = exp(R_r[σ])*( e2_r*exp(-sep_r*(length(S_r)-σ-1)) + sum(S_r[σ:end] .*
        exp.(-d.d_r .* (0:(length(S_r)-σ))) .* (1 .- U_r[σ:end]) .^ ra) )
    eps_l /= U_l[σ]^ra
    eps_r /= U_r[σ]^ra
    push!(lagS_l,S_l[σ])
    push!(lagS_r,S_r[σ])
    push!(lagU_l,U_l[σ])
    push!(lagU_r,U_r[σ])
    # Line 24
    if D_l/eps_l + D_r/eps_r <= ϵ # Finished!
      # Line 25
      (df_l,err_l) = faces(d.α,D_l,lagS_l,lagU_l)
      (df_r,err_r) = faces(d.α,D_r,lagS_r,lagU_r)

      df_l = @linq df_l |>
           where(:label .> -1) |>
           select(:length,:height,:slope,:label)
      df_l[:label] = 0
      sort!(df_l, order(:slope, rev = true))
      df_l[:length] = df_l[:length] .* T_l
      df_l[:height] = df_l[:height] .* (-d.δ * T_l^ra) .+ df_l[:length] .* d.μ
      df_l[:slope] = df_l[:slope] .* (-d.δ * T_l^(ra-1)) .+ d.μ
      df_r[:length] = df_r[:length] .* T_r
      df_r[:height] = df_r[:height] .* (d.δ * T_r^ra) .+ df_r[:length] .* d.μ
      df_r[:slope] = df_r[:slope] .* (d.δ * T_r^(ra-1)) .+ d.μ

      append!(df_l,df_r)
      return (df_l,err_l * d.δ * T_l^ra,err_r * d.δ * T_r^ra)
    end # Line 26
  end # Line 27
end

struct CrossConvexMinorantWeaklyStable <: Distribution{Multivariate,Continuous}
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  # Scale, drift and time horizon
  δ::Float64
  μ::Float64
  T::Float64
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
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,mAst_l::Int,mAst_r::Int,f::Function) =
    (β < -1 || β > 1 || α <= 0 || α > 2) ?
    error("Parameters' requirements unmet: (α,β)∈(0,2]×[-1,1]") :
    (δ <= 0) ?
    error("Parameters' requirements unmet: δ>0") :
    (T <= 0) ?
    error("Parameters' requirements unmet: T>0") :
    (β == -1 && α <= 1) ? (0 >= γ_l || γ_l >= α) ?
      error("Parameters' requirements unmet: α>γ_l>0") : (0 >= δ_l || δ_l >= d_l || d_l >= 1/α) ?
      error("Parameters' requirements unmet: 1/α>d_l>δ_l>0") : κ_l < 0 ?
      error("Parameters' requirements unmet: κ_l≥0") : Δ_l <= 0 ?
      error("Parameters' requirements unmet: Δ_l≥0") : mAst_l < 0 ?
      error("Parameters' requirements unmet: m_l*≥0") : new(α,-1.,-1.,0.,δ,μ,T,d_l,d_r,etaF(d_l*α)*α,0.,
        δ_l,δ_r,γ_l,γ_r,κ_l + Int(ceil(max(1/α,log(2)*2/(3*etaF(d_l*α)*α)))),κ_r,Δ_l,Δ_r,mAst_l,mAst_r,f) :
    (β == 1 && α <= 1) ? (0 >= γ_r || γ_r >= α) ?
      error("Parameters' requirements unmet: α>γ_r>0") : (0 >= δ_r || δ_r >= d_r || d_r >= 1/α) ?
      error("Parameters' requirements unmet: 1/α>d_r>δ_r>0") : κ_r < 0 ?
      error("Parameters' requirements unmet: κ_r≥0") : Δ_r <= 0 ?
      error("Parameters' requirements unmet: Δ_r≥0") : mAst_r < 0 ?
      error("Parameters' requirements unmet: m_r*≥0") : new(α,1.,1.,1.,δ,μ,T,d_l,d_r,0.,etaF(d_r*α)*α,
        δ_l,δ_r,γ_l,γ_r,κ_l,κ_r + Int(ceil(max(1/α,log(2)*2/(3*etaF(d_r*α)*α)))),Δ_l,Δ_r,mAst_l,mAst_r,f) :
    (0 >= γ_l || γ_l >= α || 0 >= γ_r || γ_r >= α) ?
    error("Parameters' requirements unmet: α>γ_l>0 and α>γ_r>0") : (0 >= δ_l || δ_l >= d_l || 0 >= δ_r || δ_r >= d_r || d_l >= 2/(α - β*(α <= 1 ? α : α-2)) || d_r >= 2/(α + β*(α <= 1 ? α : α-2))) ?
    error("Parameters' requirements unmet: 1/(α(1-ρ))>d_l>δ_l>0 and 1/(αρ)>dr>δr>0") : (κ_l < 0 || κ_r < 0) ?
    error("Parameters' requirements unmet: κ_l,κ_r≥0") : (Δ_l <= 0 || Δ_r <= 0) ?
    error("Parameters' requirements unmet: Δ_l,Δr≥0") : (mAst_l < 0 || mAst_r < 0) ?
    error("Parameters' requirements unmet: m_l*,m_r*≥0") : new(α, β, β*(α <= 1 ? 1 : (α-2)/α),(1+β*(α <= 1 ? 1 : (α-2)/α))/2, δ, μ, T, d_l, d_r,
      etaF(d_l*(α - β*(α <= 1 ? α : α-2))/2)*(α - β*(α <= 1 ? α : α-2))/2, etaF(d_r*(α + β*(α <= 1 ? α : α-2))/2)*(α + β*(α <= 1 ? α : α-2))/2,
      δ_l, δ_r, γ_l, γ_r,
      κ_l + Int(ceil(max(2/(α-β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_l*(α-β*(α <= 1 ? α : α-2))/2)*(α-β*(α <= 1 ? α : α-2)) ) ))),
      κ_r + Int(ceil(max(2/(α+β*(α <= 1 ? α : α-2)),log(2)*2/(3*etaF(d_r*(α+β*(α <= 1 ? α : α-2))/2)*(α+β*(α <= 1 ? α : α-2)) ) ))),
      Δ_l, Δ_r, mAst_l, mAst_r, f)
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Int,Δ_r::Int,f::Function) = CrossConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,Δ_l,Δ_r,12,12,f)
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,f::Function) = CrossConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,κ_l,κ_r,
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(17*log(2)/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),f)
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,f::Function) = CrossConvexMinorantWeaklyStable(α,β,δ,μ,T,d_l,d_r,δ_l,δ_r,γ_l,γ_r,4,4,f)
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,f::Function) = CrossConvexMinorantWeaklyStable(α,β,δ,μ,T,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α,f)
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,Δ_l::Integer,Δ_r::Integer,f::Function) = CrossConvexMinorantWeaklyStable(α,β,δ,μ,T,
    (2/3)*2/(α-β*(α <= 1 ? α : α-2)),(2/3)*2/(α+β*(α <= 1 ? α : α-2)),
    (1/3)*2/(α-β*(α <= 1 ? α : α-2)),(1/3)*2/(α+β*(α <= 1 ? α : α-2)),.95*α,.95*α,4,4,Δ_l,Δ_r,12,12,f)
  CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,f::Function) = CrossConvexMinorantWeaklyStable(α,β,δ,μ,T,
    Int(ceil(abs(log(.5^16))/log(gamma(1+1/α+(1-β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1-β*(α <= 1 ? 1 : (α-2)/α))/2))))),
    Int(ceil(abs(log(.5^16))/log(gamma(1+1/α+(1+β*(α <= 1 ? 1 : (α-2)/α))/2)/(gamma(1+1/α)*gamma(1+(1+β*(α <= 1 ? 1 : (α-2)/α))/2))))),f)
end

# @Input: CrossConvexMinorantStableMeander
# @Output: The triple (B,df,err_left,err_right) where
# B is a boolean indicating a crossing of the function f or not
function rand(d::CrossConvexMinorantWeaklyStable)
  ra = 1/d.α

  if d.α <= 1 && d.β == -1
    sc_l = d.δ * d.T^ra
    sc1_l = d.δ * d.T^(ra-1)

    F = cross_func(d.f)

    ar_l = d.α*(1-d.ρ)
    et1_l = 1/(ar_l+d.η_l)
    rar_l = 1/ar_l

    # Conditionally positive stable distribution
    dPos_l = PositiveStable(d.α,1)
    sep_l = d.d_l - d.δ_l
    e2_l = 1 /(1 -exp(-sep_l))
    mom_l = mellin(dPos_l,d.γ_l)
    dega_l = d.δ_l*d.γ_l
    dega2_l = -1 /(1 -exp(-dega_l))
    mAst_l = d.mAst_l + Int(ceil( max(0,log(mom_l)/dega_l) + rar_l ))
    # θ sequence
    U_l = Float64[]
    # Line 1
    x_l = Inf
    σ = 0
    # Line 2
    lagU_l = rand(Uniform(),d.Δ_l)
    lagS_l = rand(dPos_l,d.Δ_l)

    # Line 3 (first iteration)
    # Line 4 (Algorithm 3 with unconditional probabilities)
    S_l = rand(dPos_l,mAst_l)
    append!(S_l,chiStable1(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,mAst_l-1))

    R_l = Float64[]
    C_l = [0.]
    F_l = Float64[]
    # Last value of C
    endC_l = 0.
    # First iteration
    # Line 5
    while (length(R_l) == σ) || (length(S_l) >= length(C_l))
      t_l = length(C_l)
      # Lines 6 and 7
      (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      append!(F_l,F1_l)
      append!(C_l,endC_l .+ C1_l)
      endC_l = C_l[end]
      # Lines 8 and 9
      if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        # Line 10
        x_l = endC_l + d.κ_l
        R1_l = [maximum(C_l[t_l:end])]
        for i = (t_l-1):-1:(length(R_l)+1)
          push!(R1_l,max(R1_l[end],C_l[i]))
        end
        append!(R_l,R1_l[end:-1:1] .- C_l[(length(R_l)+1):t_l])
      else # Line 11
        # Lines 12 and 13
        (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        append!(F_l,F1_l)
        append!(C_l,endC_l .+ C1_l)
        endC_l = C_l[end]
      end # Line 14
    end # Line 15 - left
    # Line 16
    while length(U_l) < length(S_l)
      push!(U_l,exp(-d.α*F_l[length(U_l)+1]))
    end
    # Lines 17, 18 and 19
    σ = 1
    D_l = exp(R_l[σ])*( e2_l*exp(-sep_l*(length(S_l)-σ-1)) + sum(S_l[σ:end] .*
        exp.(-d.d_l .* (0:(length(S_l)-σ))) .* (1 .- U_l[σ:end]) .^ ra) )

    push!(lagS_l,S_l[σ])
    push!(lagU_l,U_l[σ])
    # Line 20
    # Construct the dataframes
    (df_l,err_l) = faces(d.α,D_l,lagS_l,lagU_l)
    df_l = @linq df_l |>
         where(:label .> -1) |>
         select(:length,:height)
    # Rescale and add the drift
    sort!(df_l, order(:slope, rev = true))
    df_l[:length] = df_l[:length] .* d.T
    df_l[:height] = df_l[:height]  .* (-sc_l) .+ df_l[:length] .* d.μ
    df_l[:slope] = df_l[:slope] .* (-sc1_l) .+ d.μ
    # Construct (the dataframes with the faces of) the sandwiching convex functions
    err_l *= sc_l
    # Check for crossing:
    aux = F(df_l)
    G = cross_func(d.f,-err_l)
    if aux == G(df_l) # Crossing / non-crossing detection!
      return (aux,df_l,err_l,0.)
    end # Line 22
    # Line 3 (second to last iterations)
    while true
      # Line 4 (Algorithm 3 with conditional probabilities)
      append!(S_l,chiStable2(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,length(S_l)-σ-1))
      # Line 5
      while (length(R_l) == σ) || (length(S_l) >= length(C_l))
        t_l = length(C_l)-1
        # Lines 6 and 7
        (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        append!(F_l,F1_l)
        append!(C_l,endC_l .+ C1_l)
        endC_l = C_l[end]
        # Lines 8 and 9
        if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
          # Line 10
          x_l = endC_l + d.κ_l
          R1_l = [maximum(C_l[t_l:end])]
          for i = (t_l-1):-1:(length(R_l)+1)
            push!(R1_l,max(R1_l[end],C_l[i]))
          end
          append!(R_l,R1_l[end:-1:1] .- C_l[(length(R_l)+1):t_l])
        else # Line 11
          # Lines 12 and 13
          (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
          append!(F_l,F1_l)
          append!(C_l,endC_l .+ C1_l)
          endC_l = C_l[end]
        end # Line 14
      end # Line 15
      # Line 16
      while length(U_l) < length(S_l)
        push!(U_l,exp(-d.α*F_l[length(U_l)+1]))
      end
      # Lines 17, 18 and 19
      σ += 1
      D_l = exp(R_l[σ])*( e2_l*exp(-sep_l*(length(S_l)-σ-1)) + sum(S_l[σ:end] .*
          exp.(-d.d_l .* (0:(length(S_l)-σ))) .* (1 .- U_l[σ:end]) .^ ra) )
      push!(lagS_l,S_l[σ])
      push!(lagU_l,U_l[σ])
      # Line 24
      # Construct the dataframes
      (df_l,err_l) = faces(d.α,D_l,lagS_l,lagU_l)
      # Rescale and add the drift
      sort!(df_l, order(:slope, rev = true))
      df_l[:length] = df_l[:length] .* d.T
      df_l[:height] = df_l[:height]  .* (-sc_l) .+ df_l[:length] .* d.μ
      df_l[:slope] = df_l[:slope] .* (-sc1_l) .+ d.μ
      # Construct (the dataframes with the faces of) the sandwiching convex functions
      df = @linq df_l |>
          where(:label .> -1) |>
          select(:length,:height)
      err_l *= sc_l
      # Check for crossing:
      aux = F(df)
      G = cross_func(d.f,-err_l)
      if aux == G(df) # Crossing / non-crossing detection!
        return (aux,df_l,err_l,0.)
      end # Line 26
    end # Line 27
  elseif d.α <= 1 && d.β == 1
    (aux,df,err) = rand(CrossConvexMinorantStableMeander(d.α,1,d.d_r,d.δ_r,d.γ_r,d.κ_r,d.Δ_r,d.mAst_r,d.f))
    df[:length] = df[:length] .* d.T
    df[:height] = df[:height] .* (d.δ * d.T^ra) .+ df[:length] .* d.μ
    df[:slope] = df[:slope] .* (d.δ * d.T^(ra-1)) .+ d.μ
    return (aux,df,0,err * d.δ * d.T^ra)
  end
  τ = rand(Beta(1-d.ρ,d.ρ))
  T_l = d.T * τ
  T_r = d.T * (1-τ)

  sc_l = d.δ * T_l^ra
  sc_r = d.δ * T_r^ra
  sc1_l = d.δ * T_l^(ra-1)
  sc1_r = d.δ * T_r^(ra-1)

  F = cross_func(d.f)

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
  U_l, U_r = Float64[], Float64[]
  # Line 1
  x_l, x_r = Inf, Inf
  σ = 0
  # Line 2
  lagU_l, lagU_r = rand(Beta(1-d.ρ,1),d.Δ_l), rand(Beta(d.ρ,1),d.Δ_r)
  lagS_l, lagS_r = rand(dPos_l,d.Δ_l), rand(dPos_r,d.Δ_r)

  # Line 3 (first iteration)
  # Line 4 (Algorithm 3 with unconditional probabilities)
  S_l, S_r = rand(dPos_l,mAst_l), rand(dPos_r,mAst_r)
  append!(S_l,chiStable1(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,mAst_l-1))
  append!(S_r,chiStable1(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,mAst_r-1))

  R_l, R_r = Float64[], Float64[]
  C_l, C_r = [0.], [0.]
  F_l, F_r = Float64[], Float64[]
  # Last value of C
  endC_l, endC_r = 0., 0.
  # First iteration
  # Line 5
  while (length(R_l) == σ) || (length(S_l) >= length(C_l))
    t_l = length(C_l)
    # Lines 6 and 7
    (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
    append!(F_l,F1_l)
    append!(C_l,endC_l .+ C1_l)
    endC_l = C_l[end]
    # Lines 8 and 9
    if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      # Line 10
      x_l = endC_l + d.κ_l
      R1_l = [maximum(C_l[t_l:end])]
      for i = (t_l-1):-1:(length(R_l)+1)
        push!(R1_l,max(R1_l[end],C_l[i]))
      end
      append!(R_l,R1_l[end:-1:1] .- C_l[(length(R_l)+1):t_l])
    else # Line 11
      # Lines 12 and 13
      (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      append!(F_l,F1_l)
      append!(C_l,endC_l .+ C1_l)
      endC_l = C_l[end]
    end # Line 14
  end # Line 15 - left
  while (length(R_r) == σ) || (length(S_r) >= length(C_r))
    t_r = length(C_r)
    # Lines 6 and 7
    (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
    append!(F_r,F1_r)
    append!(C_r,endC_r .+ C1_r)
    endC_r = C_r[end]
    # Lines 8 and 9
    if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
      # Line 10
      x_r = endC_r + d.κ_r
      R1_r = [maximum(C_r[t_r:end])]
      for i = (t_r-1):-1:(length(R_r)+1)
        push!(R1_r,max(R1_r[end],C_r[i]))
      end
      append!(R_r,R1_r[end:-1:1] .- C_r[(length(R_r)+1):t_r])
    else # Line 11
      # Lines 12 and 13
      (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
      append!(F_r,F1_r)
      append!(C_r,endC_r .+ C1_r)
      endC_r = C_r[end]
    end # Line 14
  end # Line 15 - right
  # Line 16
  while length(U_l) < length(S_l)
    push!(U_l,exp(-d.α*F_l[length(U_l)+1]))
  end
  while length(U_r) < length(S_r)
    push!(U_r,exp(-d.α*F_r[length(U_r)+1]))
  end
  # Lines 17, 18 and 19
  σ = 1
  D_l = exp(R_l[σ])*( e2_l*exp(-sep_l*(length(S_l)-σ-1)) + sum(S_l[σ:end] .*
      exp.(-d.d_l .* (0:(length(S_l)-σ))) .* (1 .- U_l[σ:end]) .^ ra) )
  D_r = exp(R_r[σ])*( e2_r*exp(-sep_r*(length(S_r)-σ-1)) + sum(S_r[σ:end] .*
      exp.(-d.d_r .* (0:(length(S_r)-σ))) .* (1 .- U_r[σ:end]) .^ ra) )

  push!(lagS_l,S_l[σ])
  push!(lagS_r,S_r[σ])
  push!(lagU_l,U_l[σ])
  push!(lagU_r,U_r[σ])
  # Line 20
  # Construct the dataframes
  (df_l,err_l) = faces(d.α,D_l,lagS_l,lagU_l)
  (df_r,err_r) = faces(d.α,D_r,lagS_r,lagU_r)

  # Rescale and add the drift
  df_l = @linq df_l |>
       where(:label .> -1) |>
       select(:length,:height,:slope,:label)
  df_l[:label] = 0
  sort!(df_l, order(:slope, rev = true))
  df_l[:length] = df_l[:length] .* T_l
  df_l[:height] = df_l[:height]  .* (-sc_l) .+ df_l[:length] .* d.μ
  df_l[:slope] = df_l[:slope] .* (-sc1_l) .+ d.μ
  df_r[:length] = df_r[:length] .* T_r
  df_r[:height] = df_r[:height] .* sc_r .+ df_r[:length] .* d.μ
  df_r[:slope] = df_r[:slope] .* sc1_r .+ d.μ
  df_l1 = deepcopy(df_l)
  df_r1 = deepcopy(df_r)

  # Construct (the dataframes with the faces of) the sandwiching convex functions
  dfL_l = @linq df_l1 |>
       select(:length,:height)
  dfU_l = copy(dfL_l)
  dfL_r = @linq df_r1 |>
      where(:label .< 1) |>
      select(:length,:height)
  dfU_r = @linq df_r1 |>
      where(:label .> -1) |>
      select(:length,:height)
  append!(dfL_l,dfL_r)
  append!(dfU_l,dfU_r)
  err_l *= sc_l

  # Check for crossing:
  aux = F(dfU_l)
  G = cross_func(d.f,-err_l)
  if aux == G(dfL_l) # Crossing / non-crossing detection!
    append!(df_l,df_r)
    return (aux,df_l,err_l,err_r * sc_r,σ)
  end # Line 22
  # Line 3 (second to last iterations)
  while true
    # Line 4 (Algorithm 3 with conditional probabilities)
    append!(S_l,chiStable2(dPos_l,d.δ_l,dega_l,dega2_l,mom_l,length(S_l)-σ-1))
    append!(S_r,chiStable2(dPos_r,d.δ_r,dega_r,dega2_r,mom_r,length(S_r)-σ-1))
    # Line 5
    while (length(R_l) == σ) || (length(S_l) >= length(C_l))
      t_l = length(C_l)-1
      # Lines 6 and 7
      (C1_l,F1_l) = downRW(d.d_l,rar_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
      append!(F_l,F1_l)
      append!(C_l,endC_l .+ C1_l)
      endC_l = C_l[end]
      # Lines 8 and 9
      if reflectedProcess(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        # Line 10
        x_l = endC_l + d.κ_l
        R1_l = [maximum(C_l[t_l:end])]
        for i = (t_l-1):-1:(length(R_l)+1)
          push!(R1_l,max(R1_l[end],C_l[i]))
        end
        append!(R_l,R1_l[end:-1:1] .- C_l[(length(R_l)+1):t_l])
      else # Line 11
        # Lines 12 and 13
        (C1_l,F1_l) = upRW(d.d_l,d.η_l,et1_l,d.κ_l,x_l-endC_l)
        append!(F_l,F1_l)
        append!(C_l,endC_l .+ C1_l)
        endC_l = C_l[end]
      end # Line 14
    end # Line 15
    while (length(R_r) == σ) || (length(S_r) >= length(C_r))
      t_r = length(C_r)-1
      # Lines 6 and 7
      (C1_r,F1_r) = downRW(d.d_r,rar_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
      append!(F_r,F1_r)
      append!(C_r,endC_r .+ C1_r)
      endC_r = C_r[end]
      # Lines 8 and 9
      if reflectedProcess(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
        # Line 10
        x_r = endC_r + d.κ_r
        R1_r = [maximum(C_r[t_r:end])]
        for i = (t_r-1):-1:(length(R_r)+1)
          push!(R1_r,max(R1_r[end],C_r[i]))
        end
        append!(R_r,R1_r[end:-1:1] .- C_r[(length(R_r)+1):t_r])
      else # Line 11
        # Lines 12 and 13
        (C1_r,F1_r) = upRW(d.d_r,d.η_r,et1_r,d.κ_r,x_r-endC_r)
        append!(F_r,F1_r)
        append!(C_r,endC_r .+ C1_r)
        endC_r = C_r[end]
      end # Line 14
    end # Line 15
    # Line 16
    while length(U_l) < length(S_l)
      push!(U_l,exp(-d.α*F_l[length(U_l)+1]))
    end
    while length(U_r) < length(S_r)
      push!(U_r,exp(-d.α*F_r[length(U_r)+1]))
    end
    # Lines 17, 18 and 19
    σ += 1
    D_l = exp(R_l[σ])*( e2_l*exp(-sep_l*(length(S_l)-σ-1)) + sum(S_l[σ:end] .*
        exp.(-d.d_l .* (0:(length(S_l)-σ))) .* (1 .- U_l[σ:end]) .^ ra) )
    D_r = exp(R_r[σ])*( e2_r*exp(-sep_r*(length(S_r)-σ-1)) + sum(S_r[σ:end] .*
        exp.(-d.d_r .* (0:(length(S_r)-σ))) .* (1 .- U_r[σ:end]) .^ ra) )
    push!(lagS_l,S_l[σ])
    push!(lagS_r,S_r[σ])
    push!(lagU_l,U_l[σ])
    push!(lagU_r,U_r[σ])
    # Line 24
    # Construct the dataframes
    (df_l,err_l) = faces(d.α,D_l,lagS_l,lagU_l)
    (df_r,err_r) = faces(d.α,D_r,lagS_r,lagU_r)

    # Rescale and add the drift
    df_l = @linq df_l |>
         where(:label .> -1) |>
         select(:length,:height,:slope,:label)
    df_l[:label] = 0
    sort!(df_l, order(:slope, rev = true))
    df_l[:length] = df_l[:length] .* T_l
    df_l[:height] = df_l[:height]  .* (-sc_l) .+ df_l[:length] .* d.μ
    df_l[:slope] = df_l[:slope] .* (-sc1_l) .+ d.μ
    df_r[:length] = df_r[:length] .* T_r
    df_r[:height] = df_r[:height] .* sc_r .+ df_r[:length] .* d.μ
    df_r[:slope] = df_r[:slope] .* sc1_r .+ d.μ
    df_l1 = deepcopy(df_l)
    df_r1 = deepcopy(df_r)

    # Construct (the dataframes with the faces of) the sandwiching convex functions
    dfL_l = @linq df_l1 |>
         select(:length,:height)
    dfU_l = copy(dfL_l)
    dfL_r = @linq df_r1 |>
        where(:label .< 1) |>
        select(:length,:height)
    dfU_r = @linq df_r1 |>
        where(:label .> -1) |>
        select(:length,:height)
    append!(dfL_l,dfL_r)
    append!(dfU_l,dfU_r)
    err_l *= sc_l

    # Check for crossing:
    aux = F(dfU_l)
    G = cross_func(d.f,-err_l)
    if aux == G(dfL_l) # Crossing / non-crossing detection!
      append!(df_l,df_r)
      return (aux,df_l,err_l,err_r * sc_r,σ)
    end # Line 26
  end # Line 27
end

export ConvexMinorantStableMeander, CrossConvexMinorantStableMeander, ConvexMinorantWeaklyStable, CrossConvexMinorantWeaklyStable, rand, params
