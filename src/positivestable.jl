using Distributions, SpecialFunctions, FastGaussQuadrature

struct PositiveStable <: ContinuousUnivariateDistribution
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  PositiveStable(α::Real,β::Real) = ( β < -1 || β > 1 ||
    (β == -1 && α <= 1) || α <= 0 || α > 2 ) ?
    error("Parameters' requirements unmet:\n (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}") :
    α == 2 ? new(2,0,0,.5) :
    new(α,β,β*(α <= 1 ? 1 :
    (α-2)/α), (1 + β*(α <= 1 ? 1 : (α-2)/α))/2)
end

import Distributions.params
function params(d::PositiveStable)
  return (d.α,d.β,d.θ,d.ρ)
end

import Distributions.minimum
function minimum(d::PositiveStable)
  return β == 1 && α == 1 ? 1 : 0
end

import Distributions.maximum
function maximum(d::PositiveStable)
  return β == 1 && α == 1 ? 1 : Inf
end

import Distributions.insupport
function insupport(d::PositiveStable,x::Real)
  return β == 1 && α == 1 ? x==1 : x >= 0
end

########################
# BASIC SIMULATION TOOLS
########################

import Distributions.rand

import Distributions.rand
function rand(d::PositiveStable) # Chambers. Mellows, and Stuck
  U = rand(Uniform(-(pi/2)*d.θ,pi/2))
  return rand(Exponential())^(1-1/d.α) * sin(d.α * (U+(pi/2)*d.θ)) /
    (cos(U) ^ (1/d.α) * cos(U - d.α * (U+(pi/2)*d.θ)) ^ (1-1/d.α))
end

function rand(d::PositiveStable,n::Integer) # Chambers. Mellows, and Stuck
  U = rand(Uniform(-(pi/2)*d.θ,pi/2),n)
  return rand(Exponential(),n).^(1-1/d.α) .* sin.(d.α .* (U .+ (pi/2)*d.θ)) ./
    (cos.(U) .^ (1/d.α) .* cos.(U .- d.α .* (U .+ (pi/2)*d.θ)) .^ (1-1/d.α))
end

function cond_rand(d::PositiveStable,lo::Real,hi::Real)
  if hi == Inf
    while true
      x = rand(d)
      if lo < x
        return x
      end
    end
  elseif lo == 0
    while true
      x = rand(d)
      if x < hi
        return x
      end
    end
  end
  mid = d.α <= 1 ? (d.β <= 0 ? 0 : (1+d.α)/2) : (d.β >= 0 ? 0 : (1.6+d.α)/2)
  p = pdf(d,[lo,hi])
  q = cdf(d,[lo,hi])
  if lo > mid && (p[2] / p[1] > q[1] - q[2]) # More likely to be accepted with this methodology
    ux = Uniform(lo,hi)
    u = Uniform()
    while true
      x = rand(ux)
      px = pdf(d,x)
      if rand(u) < px / p[1]
        return x
      end
    end
  else
    while true
      x = rand(d)
      if lo < x < hi
        return x
      end
    end
  end
end

########################
# PDF, CDF
########################

function auxV2(x::AbstractArray,a::Real,b::Real)
  y = (pi/2).*x
  t = (pi/2)*a*b
  return ((sin.(a .* y .+ t).^a) ./ cos.(y)) .^ (1 /(1-a)) .* cos.((a-1) .* y .+ t)
end

import Distributions.pdf
function pdf(d::PositiveStable,x::Real)
  if d.α == 1 && d.β == 1
    return x == 1 ? 1. : 0.
  end
  m = Int(1e4)

  pir = pi*d.ρ

  if d.α > 1
    delta = .3
    if x > 0
      if x < delta
        l = Int(1e2)
        v = 1:Int(floor(l*d.α))
        w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v ./ d.α .+ 1) .* sin.(pir .* v) ./
          ((d.ρ*pi) .* gamma.(v .+ 1))
        return sum(w .* abs(x) .^ (v .- 1))
      else
        a0 = d.α/((d.α-1)*2)
        raa = 1 /(d.α-1)
        a1 = abs(x) ^ raa
        a2 = - a1 ^ d.α
        a1 /= d.ρ
        nodes, weights = gausslegendre(m)
        s1 = d.ρ
        s2 = 1 - d.ρ
        seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
        imin = 1
        for k=1:length(seq1)
          if isfinite(seq1[k])
            imin = k
            break
          end
        end
        weights = (a0*s1) .* weights
        return a1*(sum(seq1[imin:end] .* exp.(a2 .* seq1[imin:end]) .* weights[imin:end]))
      end
    elseif x < 0
      return 0.
    else
      return gamma(1 + 1 / d.α)*sin(pir)/pir
    end
  elseif d.α < 1
    delta = 1.
    if x > delta
      l = Int(1.5e2)
      v = 1:Int(floor(l*d.α))
      w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v .* d.α .+ 1) .*
        sin.((pir*d.α) .* v) ./ (gamma.(v .+ 1) .* pir)
      return sum(w .* x .^ (-d.α .* v .- 1))
    elseif x > 0
      a0 = d.α/((d.α-1)*2)
      raa = 1 /(d.α-1)
      a1 = abs(x) ^ raa
      a2 = - a1 ^ d.α
      a1 /= -d.ρ
      nodes, weights = gausslegendre(m)
      s1 = d.ρ
      s2 = 1 - d.ρ
      seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
      imin = 1
      for k=1:length(seq1)
        if isfinite(seq1[k])
          imin = k
          break
        end
      end
      weights = (a0*s1) .* weights
      return a1*(sum(seq1[imin:end] .* exp.(a2 .* seq1[imin:end]) .* weights[imin:end]))
    elseif x < 0
      return 0.
    else
      return gamma(1 + 1 / d.α)*sin(pir)/pir
    end
  else #d.α = 1
    return x >= 0 ? pdf(Cauchy(-cos(pir),sin(pir)),x)/d.ρ : 0.
  end
end

function pdf(d::PositiveStable,X::AbstractArray{<:Real})
  if d.α == 1 && d.β == 1
    return convert(Array{Float64},X .== 1)
  end
  m = Int(1e4)
  x = vec(X)
  res = Float64[]

  pir = pi*d.ρ
  pir1 = pi*(1 - d.ρ)
  aux = gamma(1 + 1 / d.α)*sin(pir)/pir
  if d.α > 1
    l = Int(1e2)
    v = 1:Int(floor(l*d.α))
    delta = .3

    w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v ./ d.α .+ 1) .* sin.(pir .* v) ./
      ((d.ρ*pi) .* gamma.(v .+ 1))
    w1 = (-1) .^ ((v .- 1) .% 2) .* gamma.(v ./ d.α .+ 1) .* sin.(pir1 .* v) ./
      ((d.ρ*pi) .* gamma.(v .+ 1))

    a0 = d.α/((d.α-1)*2)
    raa = 1 /(d.α-1)
    a1 = abs.(x) .^ raa
    a2 = - a1 .^ d.α
    a1 ./= d.ρ
    nodes, weights = gausslegendre(m)
    s1 = d.ρ
    s2 = 1 - d.ρ
    seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
    imin = 1
    for k=1:length(seq1)
      if isfinite(seq1[k])
        imin = k
        break
      end
    end
    weights = (a0*s1) .* weights
    for i = 1:length(x)
      if x[i] > 0
        if x[i] < delta
          push!(res,sum(w .* abs(x[i]) .^ (v .- 1)))
        else
          push!(res,a1[i]*(sum(seq1[imin:end] .* exp.(a2[i] .* seq1[imin:end]) .* weights[imin:end])))
        end
      elseif x[i] < 0
        push!(res,0.)
      else
        push!(res,aux)
      end
    end
  elseif d.α<1
    l = Int(1.5e2)
    v = 1:Int(floor(l*d.α))
    delta = (1+d.α)/2
    v = 1:Int(floor(l*d.α))
    w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v .* d.α .+ 1) .*
      sin.((pir*d.α) .* v) ./ (gamma.(v .+ 1) .* pir)

    a0 = d.α/((d.α-1)*2)
    raa = 1 /(d.α-1)
    a1 = abs.(x) .^ raa
    a2 = - a1 .^ d.α
    a1 ./= -d.ρ
    nodes, weights = gausslegendre(m)
    s1 = d.ρ
    s2 = 1 - d.ρ
    seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
    a1 .*= a0*s1
    imin = 1
    for k=1:length(seq1)
      if isfinite(seq1[k])
        imin = k
        break
      end
    end

    for i = 1:length(x)
      if x[i] > delta
        push!(res,sum(w .* x[i] .^ (-d.α .* v .- 1)))
      elseif x[i] > 0
        push!(res,a1[i]*(sum(seq1[imin:end] .* exp.(a2[i] .* seq1[imin:end]) .* weights[imin:end])))
      elseif x[i] < 0
        push!(res,0.)
      else
        push!(res,aux)
      end
    end
  else #d.α = 1
    Cd = Cauchy(-cos(pir),sin(pir))
    res = [x[i] >= 0 ? pdf(Cd,x[i])/d.ρ : 0 for i=1:length(x)]
  end
  return reshape(res,size(X))
end

import Distributions.cdf
function cdf(d::PositiveStable,x::Real)
  if d.α == 1 && d.β == 1
    return x <= 1 ? 1. : 0.
  end
  pir = pi*d.ρ

  if d.α > 1
    delta = .3
    if x > 0
      if x < delta
        l = Int(1e2)
        v = 1:Int(floor(l*d.α))
        w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v ./ d.α .+ 1) .*
          sin.(pir .* v) ./ ((pi*d.ρ) .* v .* gamma.(v .+ 1))
        return sum(w .* (x .^ v))
      else
        m = Int(1e4)
        raa = d.α/(d.α-1)
        a2 = -abs(x) ^ raa
        nodes, weights = gausslegendre(m)
        s1 = d.ρ
        s2 = 1 - d.ρ
        seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
        s1 = s1/(2 * d.ρ)
        return 1 - s1*sum(exp.(a2 .* seq1) .* weights)
      end
    else
      return 0.
    end
  elseif d.α < 1
    delta = (1+d.α)/2
    for i = 1:length(x)
      if x > delta
        l = Int(1.5e2)
        v = 1:Int(floor(l*d.α))
        w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v .* d.α .+ 1) .*
          sin.((pir*d.α) .* v) ./ (v .* gamma.(v .+ 1))
        pira = d.α*pir
        return 1 - sum(w .* x .^ (-d.α .* v))/pira
      elseif x > 0
        m = Int(1e4)
        raa = d.α /(d.α-1)
        a2 = -abs(x) ^ raa
        a0 = 1/2
        nodes, weights = gausslegendre(m)
        s1 = d.ρ
        s2 = 1 - d.ρ
        seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
        imin = 1
        for k=1:length(seq1)
          if isfinite(seq1[k])
            imin = k
            break
          end
        end
        weights = (a0*s1) .* weights ./ d.ρ
        return sum(exp.(a2 .* seq1[imin:end]) .* weights[imin:end])
      else
        return 0.
      end
    end
  else #d.α = 1
    return x >= 0 ? (cdf(Cauchy(-cos(pir),sin(pir)),x)-1)/d.ρ+1 : 0.
  end
end

function cdf(d::PositiveStable,X::AbstractArray{<:Real})
  if d.α == 1 && d.β == 1
    return convert(Array{Float64},X .<= 1)
  end
  m = Int(1e4)
  x = vec(X)
  res = Float64[]
  pir = pi*d.ρ

  if d.α > 1
    delta = .3
    l = Int(1e2)
    v = 1:Int(floor(l*d.α))
    w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v ./ d.α .+ 1) .*
      sin.(pir .* v) ./ ((pi*d.ρ) .* v .* gamma.(v .+ 1))
    raa = d.α/(d.α-1)
    a2 = -abs.(x) .^ raa
    nodes, weights = gausslegendre(m)
    s1 = d.ρ
    s2 = 1 - d.ρ
    seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
    s1 = s1/(2 * d.ρ)
    for i = 1:length(x)
      if x[i] > 0
        if x[i] < delta
          push!(res,sum(w .* (x[i] .^ v)))
        else
          push!(res,1 - s1*(sum(exp.(a2[i] .* seq1) .* weights)))
        end
      else
        push!(res,0.)
      end
    end
  elseif d.α < 1
    delta = (1+d.α)/2
    l = Int(1.5e2)
    v = 1:Int(floor(l*d.α))
    w = (-1) .^ ((v .- 1) .% 2) .* gamma.(v .* d.α .+ 1) .*
      sin.((pir*d.α) .* v) ./ (v .* gamma.(v .+ 1))
    pira = d.α*pir

    raa = d.α /(d.α-1)
    a2 = -abs.(x) .^ raa
    a0 = 1/2
    nodes, weights = gausslegendre(m)
    s1 = d.ρ
    s2 = 1 - d.ρ
    seq1 = auxV2(nodes .* s1 .+ s2,d.α,d.θ)
    imin = 1
    for k=1:length(seq1)
      if isfinite(seq1[k])
        imin = k
        break
      end
    end
    weights = (a0*s1) .* weights ./ d.ρ

    for i = 1:length(x)
      if x[i] > delta
        push!(res, 1 - sum(w .* x[i] .^ (-d.α .* v))/pira)
      elseif x[i] > 0
        push!(res,sum(exp.(a2[i] .* seq1[imin:end]) .* weights[imin:end]))
      else
        push!(res,0.)
      end
    end
  else #d.α = 1
    Cd = Cauchy(-cos(pir),sin(pir))
    res = [x[i] >= 0 ? (cdf(Cd,x[i])-1)/d.ρ + 1 : 0 for i=1:length(x)]
  end
  return reshape(res,size(X))
end

import Distributions.mgf
function mgf(d::PositiveStable,x::Real)
  l = 12
  if x == 0
    return 1.
  end
  if d.α == 2
    return 2 * exp(x^2/4.)*cdf(Normal(0,sqrt(2)),x/2)
  end
  if d.β == -1 && x >= -1
    v = 0:Int(floor(l*d.α))
    w = 1 / gamma(v+1)
    return sum(w .* x .^ v)
  end

  nodes, weights = gausslegendre(m)
  nodes = nodes ./ 2 .+ .5
  weights = weights/(2 *d.ρ)
  nodes2 = 1 ./ (1 .- nodes) .^ 2
  nodes = nodes ./ (1 .- nodes)
  pir = pi*d.ρ
  fC = nodes2 .* pdf(Cauchy(-cos(pir),sin(pir)),nodes)
  if d.β == -1
    mat = exp(-(abs(x) .* nodes) .^ d.α)
    return sum(fC.*mat .* weights)
  else
    if x > 0
      return Inf
    else
      mat = exp(-(abs(x) .* nodes) .^ d.α)
      return sum(fC.*mat .* weights)
    end
  end
end

function mgf(d::PositiveStable,X::AbstractArray{<:Real})
  x = vec(X)
  l = 12
  if d.α==2
    return 2 .* exp.(x.^2 ./ 4.) .* cdf(Normal(0,sqrt(2)),x ./ 2)
  end
  nodes, weights = gausslegendre(m)
  nodes = nodes ./ 2 .+ .5
  weights = weights ./ (2 * d.ρ)
  nodes2 = 1 ./ (1 .- nodes) .^ 2
  nodes = nodes ./ (1 .- nodes)
  pir = pi*d.ρ
  fC = nodes2 .* pdf(Cauchy(-cos(pir),sin(pir)),nodes)
  res = Float64[]
    mat = exp.(-(abs.(x)*transpose(nodes)) .^ d.α)
  if d.β == -1
    v = 0:Int(floor(18*d.α))
    w = 1 ./ gamma.(v .+ 1)
    if x[i] == 0
      push!(res,1.)
    elseif x[i] >= -1
      push!(res,sum(w .* x[i] .^ v))
    else
      push!(res,sum(fC .* mat[i,:] .* weights))
    end
  else
    for i = 1:length(x)
      if x[i] > 0
        push!(res,Inf)
      elseif x[i] < 0
        push!(res,sum(fC .* mat[i,:] .* weights))
      else
        push!(res,1.)
      end
    end
  end
  return reshape(res,size(X))
end

import Distributions.mean
function mean(d::PositiveStable)
  if d.α <= 1
    return Inf
  end
  return sin(pi*d.ρ)/(d.α*d.ρ*sin(pi/d.α)*gamma(1 + 1 / d.α))
end

import Distributions.var
function var(d::PositiveStable)
  if d.α < 2 && d.β != -1
    return Inf
  elseif d.α == 1
    if abs(d.β) != 1
      return Inf
    else
      return 0.
    end
  else
    return 2 / gamma(1 + 2 / d.α) - 1 / gamma(1 + 1 / d.α) ^ 2
  end
end

function mellin(d::PositiveStable,x::Complex)
  if (real(x) >= d.α && (d.α <= 1 || (d.α < 2 && d.β != -1))) || real(x) <= -1
    return Inf
  end
  if (d.α > 1 && d.β == -1) || d.α == 2
    return gamma(1 + x) / gamma(1 + x / d.α)
  end
  return (sin(pi * d.ρ * x) * gamma(1 + x)) /
    (d.α * d.ρ * sin(pi * x / d.α) * gamma(1 + x / d.α))
end

function mellin(d::PositiveStable,X::AbstractArray{<:Complex})
  if (d.α > 1 && d.β == -1) || d.α == 2
    return gamma.(1 .+ X) ./ gamma.(1 .+ X ./ d.α)
  end

  res = Complex{Float64}[]
  for x = X
    push!(res,
      (real(x) >= d.α && (d.α <= 1 || (d.α < 2 && d.β != -1))) || real(x) <= -1 ?
      Inf : (sin(pi * d.ρ * x) * gamma(1 + x)) / (d.α * d.ρ * sin(pi * x / d.α) * gamma(1 + x / d.α))
    )
  end
  return reshape(res,size(X))
end

function mellin(d::PositiveStable,x::Real)
  if (real(x) >= d.α && (d.α <= 1 || (d.α < 2 && d.β != -1))) || real(x) <= -1
    return Inf
  end
  if (d.α > 1 && d.β == -1) || d.α == 2
    return gamma(1 + x) / gamma(1 + x / d.α)
  end
  return (sin(pi * d.ρ * x) * gamma(1 + x)) /
  (d.α * d.ρ * sin(pi * x / d.α) * gamma(1 + x / d.α))
end

function mellin(d::PositiveStable,X::AbstractArray{<:Real})
  if (d.α > 1 && d.β == -1) || d.α == 2
    return gamma.(1 .+ X) ./ gamma.(1 .+ X ./ d.α)
  end

  res = Float64[]
  for x = X
    push!(res,
      (real(x) >= d.α && (d.α <= 1 || (d.α < 2 && d.β != -1))) || real(x) <= -1 ?
      Inf : (sin(pi * d.ρ * x) * gamma(1 + x)) / (d.α * d.ρ * sin(pi * x / d.α) * gamma(1 + x / d.α))
    )
  end
  return reshape(res,size(X))
end

export Stable, rand, minimum, maximum, insupport, pdf, cdf, mgf, mean, mellin, params
