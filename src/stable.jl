using Distributions, FastGaussQuadrature

struct Stable <: ContinuousUnivariateDistribution
  α::Float64
  β::Float64
  θ::Float64
  ρ::Float64
  Stable(α::Real,β::Real) = ( β < -1. || β > 1. || α <= 0. || α > 2. ) ?
    error("Parameters' requirements unmet:\n (α,β)∈(0,2]×[-1,1]") :
    α == 2. ? new(2.,0.,0.,.5) :
    new(α,β,β*(α <= 1. ? 1. :
    (α-2.)/α), (1. + β*(α <= 1. ? 1. : (α-2.)/α))/2.)
end

import Distributions.params
function params(d::Stable)
  return (d.α,d.β,d.θ,d.ρ)
end

import Distributions.minimum
function minimum(d::Stable)
  return d.α <= 1. && d.β == 1. ? (d.α == 1. ? 1. : 0.) : -Inf
end

import Distributions.maximum
function maximum(d::Stable)
  return d.α <= 1. && d.β == -1. ? (d.α == 1. ? -1. : 0.) : Inf
end

import Distributions.insupport
function insupport(d::Stable,x::Real)
  return d.α <= 1. && abs(d.β) == 1. ? (d.α == 1. ? x == d.β : sign(x) != -d.β) : isfinite(x)
end

########################
# BASIC SIMULATION TOOLS
########################

import Distributions.rand
function rand(d::Stable) # Chambers. Mellows, and Stuck
  U = rand(Uniform(-pi/2,pi/2))
  return rand(Exponential())^(1-1/d.α) * sin(d.α * (U+(pi/2)*d.θ)) /
    (cos(U) ^ (1/d.α) * cos(U - d.α * (U+(pi/2)*d.θ)) ^ (1-1/d.α))
end

function rand(d::Stable,n::Integer) # Chambers. Mellows, and Stuck
  U = rand(Uniform(-pi/2,pi/2),n)
  return rand(Exponential(),n).^(1-1/d.α) .* sin.(d.α .* (U .+ (pi/2)*d.θ)) ./
    (cos.(U) .^ (1/d.α) .* cos.(U .- d.α .* (U .+ (pi/2)*d.θ)) .^ (1-1/d.α))
end
########################
# PDF, CDF
########################

function auxV2(x::AbstractArray,a::Real,th::Real)
  y = (pi/2).*x
  t = (pi/2)*a*th
  return ((sin.(a .* y .+ t).^a)./cos.(y)).^(1/(1-a)).*cos.((a-1) .* y .+ t)
end

import Distributions.pdf
function pdf(d::Stable,x::Real)
  return x==0 ? gamma(1+1/d.α)*sin(pi*d.ρ)/pi : (x>0 ? pdf(PositiveStable(d.α,d.β),x)*d.ρ : pdf(PositiveStable(d.α,-d.β),-x)*(1-d.ρ))
end

function pdf(d::Stable,X::AbstractArray{<:Real})
  return pdf(PositiveStable(d.α,d.β),X).*d.ρ + pdf(PositiveStable(d.α,-d.β),-X).*(1-d.ρ)
end

import Distributions.cdf
function cdf(d::Stable,x::Real)
  return 1 - d.ρ + cdf(PositiveStable(d.α,d.β),x)*d.ρ - cdf(PositiveStable(d.α,-d.β),-x)*(1-d.ρ)
end

function cdf(d::Stable,X::AbstractArray{<:Real})
  return 1 .- d.ρ .+ cdf(PositiveStable(d.α,d.β),X).*d.ρ .- cdf(PositiveStable(d.α,-d.β),-X).*(1-d.ρ)
end

import Distributions.cf
function cf(d::Stable,x::Real)
  if x==0
    return 1
  end
  if d.α==1 && abs(d.β)==1
    return exp(d.β*im*x)
  end
  return exp(-abs(x)^d.α*exp(-pi*d.θ*d.α*sign(x)*im/2))
end

function cf(d::Stable,X::AbstractArray{<:Real})
  if d.α==1 && abs(d.β)==1
    return exp((d.β .* im) .* X)
  end
  x = vec(X)
  s1 = exp(-pi*d.θ*d.α*im/2)
  s2 = exp(pi*d.θ*d.α*im/2)
  return reshape(exp.(-abs.(x) .^ d.α .* [x[i]>0 ? s1 : s2 for i=1:length(x)]),size(X))
end

import Distributions.mgf
function mgf(d::Stable,x::Real)
  if x==0
    return 1
  end
  if d.α==2
    return exp(x^2)
  elseif d.α==1 && abs(d.β)==1
      return exp(d.β*x)
  elseif x>0
    if d.β==-1 && d.α!=1
        return exp(sign(d.α-1)*x^d.α)
    else
      return Inf
    end
  elseif d.β==1 && d.α!=1
    return exp(sign(d.α-1)*(-x)^d.α)
  end
  return Inf
end

function mgf(d::Stable,X::AbstractArray{<:Real})
  if d.α==2
    return exp.(X .^ 2)
  elseif d.α==1 && abs(d.β)==1
    return exp.(d.β .* X)
  end
  x = vec(X)
  res = exp(sign(d.α-1).*abs(x).^d.α)
  if d.β==-1
    return reshape([x[i]>=0 ? res[i] : Inf for i=1:length(x)],size(X))
  elseif d.β==1
    return reshape([x[i]<=0 ? res[i] : Inf for i=1:length(x)],size(X))
  end
  return reshape([x[i]==0 ? 1 : Inf for i=1:length(x)],size(X))
end

import Distributions.mean
function mean(d::Stable)
  if d.α<=1
    return Inf
  end
  cr=1-d.ρ
  return (sin(pi*d.ρ)-sin(pi*cr))/(d.α*sin(pi/d.α)*gamma(1+1/d.α))
end

import Distributions.var
function var(d::Stable)
  if d.α == 1. && abs(d.β) == 1.
      return 0.
  else
    return Inf
  end
end

function mellin(d::Stable,x::Complex)
  if (real(x) >= d.α && (d.α <= 1. || (d.α < 2. && d.β != -1.))) || real(x) <= -1.
    return Inf
  end
  if (d.α > 1. && d.β == -1.) || d.α == 2.
    return d.ρ * gamma(1. + x) / gamma(1. + x / d.α)
  end
  return d.ρ * (sin(pi * d.ρ * x) * gamma(1. + x)) /
    (d.α * d.ρ * sin(pi * x / d.α) * gamma(1. + x / d.α))
end

function mellin(d::Stable,X::AbstractArray{<:Complex})
  if (d.α > 1. && d.β == -1.) || d.α == 2.
    return d.ρ .* gamma.(1. .+ X) ./ gamma.(1. .+ X ./ d.α)
  end

  res = Complex{Float64}[]
  for x = X
    push!(res,
      (real(x) >= d.α && (d.α <= 1. || (d.α < 2. && d.β != -1.))) || real(x) <= -1. ?
      Inf : (sin(pi * d.ρ * x) * gamma(1. + x)) / (d.α * d.ρ * sin(pi * x / d.α) * gamma(1. + x / d.α))
    )
  end
  return reshape(d.ρ .* res,size(X))
end

function mellin(d::Stable,x::Real)
  if (real(x) >= d.α && (d.α <= 1. || (d.α < 2. && d.β != -1.))) || real(x) <= -1.
    return Inf
  end
  if (d.α > 1. && d.β == -1.) || d.α == 2.
    return d.ρ * gamma(1. + x) / gamma(1. + x / d.α)
  end
  return d.ρ * (sin(pi * d.ρ * x) * gamma(1. + x)) /
  (d.α * d.ρ * sin(pi * x / d.α) * gamma(1. + x / d.α))
end

function mellin(d::Stable,X::AbstractArray{<:Real})
  if (d.α > 1. && d.β == -1.) || d.α == 2.
    return d.ρ .* gamma.(1. .+ X) ./ gamma.(1. .+ X ./ d.α)
  end

  res = Float64[]
  for x = X
    push!(res,
      (real(x) >= d.α && (d.α <= 1. || (d.α < 2. && d.β != -1.))) || real(x) <= -1. ?
      Inf : (sin(pi * d.ρ * x) * gamma(1. + x)) / (d.α * d.ρ * sin(pi * x / d.α) * gamma(1. + x / d.α))
    )
  end
  return reshape(d.ρ .* res,size(X))
end

export Stable, rand, minimum, maximum, insupport, pdf, cdf, cf, mgf, mean, mellin, params
