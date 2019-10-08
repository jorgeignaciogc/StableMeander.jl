# StableMeander.jl

A Julia package for ε-strong simulation (εSS) of the stable meanders and related distributions. It supports a few methods and an auxiliary distribution (see below for details). Specifically, this package includes the following distributions (using Zolotarev's (C) form of parametrization):
<ul>
<li>Stable - Stable random variable with parameters (α,β)∈(0,2]×[-1,1].</li>
<li>PositiveStable - Stable random variable conditioned to be positive with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}.</li>
<li>StableMeander - The marginal at time 1 of a normalised stable meander on [0,1] with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1}.</li>
<li>MvStableMeander - The finite dimensional distribution of a normalised normalised stable meander on [0,t[m]] at times t[1], ..., t[m] with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1} and 0 < t[1] < ... < t[m].</li>
</ul>

## Table of Contents

1. [Schema](#schema) 
2. [Remarks and References](#references)
3. [Examples](#examples)
4. [Author and Contributor List](#authors)

<a name="schema"/>

## Schema

The distributions included support most of the standard functions as outlined in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

### Stable - _Type_

```julia
Stable <: ContinuousUnivariateDistribution
```
This type has a single standard constructor `Stable(α::Real,β::Real)` with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1} and supports the methods `minimum`, `maximum`, `insupport`, `pdf`, `cdf`, `cf`, `mgf`, `mean`, `var`, `mellin`, `params` and `rand`.

#### Remarks

* Method `params(d::Stable)` returns the tuple (α,β,θ,ρ) following Zolotarev's (C) form (i.e.,
`ρ=1-cdf(d,0)` and `θ=2*ρ-1`).
* Method `mellin(d::PositiveStable,X::T)` returns the [Mellin transform](https://en.wikipedia.org/wiki/Mellin_transform), where `T` is either `F` or `AbstractArray{F}` and `F` is either`Real` or `Complex`.
* Method `rand(d::Stable)` is based on [Chambers-Mellows-Stuck algorithm](https://en.wikipedia.org/wiki/Stable_distribution#Simulation_of_stable_variables).


### PositiveStable - _Type_

```julia
PositiveStable <: ContinuousUnivariateDistribution
```
This type has a single standard constructor `PositiveStable(α::Real,β::Real)` with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1} and supports the methods `minimum`, `maximum`, `insupport`, `pdf`, `cdf`, `cf`, `mgf`, `mean`, `var`, `mellin`, `params` and `rand`.

#### Remarks

* Method `params(d::PositiveStable)` returns the tuple (α,β,θ,ρ) following Zolotarev's (C) form.
* Method `mellin(d::PositiveStable,X::T)` returns the [Mellin transform](https://en.wikipedia.org/wiki/Mellin_transform), where `T` is either `F` or `AbstractArray{F}` and where `F` is either`Real` or `Complex`.


### StableMeander - _Type_

```julia
StableMeander <: ContinuousUnivariateDistribution
```
This type has a single standard constructor `StableMeander(α::Real,β::Real)` with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1} and supports the methods `minimum`, `maximum`, `insupport`, `mean`, `params`, `rand` and two samplers `precise_sampler` and `local_sampler`.

#### Remarks

* Method `params(d::StableMeander)` returns the tuple (α,β,θ,ρ) following Zolotarev's (C) form.
* If β=1 and α<1, constructor automatically defaults to `Stable(α,1)` since they agree.
* `precise_sampler(d::StableMeander)` returns a subtype [Sampler](https://juliastats.github.io/Distributions.jl/stable/extends.html) of sub type `PreciseStableMeander`. The optional arguments in `precise_sampler(d::StableMeander,args...)` are as in the constructor below of `PreciseStableMeander` below.
* `local_sampler(d::StableMeander,...)` returns a subtype [Sampler](https://juliastats.github.io/Distributions.jl/stable/extends.html) of sub type `LocStableMeander`. The optional arguments in `local_sampler(d::StableMeander,args...)` are as in the constructor below of `LocStableMeander` below.
* `rand(d::StableMeander)` calls `rand(precise_sampler(d))[1]`.


### PreciseStableMeander - _Type_

```
PreciseStableMeander <: Sampleable{Multivariate,Continuous}
```
This is a sub type for arbitrarily precise samples of `StableMeander` through _ε-strong simulation_ (εSS), which has multiple hyperparameters (see the [references](#references) for more details and the conditions they must satisfy). The parameter `ε>0` specifies the _almost sure_ precision of the sample. 

The output of `rand` is a tuple `(x,x+err,s)`, where `0<err<ε`, the true random variable `y` lies in the interval `[x,x+err]` and `s` is the number of steps that the internal process ran for (beyond the user-defined warm up period `Δ`). This sub type supports `params`, `length` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,mAst::Integer,ε::Real)`,
* `PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,mAst::Integer)`,
* `PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer)`,
* `PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real)`,
* `PreciseStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real)`,
* `PreciseStableMeander(α::Real,β::Real,Δ::Integer,ε::Real)`,
* `PreciseStableMeander(α::Real,β::Real,ε::Real)`,
* `PreciseStableMeander(α::Real,β::Real)`.


### LocStableMeander - _Type_

```
LocStableMeander <: Sampleable{Multivariate,Continuous}
```
This is a sub type for sampling `f(X)` where `X` has distribution `StableMeander` (eploiting εSS), which has multiple hyperparameters (see the [references](#references) for more details and the conditions they must satisfy). The parameter `f::Function` must be discrete and satisfy: if `x<y` and `f(x)=f(y)`, then `f(z)=f(x)` for any `z∈(x,y)`. (An example of such a function is `f(x)=floor(x)`.) 

The output of `rand` is a tuple `(f(x),x,x+err,s)`, where the true random variable `y` satisfies `f(y)=f(x)` and `s` is the number of steps that the internal process ran for (beyond the user-defined warm up period `Δ`). This sub type supports `params`, `length` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,mAst::Integer,f::Function)`,
* `LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,f::Function)`,
* `LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,f::Function)`,
* `LocStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,f::Function)`,
* `LocStableMeander(α::Real,β::Real,Δ::Integer,f::Function)`,
* `LocStableMeander(α::Real,β::Real,f::Function)`.


### MvStableMeander - _Type_

```julia
MvStableMeander <: ContinuousMultivariateDistribution
```
This type has a single standard constructor `MvStableMeander(α::Real,β::Real,t::Array{Real,1})` with parameters (α,β)∈(0,2]×[-1,1]-(0,1]×{-1} and `0<t[1]<...<t[m]`, where `m=length(t)`. This type supports the methods `insupport`, `params`, `rand` and two samplers `precise_sampler` and `local_sampler`.

#### Remarks

* Method `params(d::MvStableMeander)` returns the tuple (α,β,θ,ρ,t) following Zolotarev's (C) form.
* `precise_sampler(d::StableMeander)` returns a subtype [Sampler](https://juliastats.github.io/Distributions.jl/stable/extends.html) of sub type `PreciseMvStableMeander`. The optional arguments in `precise_sampler(d::MvStableMeander,args...)` are as in the constructor below of `PreciseMvStableMeander` below.
* `local_sampler(d::StableMeander,...)` returns a subtype [Sampler](https://juliastats.github.io/Distributions.jl/stable/extends.html) of sub type `LocMvStableMeander`. The optional arguments in `local_sampler(d::MvStableMeander,args...)` are as in the constructor below of `LocMvStableMeander` below.
* `rand(d::MvStableMeander)` calls `rand(precise_sampler(d))[1]`.


### PreciseMvStableMeander - _Type_

```
PreciseMvStableMeander <: Sampleable{Multivariate,Continuous}
```
This is an auxiliary sub type for arbitrarily precise samples of `MvStableMeander` through _ε-strong simulation_ (εSS), which has multiple hyperparameters (see the [references](#references) for more details and the conditions they must satisfy). The parameter `ε>0` specifies the _almost sure_ precision of the sample (in the maximum norm). 

The output of `rand` is a tuple `(x,x .+ err,s)`, where `0 .< err`, `sum(err)<ε`, the true random vector `y` satisfies `x .< y .< x .+ err` and `s` is the number of steps that the internal process ran for (beyond the user-defined warm up periods `Δ_l` and `Δ_r`). This sub type supports `params`, `length` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,mAst_l::Integer,mAst_r::Integer,ε::Real)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,mAst_l::Integer,mAst_r::Integer)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},Δ_l::Integer,Δ_r::Integer,ε::Real)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1},ε::Real)`,
* `PreciseMvStableMeander(α::Real,β::Real,t::Array{Real,1})`.


### LocMvStableMeander - _Type_

```
LocMvStableMeander <: Sampleable{Multivariate,Continuous}
```
This is a sub type for sampling `f(X)` where `X` has distribution `MvStableMeander` (eploiting εSS), which has multiple hyperparameters (see the [references](#references) for more details and the conditions they must satisfy). The parameter `f::Function` must be discrete and satisfy: if `x .<= y` and `f(x)=f(y)`, then `f(z)=f(x)` for any `z` satisfying `x .<= z .<= y`. (An example of such a function is `f(x)=floor.(cumsum(x))`.) 

The output of `rand` is a tuple `(f(x),x,x .+ err,s)`, where the true random vector `y` satisfies `f(y)=f(x)` and `s` is the number of steps that the internal process ran for (beyond the user-defined warm up periods `Δ_l` and `Δ_r`). This sub type supports `params`, `length` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,mAst_l::Integer,mAst_r::Integer,f::Function)`,
* `LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,f::Function)`,
* `LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,f::Function)`,
* `LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,f::Function)`,
* `LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},Δ_l::Integer,Δ_r::Integer,f::Function)`,
* `LocMvStableMeander(α::Real,β::Real,t::Array{Real,1},f::Function)`.


### ConvexMinorantStableMeander - _Type_

```julia
ConvexMinorantStableMeander <: ContinuousMultivariateDistribution
```
This type represents the inifinite-dimensional random element `C` defined as the largest convex function (the _convex minorant_) of a normalised stable meander over the interval [0,1]. It is a piece-wise linear function with infinitely many line-segments. Its simulation can only be _arbitrarily precise_ by virtue of its infinite-dimensionality. Therefore a sample is obtained from _ε-strong simulation_ (εSS), which has multiple hyperparameters (see the [references](#references) for more details and the conditions they must satisfy). The parameter `ε>0` specifies the _almost sure_ precision of the sample (in the supremum norm). 

The output of `rand` is a tuple `(df,err,s)` satisfying the following properties. (I) `df_l=@linq df |> where(:label .< 1) |> select(:length,:height)` and `df_u = @linq df |> where(:label .> -1) |> select(:length,:height)` are dataframes with the _lengths_ and _heights_ of the (sorted) line segments that comprise the piece-wise linear functions with vertices `C_d(sum(df_d[:length][1:i])) = sum(df_d[:height][1:i])` and `C_u(sum(df_u[:length][1:i])) = sum(df_u[:height][1:i])`, respectively, satisfying `C_d(x) <= C(x) <= C_u(x) <= C_d(x)+err` for every `0<x<1`, where `C` is the convex minorant of the respective stable meander. (II) The random error `err` satisfies `0<err<ε`. (III) `s` is the number of steps that the internal process ran for (beyond the user-defined warm up period `Δ`). This type supports `rand` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,mAst::Integer,ε::Real)`,
* `ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,mAst::Integer)`,
* `ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer)`,
* `ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real)`,
* `ConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real)`,
* `ConvexMinorantStableMeander(α::Real,β::Real,Δ::Integer,ε::Real)`,
* `ConvexMinorantStableMeander(α::Real,β::Real,ε::Real)`,
* `ConvexMinorantStableMeander(α::Real,β::Real)`.


### CrossConvexMinorantStableMeander - _Type_

```julia
CrossConvexMinorantStableMeander <: ContinuousMultivariateDistribution
```
This type represents the inifinite-dimensional random element `C` defined as the largest convex function (the _convex minorant_) of a normalised stable meander over the interval [0,1]. It is a piece-wise linear function with infinitely many line-segments. By exploiting εSS, we may identify whether the convex minorant crosses some function `f::Function` (hence we rely on multiple hyperparameters, see the [references](#references) for more details and the conditions they must satisfy). The function `f` is assumed to be convex (thus, crossings need only be tested at the vertices of `C`). 

The output of `rand` is a tuple `(bool,df,err,s)` satisfying the following properties. (I) `bool` is a Boolean answering the question _did the true convex minorant `C` cross the function `f`?_. (II) `df_l=@linq df |> where(:label .< 1) |> select(:length,:height)` and `df_u = @linq df |> where(:label .> -1) |> select(:length,:height)` are dataframes with the _lengths_ and _heights_ of the (sorted) line segments that comprise the piece-wise linear functions with vertices `C_d(sum(df_d[:length][1:i])) = sum(df_d[:height][1:i])` and `C_u(sum(df_u[:length][1:i])) = sum(df_u[:height][1:i])`, respectively, satisfying `C_d(x) <= C(x) <= C_u(x) <= C_d(x)+err` for every `0<x<1`, where `C` is the convex minorant of the respective stable meander. (III) `s` is the number of steps that the internal process ran for (beyond the user-defined warm up period `Δ`). This type supports `rand` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,mAst::Integer,f::Function)`,
* `CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,Δ::Integer,f::Function)`,
* `CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,κ::Real,f::Function)`,
* `CrossConvexMinorantStableMeander(α::Real,β::Real,d::Real,δ::Real,γ::Real,f::Function)`,
* `CrossConvexMinorantStableMeander(α::Real,β::Real,Δ::Integer,f::Function)`,
* `CrossConvexMinorantStableMeander(α::Real,β::Real,f::Function)`.


### ConvexMinorantWeaklyStable - _Type_

```julia
ConvexMinorantWeaklyStable <: ContinuousMultivariateDistribution
```
This type represents the inifinite-dimensional random element `C` defined as the largest convex function (the _convex minorant_) of a weakly stable process with parameters `(α,β)`, scale `δ` and drift `μ` over the interval [0,T]. It is a piece-wise linear function with infinitely many line-segments. Its simulation can only be _arbitrarily precise_ by virtue of its infinite-dimensionality. Therefore a sample is obtained from _ε-strong simulation_ (εSS), which has multiple hyperparameters (see the [references](#references) for more details and the conditions they must satisfy). The parameter `ε>0` specifies the _almost sure_ precision of the sample (in the supremum norm). 

The output of `rand` is a tuple `(df,err_l,err_r,s)` satisfying the following properties. (I) `df_l=@linq df |> where(:label .< 1) |> select(:length,:height)` and `df_u = @linq df |> where(:label .> -1) |> select(:length,:height)` are dataframes with the _lengths_ and _heights_ of the (sorted) line segments that comprise the piece-wise linear functions `C_d(sum(df_d[:length][1:i])) = sum(df_d[:height][1:i])` and `C_u(sum(df_u[:length][1:i])) = sum(df_u[:height][1:i])`, respectively, satisfying `C_d(x) <= C(x) <= C_u(x) <= C_d(x)+err_l+err_r` for every `0<x<T`, where `C` is the convex minorant of the respective weakly stable process. (II) The random errors `err_l` and `err_r` satisfy `0<err_l+err_r<ε` and `C_d(x)+err_l=C_u(x)` on a subinterval `[0,r]` (in particular `C_d(0)+err_l=C_u(x)=C(x)=0`). (III)  `s` is the number of steps that the internal process ran for (beyond the user-defined warm up periods `Δ_l` and `Δ_r`). This type supports `rand` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,mAst_l::Integer,mAst_r::Integer,ε::Real)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,mAst_l::Integer,mAst_r::Integer)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,Δ_l::Integer,Δ_r::Integer,ε::Real)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,ε::Real)`,
* `ConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real)`.


### CrossConvexMinorantWeaklyStable - _Type_

```julia
CrossConvexMinorantWeaklyStable <: ContinuousMultivariateDistribution
```
This type represents the inifinite-dimensional random element `C` defined as the largest convex function (the _convex minorant_) of a weakly stable process over the interval [0,T]. It is a piece-wise linear function with infinitely many line-segments. By exploiting εSS, we may identify whether the convex minorant crosses some function `f::Function` (hence we rely on multiple hyperparameters, see the [references](#references) for more details and the conditions they must satisfy). The function `f` is assumed to be convex (thus, crossings need only be tested at the vertices of `C`). 

The output of `rand` is a tuple `(bool,df,err_l,err_r,s)` satisfying the following properties. (I) `bool` is a Boolean answering the question _did the true convex minorant `C` cross the function `f`?_. (II) `df_l=@linq df |> where(:label .< 1) |> select(:length,:height)` and `df_u = @linq df |> where(:label .> -1) |> select(:length,:height)` are dataframes with the _lengths_ and _heights_ of the (sorted) line segments that comprise the piece-wise linear functions with vertices `C_d(sum(df_d[:length][1:i])) = sum(df_d[:height][1:i])` and `C_u(sum(df_u[:length][1:i])) = sum(df_u[:height][1:i])`, respectively, satisfying `C_d(x) <= C(x) <= C_u(x) <= C_d(x)+err` for every `0<x<1`, where `C` is the convex minorant of the respective stable meander. (II) The random errors `err_l` and `err_r` satisfy `0<err_l+err_r<ε` and `C_d(x)+err_l=C_u(x)` on a subinterval `[0,r]` (in particular `C_d(0)+err_l=C_u(x)=C(x)=0`). (III)  `s` is the number of steps that the internal process ran for (beyond the user-defined warm up periods `Δ_l` and `Δ_r`). This type supports `rand` and has the following constructors (for every omitted parameter, the constructor uses a suggested value):

* `CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,mAst_l::Integer,mAst_r::Integer,f::Function)`,
* `CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,Δ_l::Integer,Δ_r::Integer,f::Function)`,
* `CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,κ_l::Real,κ_r::Real,f::Function)`,
* `CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,d_l::Real,d_r::Real,δ_l::Real,δ_r::Real,γ_l::Real,γ_r::Real,f::Function)`,
* `CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,Δ_l::Integer,Δ_r::Integer,f::Function)`,
* `CrossConvexMinorantWeaklyStable(α::Real,β::Real,δ::Real,μ::Real,T::Real,f::Function)`.


<a name="references"/>

## Remarks and References

### StableMeander, MvStableMeander, ConvexMinorantStableMeander, ConvexMinorantWeaklyStable
These distributions' implementations rely on a recent paper by the authors of the package. See the article for details at: 
Jorge González Cázares and Aleksandar Mijatović and Gerónimo Uribe Bravo, *ε-strong simulation of the convex minorants of stable processes and meanders*, [arXiv:...](https://arxiv.org/abs/...) (2019). In this reference, the variables `Δ` and `mAst` are denoted <a href="https://www.codecogs.com/eqnedit.php?latex=m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m" title="m" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=m^*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m^*" title="m^*" /></a>, respectively.
Throughout the paper, the authors work on the parameters (α,ρ) where ρ is the positivity parameter (i.e. the probability that `rand(Stable(α,β))` is positive) and can be computed from (α,β) (see Appendix A in the reference).


<a name="examples"/>

## Examples

### Example 1

The true value of `mean(StableMeander(α,β))` can be computed thanks to Corollary 10 in [arXiv:...](https://arxiv.org/abs/...) (2019). We will check this empirically by comparing it with the mean of ε-strong samples.

```julia
# Speed: real world running time
Random.seed!(2019)

d = StableMeander(1.5,0)
d2 = precise_sampler(d,35,.5^32)

# Number of samples:
N = Int(1e4)

@time [rand(d2) for i=1:N]

# Example 1. Test means

# Estimate and bootstrap confidence intervals
function estim(α,β,conf_level,N)
# Define distributions
d = PreciseStableMeander(α,β)
boot = DiscreteUniform(1,N)

Xd = zeros(0)
Xu = zeros(0)
for i = 1:N
aux = rand(d)
push!(Xd,aux[1])
push!(Xu,aux[2])
end

Mu = zeros(0)
Md = zeros(0)
for i = 1:N
I = Int.(rand(boot,N))
push!(Mu,mean([Xu[I[i]] for i=1:N]))
push!(Md,mean([Xd[I[i]] for i=1:N]))
end

mu = sort(Mu)[Int(ceil(N*(1+conf_level)/2))]
md = sort(Md)[Int(floor(N*(1-conf_level)/2))]

res = (mean(StableMeander(α,β)),(mean(Mu)+mean(Md))/2,mu,md)

println("(α,β) = (",α,",",β,")",
"\nReal = ",round(res[1],sigdigits=8),
"\nEstimate = ",round(res[2],sigdigits=8),
"\nEstimate up = ",round(mu,sigdigits=8),
"\nEstimate down = ",round(md,sigdigits=8))
return res
end

(α,β) = (convert(Array{Float64},1.1:.1:2),convert(Array{Float64},-1:1:1))
r_mean = zeros(Float64, length(α), length(β))
e_mean = zeros(Float64, length(α), length(β))
u_mean = zeros(Float64, length(α), length(β))
d_mean = zeros(Float64, length(α), length(β))

conf_level = .95
N = Int(1e4)

# Fix the random seed
Random.seed!(2019)
for j = 1:length(β)
for i = 1:length(α)
@time (r_mean[i,j],e_mean[i,j],u_mean[i,j],d_mean[i,j]) = estim(α[i],β[j],conf_level,N)
end
end

# Plot it
function plot_mean(j)
df = DataFrame()
# Labels
lab = [i <= length(α) ? "Real" : i <= 2*length(α) ? "Estimate" :
i <= 3*length(α) ? "CI (up)" : "CI (down)" for i = 1:Int(4*length(α))]
df[:Label] = lab
# x-axis grid
df[:x] = vcat(α,α,α,α)
# y-axis coordinates of the theoretical CDF
df[:y] = vcat(r_mean[:,j],e_mean[:,j],u_mean[:,j],d_mean[:,j])
# Plot comparing the empirical CDF and CDF
plot(df, x=:x, y=:y, color=:Label, Geom.line, Guide.xlabel("α"),
Guide.xticks(ticks=[1:.25:2;]), Guide.colorkey(title="Labels"),
Guide.Theme(panel_fill = nothing, major_label_color = "white",
key_title_color = "white", key_label_color = "white"),
Guide.title(string("Mean comparison for β = ",β[j])))
end

plot_mean(2)

```

### Example 2
The following code plots a random sample of a `ConvexMinorantStableMeander` and a `ConvexMinorantWeaklyStable`.

```julia

# Example 2. Plot sample of convex minorant
using Distributions, StatsBase, SpecialFunctions, Random, FastGaussQuadrature, Gadfly, DataFrames

function rand_plot(d::ConvexMinorantStableMeander)
(df,err) = rand(d)
dfL = @linq df |>
where(:label .< 1) |>
select(:length,:height)
x = [0.]
y = [0.]
append!(x,cumsum(dfL[:length]))
append!(y,cumsum(dfL[:height]))
n = length(x)
dfU = @linq df |>
where(:label .> -1) |>
select(:length,:height)
push!(x,0.)
push!(y,0.)
append!(x,cumsum(dfU[:length]))
append!(y,cumsum(dfU[:height]))

df0 = DataFrame()
df0[:x] = x
df0[:y] = y
df0[:Label] = [i <= n ? "Lower" : "Upper" for i = 1:(2*n)]
# Plot both
plot(df0, x=:x, y=:y, color=:Label, Geom.line, Guide.xlabel("t"),
Guide.xticks(ticks=[0:.25:1;]), Guide.colorkey(title="Estimate"),
Guide.Theme(panel_fill = nothing, major_label_color = "white",
key_title_color = "white", key_label_color = "white"),
Guide.title("Simulation of C(Z^me)"))
end

function rand_plot(d::ConvexMinorantWeaklyStable)
(df,err_l,err_r) = rand(d)
dfL = @linq df |>
where(:label .< 1) |>
select(:length,:height)
x = [0.]
y = [0.]
append!(x,cumsum(dfL[:length]))
append!(y,cumsum(dfL[:height]))

n = length(x)
dfU = @linq df |>
where(:label .> -1) |>
select(:length,:height)
push!(x,0.)
push!(y,-err_l)
append!(x,cumsum(dfU[:length]))
append!(y,-err_l .+ cumsum(dfU[:height]))

df0 = DataFrame()
df0[:x] = x
df0[:y] = y
df0[:Label] = [i <= n ? "Lower" : "Upper" for i = 1:(2*n)]
# Plot both
plot(df0, x=:x, y=:y, color=:Label, Geom.line, Guide.xlabel("t"),
Guide.xticks(ticks=[0:.25:1;]), Guide.colorkey(title="Estimate"),
Guide.Theme(panel_fill = nothing, major_label_color = "white",
key_title_color = "white", key_label_color = "white"),
Guide.title("Simulation of C(Z)"))
end

Random.seed!(2019)

(α,β) = (1.5,-1.)
(δ,μ,T) = (2.,1.,1.)

rand_plot(ConvexMinorantStableMeander(α,β))
rand_plot(ConvexMinorantWeaklyStable(α,β,δ,μ,T))

```

<a name="authors"/>


## Author and Contributor List
Jorge I. González Cázares


Aleksandar Mijatović 
Gerónimo Uribe Bravo
