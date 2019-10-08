__precompile__(true)

module StableMeander

  using Distributions
  using StatsBase
  using LambertW
  using FastGaussQuadrature
  using SpecialFunctions
  using DataFrames
  using DataFramesMeta


  export
    # distribution types
    Stable,
    PositiveStable,
    StableMeander,
    PreciseStableMeander,
    LocStableMeander,
    MvStableMeander,
    PreciseMvStableMeander,
    LocMvStableMeander,
    ConvexMinorantStableMeander,
    CrossConvexMinorantStableMeander,
    ConvexMinorantWeaklyStable,
    CrossConvexMinorantWeaklyStable,

    # methods
    rand,
    maximum,
    minimum,
    insupport,
    pdf,
    cdf,
    mgf,
    cf,
    mellin,
    mean,
    params,
    length

  # Source Files

  include("stable.jl")
  include("positivestable.jl")
  include("stablemeander.jl")
  include("convexminorantstable.jl")

end
