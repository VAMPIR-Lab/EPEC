module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics
using GLMakie

using Infiltrator
    

include("problems.jl")
include("../examples/simple_racing.jl")

export OptimizationProblem, solve

end # module EPEC
