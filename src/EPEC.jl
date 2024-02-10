module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics
using GLMakie

using Infiltrator
    

include("problems.jl")
include("../examples/lifted_constrained.jl")

export OptimizationProblem, solve

end # module EPEC
