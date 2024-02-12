module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics
using GLMakie
using Infiltrator

    
include("problems.jl")
include("../examples/tag_chain.jl")

export OptimizationProblem, solve

end # module EPEC
