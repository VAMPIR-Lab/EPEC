module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics

using Infiltrator
    

include("problems.jl")
include("../examples/stackelnash_racing.jl")

export OptimizationProblem, solve

end # module EPEC
