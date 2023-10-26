module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics

using Infiltrator

include("problems.jl")

export OptimizationProblem

end # module EPEC
