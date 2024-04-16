module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics
using GLMakie
using Infiltrator

include("problems.jl")
include("../racing/racing.jl")
include("../racing/visualize_racing.jl")

export OptimizationProblem, solve, setup, solve_simulation, animate

end # module EPEC
