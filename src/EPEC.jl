module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics
using GLMakie
using JLD2
using Random
using Infiltrator

include("problems.jl")
include("../racing/racing.jl")
include("../racing/visualize_racing.jl")
include("../racing/random_racing_helper.jl")

export OptimizationProblem, solve, setup, solve_simulation, animate

end # module EPEC
