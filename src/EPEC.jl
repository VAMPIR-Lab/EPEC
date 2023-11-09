module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics

using Infiltrator
    
PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")

include("problems.jl")
include("../examples/stackelnash_racing.jl")

export OptimizationProblem, solve

end # module EPEC
