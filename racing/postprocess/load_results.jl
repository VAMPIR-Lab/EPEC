#using EPEC
#using LaTeXStrings
#using Plots
#using GLMakie
using JLD2
#using LaTeXStrings

#include("racing.jl")
#include("random_racing_helper.jl")

#probs = setup(; T=10,
#    Δt=0.1,
#    r=1.0,
#    α1=1e-3,
#    α2=1e-4,
#    α3=1e-1,
#    β=1e-1, #.5, # sensitive to high values
#    cd=0.2, #0.25,
#    d=1.5, # actual road width (±)
#    u_max_nominal=1.0,
#    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
#    box_length=5.0,
#    box_width=2.0,
#    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
#    );

data_dir = "data"
x0s_filename = "x0s_100samples_2024-04-16_0001"
results_suffix = "_(x0s_100samples_2024-04-16_0001)_2024-04-16_0001_50steps";
init_file = jldopen("$(data_dir)/$(x0s_filename).jld2", "r")
x0s = init_file["x0s"]

#xdim=4
#udim=2