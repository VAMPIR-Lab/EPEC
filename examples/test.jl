using EPEC
using GLMakie
include("simple_racing.jl")


probs = setup(; T=10, 
    Δt = 0.1, 
    r=1.0, 
    α1 = 0.01,
    α2 = 0.001,
    cd = 0.05,
    u_max_nominal = 2.0, 
    u_max_drafting = 5.0,
    box_length=5.0,
    box_width=1.0,
    lat_max = 10.0)

x0 = [2, 0, 0, 5, 0, 0, 0, 7]
sim_results = solve_simulation(probs, 50; x0);
visualize(probs, sim_results; save=false);