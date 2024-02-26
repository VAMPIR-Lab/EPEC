using EPEC
using GLMakie
using Plots
include("../racing/phantom_racing.jl")
#include("../racing/racing.jl")
include("../racing/visualize_racing.jl")

probs = setup(; T=2,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    β=1e-1, #.5, # sensitive to high values
    cd=0.2, #0.25,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=2.0);


#x0 = [1.0, 3, 0, 1, -1, 2, 0, 1.5] # it's helpful to start from an initial velocity difference for oscillating behavior but so sensitive

# sometimes I randomly sample from x0s_2000samples_2024-02-01_1739 -- I call include("racing/process_results.jl") first
#i = rand(1:2000)
#@info "$i"
#x0 = x0s[i];

x0 = [-1.097800632523576
0.0
0.0
1.6000274227911389
0.5669216447507093
1.22438749327142
0.0
0.29792647730429356]

sim_results = solve_simulation(probs, 1; x0);
animate(probs, sim_results; save=false);
