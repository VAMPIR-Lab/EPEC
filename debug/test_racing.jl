using EPEC
#using GLMakie
#using LinearAlgebra
#using Plots
#include("../racing/racing.jl")
#include("../racing/visualize_racing.jl")

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    α3=1e-1,
    β=1e-1, #.5, # sensitive to high values
    cd=0.2, #0.25,
    d=2.0,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=4.5);

#x0 = [1.0, 3, 0, 1, -1, 2, 0, 1.5] # it's helpful to start from an initial velocity difference for oscillating behavior but so sensitive
x0 = [.5, 0, 1, π/2, 0, 1, 1, π/2]	

road = Dict(3 => 0, 6 => 0, 9 => -1, 12 => -2.5, 15 => -2.5, 18 => -1, 21 => 0, 24 => .5, 27 => 0);
sim_results = solve_simulation(probs, 50; x0, road, mode=9);

animate(probs, sim_results; save=false, mode=9, road);

#(; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z) = solve_seq_adaptive(probs, x0);
#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)

#prefs = zeros(Int, length(sim_results))
#for key in keys(sim_results)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = sim_results[key].lowest_preference;
#end

#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")    