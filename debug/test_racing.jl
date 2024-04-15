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
    d=2.0, # actual road width (±)
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
    );

#x0 = [1.0, 3, 0, 1, -1, 2, 0, 1.5] # it's helpful to start from an initial velocity difference for oscillating behavior but so sensitive
#x0 = [.5, 0, 1, π/2, 0, 2, 1, π/2]	
#x0 = [-.5, 0, 2, π/2, 0, 2, 1, π/2]	
#x0 = [0, 2, 1, π/2,-.5, 0, 2, π/2]	
#
x0 = [ -0.9339583266548017
0.0
2.5321772895121644
1.5707963267948966
-1.1050599392379927
-2.950381877667683
1.6054315472628076
1.5707963267948966]

#road = Dict(3 => 0, 6 => 0, 9 => -1, 12 => -2.5, 15 => -2.5, 18 => -1, 21 => 0, 24 => .5, 27 => 0);
road = Dict(0 => 0, 2 => 0, 4=>-.2, 6=>-.4, 8=>-.8, 10=>-1.6, 12=>-2.2, 14=>-2.2, 16=>-1.6, 18=>-.7, 20=>.7, 22=>1.6, 24=>2.2, 26=>2.2, 28=>1.6, 30=>.8,32=>.4,34=>.2,36=>0,38=>0, 40=>-.01);
mode = 1;
sim_results = solve_simulation(probs, 50; x0, road, mode);
animate(probs, sim_results; save=false, mode, road);

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