using EPEC
using GLMakie
include("../examples/racing.jl")

probs = setup(; T=10, 
	Δt = 0.1, 
	r = 1.0, 
	α1 = 1e-1,
	α2 = 1e-4,
	β = 1e1,
	cd = 0.01,
	u_max_nominal = 1.0, 
	u_max_drafting = 3.0,
	box_length = 5.0,
	box_width = 1.0,
	lat_max = 1.0);

x0 = [1., 3, 0, 1, -1, 2, 0, 1]

(; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z) = solve_seq_adaptive(probs, x0);
#show_me(θ_bilevel, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
#sim_results = solve_simulation(probs, 10; x0);

#for key in keys(sim_results)
#    println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#end

#animate(probs, sim_results; save=false);
