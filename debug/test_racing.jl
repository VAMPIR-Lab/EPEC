using EPEC
using GLMakie
using Plots
include("../examples/racing.jl")

probs = setup(; T=10, 
	Δt = 0.1, 
	r = 1.0, 
	α1 = 1e-2,
	α2 = 1e-4,
	β = 1e0,
	cd = 0.2,
	u_max_nominal = 2.0, 
	u_max_drafting = 3.0,
	box_length = 5.0,
	box_width = 1.0,
	lat_max = 1.5);

#x0 = [1., 3, 0, 1, -1, 2, 0, 1]
x0 = [0, 0, 0, 1.5, 0.25, 1, 0, 1]
#(; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z) = solve_seq_adaptive(probs, x0);
#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)

sim_results = solve_simulation(probs, 200; x0, only_want_gnep=false);
animate(probs, sim_results; save=false);

prefs = zeros(Int, length(sim_results))
for key in keys(sim_results)
    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
	prefs[key] = sim_results[key].lowest_preference;
end

histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")