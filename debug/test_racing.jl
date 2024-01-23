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
	u_max_drafting = 5.0,
	box_length = 5.0,
	box_width = 1.0,
	lat_max = 1.0);

x0 = [1., 3, 0, 1, -1, 2, 0, 1]

sim_results = solve_simulation(probs, 10; x0);
