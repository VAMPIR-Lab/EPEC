using EPEC
using GLMakie
include("../examples/simple_racing.jl")

probs = setup(; T=10, 
	Δt = 0.1, 
	r=1.0, 
	α1 = 1e-4,
	α2 = 0e0,
	cd = 0.01,
	u_max_nominal = 2.0, 
	u_max_drafting = 6.0,
	box_length=5.0,
	box_width=1.0,
	lat_max = 1);

x0 = [0, .5, 0, 5, -1, 0, 0, 6]
(; P1, P2, gd_both, h, U1, U2) = solve_seq(probs, x0);
(f, ax, XA, XB, lat) = visualize(probs);
display(f)
update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)