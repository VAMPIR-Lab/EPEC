using EPEC
using GLMakie
include("../examples/simple_racing.jl")

probs = setup(; T=1, 
	Δt = 0.1, 
	r = 1.0, 
	α1 = 1e-2,
	α2 = 0.,
	β = 1e1,
	cd = 0.01,
	u_max_nominal = 2.0, 
	u_max_drafting = 5.0,
	box_length = 5.0,
	box_width = 1.0,
	lat_max = 1.5);

#x0 = [0.024854049940194006
#2.0390008441072673
#0.09708099864039535
#5.389942809396402
#-0.9631049795145318
#1.2392045199100574
#0.33790040971200136
#6.392074687707442]

x0 =  [ -0.08320811747237386
10.724680707142564
 0.1876399601770615
 9.494661937893104
-0.8612120474445999
10.047477657701165
-0.7481148411504008
10.198952048696231]

#x0 = [	-0.129881229330084
# 67.53598465242898
# -0.4214931085044126
# 14.28762513526873
# -0.999999959967876
# 67.03301075283929
# -1.2582422206531267
# 14.073867298367208
#]

(; P1, P2, gd_both, h, U1, U2) = solve_seq(probs, x0);
#after infiltrate:
#show_me(safehouse.θ_out, safehouse.w; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

# visualize P1, P2:
#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)

#  try sim
sim_results = solve_simulation(probs, 100; x0);
animate(probs, sim_results; save=false);