#todo realized cost
#todo randomized initial conditions
#compare: sp, gnep, bilevel (shared brain)
#if it works, also compare bilevel (distributed brain)

using Random
#using EPEC
#using GLMakie
using Plots
#include("../examples/racing.jl")

#probs = setup(; T=10, 
#	Δt = 0.1, 
#	r = 1.0, 
#	α1 = 1e-1,
#	α2 = 1e-4,
#	β = 1e0, #.5, # sensitive to high values
#	cd = 0.2, #0.25,
#	u_max_nominal = 1.0, 
#	u_max_drafting = 2.5, #2.5, # sensitive to high difference over nominal 
#	box_length = 5.0,
#	box_width = 2.0,
#	lat_max = 1.5);

#lat_max = probs.params.lat_max;
#r_offset_min = probs.params.r;
lat_max = 1.5;
r_offset_min = 1.;
r_offset_max = 3.;
# select x0a at origin
sample_size = 1000;

# choose random P1 lateral position
a_lat_pos0_arr = -lat_max .+ 2*lat_max .* rand(MersenneTwister(), sample_size)
# fix P1 longitudinal pos at 0
a_pos0_arr = hcat(a_lat_pos0_arr, zeros(sample_size, 1))
# choose random radial offset for P2
# for loop ... prune until uniform distribution within lanemax:
r_offset_arr = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister(), sample_size))
ϕ_offset_arr = rand(MersenneTwister(), sample_size) * 2 * π
# compute P2 pos wrt P1
b_offset_arr = hcat(r_offset_arr .* cos.(ϕ_offset_arr), r_offset_arr .* sin.(ϕ_offset_arr))
b_pos0_arr = a_pos0_arr .+ b_offset_arr
# prune out of bounds lateral positions

scatter(a_pos0_arr[:, 1], a_pos0_arr[:, 2])
scatter!(b_pos0_arr[:, 1], b_pos0_arr[:, 2])

# choose random offset for P2
lat_offset_min = -lat_max
long_offset_min = -2
long_offset_max = 2
b_lat_offset0_arr = lat_max .+ 2*lat_max .* rand(MersenneTwister(), sample_size)
b_long_offset0_arr = y_min .+ (y_max - y_min) .* rand(MersenneTwister(), sample_size)





x_min = -1
x_max = 1
y_min = -1
y_max = 1
xs = x_min .+ (x_max - x_min) .* rand(MersenneTwister(), sample_size)
ys = y_min .+ (y_max - y_min) .* rand(MersenneTwister(), sample_size)

if (xs.^2 + ys.^2)
scatter(xs, ys)


rand(MersenneTwister(0), 2, 2)
# select x0 around origin r < d < n * r distance away

#x0 = [1., 3, 0, 1, -1, 2, 0, 1.5] # it's helpful to start from an initial velocity difference for oscillating behavior but so sensitive
#x0 = [.5, 0, 0, 2, 0, 1, 0, 1]

#x0 = [
#	1.2311987492087133
#	101.10852498215952
#	  0.25398773930502044
#	  7.670003557087554
#	  0.2697550632692793
#	100.80995899345245
#	  0.569618865585001
#	  6.604781659301143]

#(; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z) = solve_seq_adaptive(probs, x0);
#(f, ax, XA, XB, lat) = visualize(; rad = sqrt(probs.params.r) / 2, lat = probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T = probs.params.T, lat = lat)

#sim_results = solve_simulation(probs, 200; x0, only_want_gnep=true);
#animate(probs, sim_results; save=false);

#prefs = zeros(Int, length(sim_results))
#for key in keys(sim_results)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = sim_results[key].lowest_preference;
#end

#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")