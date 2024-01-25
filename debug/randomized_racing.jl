#todo realized cost DONE
#todo randomized initial conditions DONE
#compare: sp, gnep, bilevel (shared brain)
#if it works, also compare bilevel (distributed brain)

using Random
using EPEC
using GLMakie
using Plots
include("../examples/racing.jl")

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-1,
    α2=1e-4,
    β=1e0, #.5, # sensitive to high values
    cd=0.2, #0.25,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=1.5);

sample_size = 2;
r_offset_max = 3.0; # maximum distance between P1 and P2
long_vel_max = 2.0; # maximum longitudunal velocity
lat_max = probs.params.lat_max;
r_offset_min = probs.params.r;

# choose random P1 lateral position inside the lane limits, long pos = 0
a_lat_pos0_arr = -lat_max .+ 2 * lat_max .* rand(MersenneTwister(), sample_size) # .5 .* ones(sample_size)
# fix P1 longitudinal pos at 0
a_pos0_arr = hcat(a_lat_pos0_arr, zeros(sample_size, 1))
b_pos0_arr = zeros(size(a_pos0_arr))
# choose random radial offset for P2
for i in 1:sample_size
    r_offset = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister()))
    ϕ_offset = rand(MersenneTwister()) * 2 * π
    b_lat_pos0 = a_pos0_arr[i, 1] + r_offset * cos(ϕ_offset)
    # reroll until we b lat pos is inside the lane limits
    while b_lat_pos0 > lat_max || b_lat_pos0 < -lat_max
        r_offset = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister()))
        ϕ_offset = rand(MersenneTwister()) * 2 * π
        b_lat_pos0 = a_pos0_arr[i, 1] + r_offset * cos(ϕ_offset)
    end
    b_long_pos0 = a_pos0_arr[i, 2] + r_offset * sin(ϕ_offset)
    b_pos0_arr[i, :] = [b_lat_pos0, b_long_pos0]
end

@assert minimum(sqrt.(sum((a_pos0_arr .- b_pos0_arr) .^ 2, dims=2))) >= 1.0 # probs.params.r
@assert all(-lat_max .< b_pos0_arr[:, 1] .< lat_max)
#scatter(a_pos0_arr[:, 1], a_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)
#scatter!(b_pos0_arr[:, 1], b_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)

# keep lateral velocity zero
a_vel0_arr = hcat(zeros(sample_size), long_vel_max .* rand(MersenneTwister(), sample_size))
b_vel0_arr = hcat(zeros(sample_size), long_vel_max .* rand(MersenneTwister(), sample_size))

x0_arr = hcat(a_pos0_arr, a_vel0_arr, b_pos0_arr, b_vel0_arr)

# uuh
x0s = Dict{Int,Vector{Float64}}()

for (index, row) in enumerate(eachrow(x0_arr))
    x0s[index] = row
end
sim_results = Dict()
costs = Dict()

# how to multithread?
for (index, x0) in x0s
    results = solve_simulation(probs, 50; x0, only_want_gnep=true)
    z_arr = zeros(length(results), 12)

    for t in eachindex(results)
        z_arr[t, 1:4] = results[t].x0[1:4]
        z_arr[t, 5:6] = results[t].U1[1, :]
        z_arr[t, 7:10] = results[t].x0[5:8]
        z_arr[t, 11:12] = results[t].U2[1, :]
    end
    Z = [z_arr[:]; zeros(8)] # making it work with f1(Z) and f2(Z)
    a_cost = probs.OP1.f(Z)
    b_cost = probs.OP2.f(Z)
    sim_results_dict[index] = results
    costs[index] = [a_cost, b_cost]
end

#using Threads
#num_threads = Threads.nthreads()

#x0 = [1., 3, 0, 1, -1, 2, 0, 1.5]
#sample = rand(1:sample_size)
#x0 = x0_arr[sample, :];

#(; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z) = solve_seq_adaptive(probs, x0);
#(f, ax, XA, XB, lat) = visualize(; rad=sqrt(probs.params.r) / 2, lat=probs.params.lat_max + sqrt(probs.params.r) / 2);
#display(f)
#update_visual!(ax, XA, XB, x0, P1, P2; T=probs.params.T, lat=lat)

#sim_results = solve_simulation(probs, 50; x0, only_want_gnep=true);
#animate(probs, sim_results; save=false);

#prefs = zeros(Int, length(sim_results))
#for key in keys(sim_results)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = sim_results[key].lowest_preference;
#end

#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")

#z_arr = zeros(length(sim_results), 12)

#for t in eachindex(sim_results)
#    z_arr[t, 1:4] = sim_results[t].x0[1:4]
#    z_arr[t, 5:6] = sim_results[t].U1[1, :]
#    z_arr[t, 7:10] = sim_results[t].x0[5:8]
#    z_arr[t, 11:12] = sim_results[t].U2[1, :]
#end
#Z = [z_arr[:]; zeros(8)] # making it work with f1(Z) and f2(Z)

#a_cost = probs.OP1.f(Z)
#b_cost = probs.OP2.f(Z)