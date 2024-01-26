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

sample_size = 1000;
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
Plots.scatter(a_pos0_arr[:, 1], a_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)
Plots.scatter!(b_pos0_arr[:, 1], b_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)

# keep lateral velocity zero
a_vel0_arr = hcat(zeros(sample_size), long_vel_max .* rand(MersenneTwister(), sample_size))
b_vel0_arr = hcat(zeros(sample_size), long_vel_max .* rand(MersenneTwister(), sample_size))

x0_arr = hcat(a_pos0_arr, a_vel0_arr, b_pos0_arr, b_vel0_arr)

# uuh
x0s = Dict{Int,Vector{Float64}}()

for (index, row) in enumerate(eachrow(x0_arr))
    x0s[index] = row
end
gnep_results = Dict()
bilevel_results = Dict()
gnep_costs = Dict()
bilevel_costs = Dict()

function compute_realized_cost(res)
    z_arr = zeros(length(res), 12)

    for t in eachindex(res)
        z_arr[t, 1:4] = res[t].x0[1:4]
        z_arr[t, 5:6] = res[t].U1[1, :]
        z_arr[t, 7:10] = res[t].x0[5:8]
        z_arr[t, 11:12] = res[t].U2[1, :]
    end
    Z = [z_arr[:]; zeros(8)] # making it work with f1(Z) and f2(Z)
    a_cost = probs.OP1.f(Z)
    b_cost = probs.OP2.f(Z)
    (a_cost, b_cost)
end

# how to multithread?
#using Threads
#num_threads = Threads.nthreads()
for (index, x0) in x0s
    @info "Solving $index: $x0"

    try
        gnep_res = solve_simulation(probs, 50; x0, only_want_gnep=true)
        gnep_a_cost, gnep_b_cost = compute_realized_cost(gnep_res)
        gnep_results[index] = gnep_res
        gnep_costs[index] = [gnep_a_cost, gnep_b_cost]
    catch err
        @info "gnep failed $index: $x0"
        println(err)
    end

    try
        bilevel_res = solve_simulation(probs, 50; x0, only_want_gnep=false)
        bilevel_a_cost, bilevel_b_cost = compute_realized_cost(bilevel_res)

        bilevel_results[index] = bilevel_res
        bilevel_costs[index] = [bilevel_a_cost, bilevel_b_cost]
    catch err
        @info "bilevel failed $index: $x0"
        println(err)
    end
end

# save data
using JLD2

jldsave("2024-01-25.jld2"; x0s, gnep_results, gnep_costs, bilevel_results, bilevel_costs)

bilevel_costs_arr = []
gnep_costs_arr = []

for (index, gnep_cost) in gnep_costs
	if haskey(bilevel_costs, index)
		push!(gnep_costs_arr, gnep_cost)
		push!(bilevel_costs_arr, bilevel_costs[index])
	end
end

using Statistics

P1_gnep_costs = [v[1] for v in gnep_costs_arr]
P1_bilevel_costs = [v[1] for v in bilevel_costs_arr]
P2_gnep_costs = [v[2] for v in gnep_costs_arr]
P2_bilevel_costs = [v[2] for v in bilevel_costs_arr]

# bilevel vs gnep
P1_cost_diff = P1_bilevel_costs .- P1_gnep_costs
P2_cost_diff  = P2_bilevel_costs .- P2_gnep_costs
P1_rel_cost_diff = (P1_bilevel_costs .- P1_gnep_costs) ./ P1_gnep_costs
P2_rel_cost_diff  = (P2_bilevel_costs .- P2_gnep_costs) ./ P2_gnep_costs
println("					mean 		std 			min			max")
println("P1 cost \"bilevel - gnep\" abs :  $(mean(P1_cost_diff))  $(std(P1_cost_diff))  $(minimum(P1_cost_diff))  $(maximum(P1_cost_diff))")
println("P2 cost \"bilevel - gnep\" abs :  $(mean(P2_cost_diff))  $(std(P2_cost_diff))  $(minimum(P2_cost_diff))  $(maximum(P2_cost_diff))")
println("P1 cost \"bilevel - gnep\" rel%:  $(mean(P1_rel_cost_diff)*100)  $(std(P1_rel_cost_diff)*100)  $(minimum(P1_rel_cost_diff)*100)  $(maximum(P1_rel_cost_diff)*100)")
println("P2 cost \"bilevel - gnep\" rel%:  $(mean(P2_rel_cost_diff)*100)  $(std(P2_rel_cost_diff)*100)  $(minimum(P2_rel_cost_diff)*100)  $(maximum(P2_rel_cost_diff)*100)")