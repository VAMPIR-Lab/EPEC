#todo realized cost DONE
#todo randomized initial conditions DONE
#compare: sp, gnep, bilevel (shared brain) DONE (except sp)
#if it works, also compare bilevel (distributed brain)
# save at intervals
#ivestigate max min huge outliers

using EPEC
using GLMakie
using Plots
using Dates
using JLD2
#using ProgressMeter

include("racing.jl")
include("random_racing_helper.jl")

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

is_x0s_from_file = false;
is_results_from_file = false;
data_dir = "data"
init_filename = "x0s_100samples_2024-01-26_0912";
results_filename = "results_x0s_1000samples_2024-01-25_2315_2024-01-26_0125_100steps";
sample_size = 10;
time_steps = 50;
r_offset_max = 3.0; # maximum distance between P1 and P2
a_long_vel_max = 3.0; # maximum longitudunal velocity for a
b_long_vel_delta_max = 1.0 # maximum longitudunal delta velocity for a
lat_max = probs.params.lat_max;
r_offset_min = probs.params.r;

x0s = Dict{Int,Vector{Float64}}()

if (is_x0s_from_file)
    # WARNING params not loaded from file
    init_file = jldopen("$(data_dir)/$(init_filename).jld2", "r")
    x0s = init_file["x0s"]
    #@infiltrate
    #Plots.scatter(x0_arr[:, 1], x0_arr[:, 2], aspect_ratio=:equal, legend=false)
    #Plots.scatter!(x0_arr[:, 5], x0_arr[:, 6], aspect_ratio=:equal, legend=false)
else
    x0s = generate_x0s(sample_size, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    jldsave("$(data_dir)/x0s_$(sample_size)samples_$(Dates.format(now(),"YYYY-mm-dd_HHMM")).jld2"; x0s, lat_max, r_offset_min, r_offset_max,  a_long_vel_max, b_long_vel_delta_max)
end

sp_results = []
gnep_results = []
bilevel_results = []

if is_results_from_file
    results_file = jldopen("$(data_dir)/$(results_filename).jld2", "r")
    sp_results = results_file["sp_results"]
    gnep_results = results_file["gnep_results"]
    bilevel_results = results_file["bilevel_results"]
else
    sp_results = Dict()
    gnep_results = Dict()
    bilevel_results = Dict()

    # Create a progress bar
    #prog = Progress(length(x0s), barlen=50)

    start = time()

    for (index, x0) in x0s
        try
            sp_res = solve_simulation(probs, time_steps; x0, only_want_sp=true)
            sp_results[index] = sp_res
        catch err
            @info "sp failed $index: $x0"
            println(err)
        end

        try
            gnep_res = solve_simulation(probs, time_steps; x0, only_want_gnep=true)
            gnep_results[index] = gnep_res
        catch err
            @info "gnep failed $index: $x0"
            println(err)
        end

        try
            bilevel_res = solve_simulation(probs, time_steps; x0, only_want_gnep=false)
            bilevel_results[index] = bilevel_res
        catch err
            @info "bilevel failed $index: $x0"
            println(err)
        end

        #next!(prog)
    end
    #elapsed = time() - start
    #finish!(prog)

    # save
    if is_x0s_from_file
        jldsave("$(data_dir)/results_$(init_filename)_$(Dates.format(now(),"YYYY-mm-dd_HHMM"))_$(time_steps)steps.jld2"; params=probs.params, x0s, sp_results, gnep_results, bilevel_results, elapsed)
    else
        jldsave("$(data_dir)/results_$(Dates.format(now(),"YYYY-mm-dd_HHMM")).jld2"; params=probs.params, x0s, sp_results, gnep_results, bilevel_results, elapsed)
    end
end

# only look up to time_steps
sp_costs = Dict()
gnep_costs = Dict()
bilevel_costs = Dict()

for (index, sp_res) in sp_results
    sp_costs[index] = compute_realized_cost(Dict(i => sp_res[i] for i in 1:time_steps))
end

for (index, gnep_res) in gnep_results
    gnep_costs[index] = compute_realized_cost(Dict(i => gnep_res[i] for i in 1:time_steps))
end

for (index, bilevel_res) in bilevel_results
    bilevel_costs[index] = compute_realized_cost(Dict(i => bilevel_res[i] for i in 1:time_steps))
end

# sp fails so much!
all_costs = extract_costs(gnep_costs, gnep_costs, bilevel_costs)

@info "total Δcost"
Δcost_total = compute_player_Δcost(all_costs.gnep.total, all_costs.bilevel.total)
print_mean_min_max(Δcost_total.P1_abs, Δcost_total.P2_abs, Δcost_total.P1_rel, Δcost_total.P2_rel)

@info "lane Δcost"
Δcost_lane = compute_player_Δcost(all_costs.gnep.lane, all_costs.bilevel.lane)
print_mean_min_max(Δcost_lane.P1_abs, Δcost_lane.P2_abs, Δcost_lane.P1_rel, Δcost_lane.P2_rel)

@info "control Δcost:"
Δcost_control = compute_player_Δcost(all_costs.gnep.control, all_costs.bilevel.control)
print_mean_min_max(Δcost_control.P1_abs, Δcost_control.P2_abs, Δcost_control.P1_rel, Δcost_control.P2_rel)

@info "terminal Δcost:"
Δcost_terminal = compute_player_Δcost(all_costs.gnep.terminal, all_costs.bilevel.terminal)
print_mean_min_max(Δcost_terminal.P1_abs, Δcost_terminal.P2_abs, Δcost_terminal.P1_rel, Δcost_terminal.P2_rel)


# best
P1_best_ind = all_costs.ind[Δcost_total.P1_max_ind]
P2_best_ind = all_costs.ind[Δcost_total.P2_max_ind] 
## worst
P1_worst_ind = all_costs.ind[Δcost_total.P1_min_ind] 
P2_worst_ind = all_costs.ind[Δcost_total.P2_min_ind] 

animate(probs, gnep_results[P1_worst_ind]; save=false);
animate(probs, bilevel_results[P2_worst_ind]; save=false);

#prefs = zeros(Int, length(sim_results))
#for key in keys(sim_results)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = sim_results[key].lowest_preference;
#end

#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")