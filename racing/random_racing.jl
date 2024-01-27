#todo realized cost DONE
#todo randomized initial conditions DONE
#compare: sp, gnep, bilevel (shared brain) DONE (except sp)
#if it works, also compare bilevel (distributed brain)
# save at intervals DONE
#ivestigate max min huge outliers DONE

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
    α1=1e-2,
    α2=1e-4,
    α3=1e-2,
    β=1e-2, #.5, # sensitive to high values
    cd=0.2, #0.25,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=1.5);

is_x0s_from_file = true;
is_results_from_file = true;
data_dir = "data"
init_filename = "x0s_100samples_2024-01-27_1012";
results_filename = "results_x0s_100samples_2024-01-27_1012_2024-01-27_1026_100steps";
sample_size = 100;
time_steps = 100;
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
    init_filename = "x0s_$(sample_size)samples_$(Dates.format(now(),"YYYY-mm-dd_HHMM"))"
    jldsave("$(data_dir)/$(init_filename).jld2"; x0s, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
end


sp_results = []
gnep_results = []
bilevel_results = []
elapsed = []

if is_results_from_file
    results_file = jldopen("$(data_dir)/$(results_filename).jld2", "r")
    sp_results = results_file["sp_results"]
    gnep_results = results_file["gnep_results"]
    bilevel_results = results_file["bilevel_results"]
    elapsed = results_file["elapsed"]
else
    sp_results = Dict()
    gnep_results = Dict()
    bilevel_results = Dict()

    # Create a progress bar
    #prog = Progress(length(x0s), barlen=50)

    start = time()

    for (index, x0) in x0s
        try
            sp_res = solve_simulation(probs, time_steps; x0=x0, only_want_sp=true)
            sp_results[index] = sp_res
        catch er
            @info "sp errored $index"
            println(er)
        end

        try
            gnep_res = solve_simulation(probs, time_steps; x0=x0, only_want_gnep=true)
            gnep_results[index] = gnep_res
        catch er
            @info "gnep errored $index"
            println(er)
        end

        try
            bilevel_res = solve_simulation(probs, time_steps; x0=x0)
            bilevel_results[index] = bilevel_res
        catch er
            @info "bilevel errored $index"
            println(er)
        end

        #next!(prog)
    end
    elapsed = time() - start
    #finish!(prog)

    # save
    jldsave("$(data_dir)/results_$(init_filename)_$(Dates.format(now(),"YYYY-mm-dd_HHMM"))_$(time_steps)steps.jld2"; params=probs.params, x0s, sp_results, gnep_results, bilevel_results, elapsed)
end

sp_steps = Dict()
gnep_steps = Dict()
bilevel_steps = Dict()

sp_costs = Dict()
gnep_costs = Dict()
bilevel_costs = Dict()

for (index, res) in sp_results
    len = length(res)
    sp_steps[index] = len
    sp_costs[index] = compute_realized_cost(res)
end

for (index, res) in gnep_results
    len = length(res)
    gnep_steps[index] = len
    gnep_costs[index] = compute_realized_cost(res)
end

for (index, res) in bilevel_results
    len = length(res)
    bilevel_steps[index] = len
    bilevel_costs[index] = compute_realized_cost(res)
end

#@info "Average sp time steps: $(mean(values(sp_steps)))"
#@info "Average gnep time steps: $(mean(values(gnep_steps)))"
#@info "Average bilevel time steps: $(mean(values(bilevel_steps)))"
#bins = 1:time_steps+1
#histogram1 = histogram(collect(values(sp_steps)), bins=bins, label="sp")
#histogram2 = histogram(collect(values(gnep_steps)), bins=bins, label="gnep")
#histogram3 = histogram(collect(values(bilevel_steps)), bins=bins, xlabel="# steps", label="bilevel")
#Plots.plot(histogram1, histogram2, histogram3, layout=(3, 1), legend=true, ylabel="# x0")


#p = Plots.plot()
#for (i, c) in bilevel_costs
#    Plots.plot!(p, c.a.running.total, label="")
#end
#p

#Plots.plot(a_cost_breakdown.running.lane)
#Plots.plot!(b_cost_breakdown.running.lane)

# trim with specified time steps
sp_costs_tr = trim_by_steps(sp_costs, sp_steps; min_steps=time_steps)
gnep_costs_tr = trim_by_steps(gnep_costs, gnep_steps; min_steps=time_steps)
bilevel_costs_tr = trim_by_steps(bilevel_costs, bilevel_steps; min_steps=time_steps)
all_costs = extract_intersected_costs(gnep_costs_tr, gnep_costs_tr, bilevel_costs_tr)

@info "total Δcost = bilevel - gnep"
Δcost_total = compute_player_Δcost(all_costs.gnep.total, all_costs.bilevel.total)
print_mean_min_max(Δcost_total.P1_abs, Δcost_total.P2_abs, Δcost_total.P1_rel, Δcost_total.P2_rel)

@info "lane Δcost = bilevel - gnep"
Δcost_lane = compute_player_Δcost(all_costs.gnep.lane, all_costs.bilevel.lane)
print_mean_min_max(Δcost_lane.P1_abs, Δcost_lane.P2_abs, Δcost_lane.P1_rel, Δcost_lane.P2_rel)

@info "control Δcost = bilevel - gnep"
Δcost_control = compute_player_Δcost(all_costs.gnep.control, all_costs.bilevel.control)
print_mean_min_max(Δcost_control.P1_abs, Δcost_control.P2_abs, Δcost_control.P1_rel, Δcost_control.P2_rel)

@info "velocity Δcost = bilevel - gnep"
Δcost_velocity = compute_player_Δcost(all_costs.gnep.velocity, all_costs.bilevel.velocity)
print_mean_min_max(Δcost_velocity.P1_abs, Δcost_velocity.P2_abs, Δcost_velocity.P1_rel, Δcost_velocity.P2_rel)

@info "terminal Δcost = bilevel - gnep"
Δcost_terminal = compute_player_Δcost(all_costs.gnep.terminal, all_costs.bilevel.terminal)
print_mean_min_max(Δcost_terminal.P1_abs, Δcost_terminal.P2_abs, Δcost_terminal.P1_rel, Δcost_terminal.P2_rel)

# best
best_ind_P1 = all_costs.ind[Δcost_total.P1_max_ind]
best_ind_P2 = all_costs.ind[Δcost_total.P2_max_ind]
#@assert(best_ind_P1 == best_ind_P2)
## worst
worst_ind_P1 = all_costs.ind[Δcost_total.P1_min_ind]
worst_ind_P2 = all_costs.ind[Δcost_total.P2_min_ind]
#@assert(worst_ind_P1 == worst_ind_P2)

#animate(probs, gnep_results[best_ind_P1]; save=false, filename="gnep_best_case.mp4");
#animate(probs, bilevel_results[best_ind_P1]; save=false, filename="bilevel_best_case.mp4");
#animate(probs, gnep_results[worst_ind_P1]; save=false, filename="gnep_worst_case.mp4");
#animate(probs, bilevel_results[worst_ind_P1]; save=false, filename="bilevel_worst_case.mp4");

#res = gnep_results[worst_ind_P1];
#res = bilevel_results[worst_ind_P1];
#prefs = zeros(Int, length(res));
#for key in keys(res)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = res[key].lowest_preference;
#end
#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")