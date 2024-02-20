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
    α1=1e-3,
    α2=1e-4,
    α3=1e-1,
    β=1e-1, #.5, # sensitive to high values
    cd=0.2, #0.25,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=2.0);

is_x0s_from_file = false;
is_results_from_file = false;
data_dir = "data"
init_filename = "x0s_10000samples_2024-01-26_2216";
results_filename = "results_x0s_10000samples_2024-01-26_2216_2024-01-27_2035_200steps";
sample_size = 10;
time_steps = 100;
r_offset_max = 3.0; # maximum distance between P1 and P2
a_long_vel_max = 3.0; # maximum longitudunal velocity for a
b_long_vel_delta_max = 1.0 # maximum longitudunal delta velocity for a
lat_max = probs.params.lat_max;
r_offset_min = probs.params.r + probs.params.col_buffer;


if (is_x0s_from_file)
    # WARNING params not loaded from file
    init_file = jldopen("$(data_dir)/$(init_filename).jld2", "r")
    x0s = init_file["x0s"]
else
    x0s = generate_x0s(sample_size, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    init_filename = "x0s_$(sample_size)samples_$(Dates.format(now(),"YYYY-mm-dd_HHMM"))"
    jldsave("$(data_dir)/$(init_filename).jld2"; x0s, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    x0s
end


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
    x0s_len = length(x0s)
    progress = 0

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
        global progress
        progress += 1
        @info "Progress $(progress/x0s_len*100)%"
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


# trim with specified time steps
trim_steps = 200
is_trimming = false # too much for plots otherwise
#gnep_costs_tr = trim_by_steps(gnep_costs, gnep_steps; min_steps=trim_steps)
#bilevel_costs_tr = trim_by_steps(bilevel_costs, bilevel_steps; min_steps=trim_steps)

for (index, res) in sp_results
    len = length(res)
    sp_steps[index] = len

    if is_trimming
        if len >= trim_steps
            sp_costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
        end
    else
        sp_costs[index] = compute_realized_cost(res)
    end
end

for (index, res) in gnep_results
    len = length(res)
    gnep_steps[index] = len

    if is_trimming
        if len >= trim_steps
            gnep_costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
        end
    else
        gnep_costs[index] = compute_realized_cost(res)
    end
end

for (index, res) in bilevel_results
    len = length(res)
    bilevel_steps[index] = len

    if is_trimming
        if len >= trim_steps
            bilevel_costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
        end
    else
        bilevel_costs[index] = compute_realized_cost(res)
    end
end

#@info "Average sp time steps: $(mean(values(sp_steps)))"
#@info "Average gnep time steps: $(mean(values(gnep_steps)))"
#@info "Average bilevel time steps: $(mean(values(bilevel_steps)))"


#bins = 1:10:time_steps+1
#histogram1 = histogram(collect(values(sp_steps)), bins=bins, label="sp")
#histogram2 = histogram(collect(values(gnep_steps)), bins=bins, label="gnep")
#histogram3 = histogram(collect(values(bilevel_steps)), bins=bins, xlabel="# steps", label="bilevel")
#Plots.plot(histogram1, histogram2, histogram3, layout=(3, 1), legend=true, ylabel="# x0")

#Plots.plot(a_cost_breakdown.running.lane)
#Plots.plot!(b_cost_breakdown.running.lane)

# find common runs
#common_runs = intersect(keys(gnep_costs), keys(bilevel_costs))
#gnep_costs_com = Dict(i => gnep_costs_tr[i] for i in common_runs)
#bilevel_costs_com = Dict(i => bilevel_costs_tr[i] for i in common_runs)
#plot_x0s(Dict(i => x0s[i] for i in common_runs))

#collect(x0s[common_runs])

#Plots.scatter(x0s[:, 1], x0_arr[:, 2], aspect_ratio=:equal, legend=false)
#Plots.scatter!(x0_arr[:, 5], x0_arr[:, 6], aspect_ratio=:equal, legend=false)
#plot_running_costs(sp_costs; T=trim_steps, is_cumulative=false, alpha=.2, sup_title="sp running costs")
#plot_running_costs(gnep_costs; T=trim_steps, is_cumulative=false, alpha=.1, sup_title="all gnep running costs")
#plot_running_costs(bilevel_costs; T=trim_steps, is_cumulative=false, alpha=.1, sup_title="all bilevel running costs")
#plot_running_costs(sp_costs; T=trim_steps, is_cumulative=true, alpha=.1, sup_title="sp cumulative running costs")
#plot_running_costs(gnep_costs; T=trim_steps, is_cumulative=true, alpha=.1, sup_title="gnep cumulative running costs")
#plot_running_costs(bilevel_costs; T=trim_steps, is_cumulative=true, alpha=.1, sup_title="bilevel cumulative running costs")

# this does't work right now
#gnep_costs_com_arr = extract_costs(gnep_costs_tr, common_runs)
#bilevel_costs_com_arr = extract_costs(bilevel_costs_tr, common_runs)
#all_costs = (ind=common_runs, gnep=gnep_costs_com_arr, bilevel=bilevel_costs_com_arr) 

all_costs = extract_intersected_costs(gnep_costs, gnep_costs, bilevel_costs)
@info "Until $(trim_steps) time steps (n=$(length(all_costs.gnep.total)))"
@info "total Δcost = bilevel - gnep"
Δcost_total = compute_Δcost(all_costs.gnep.total, all_costs.bilevel.total)
print_mean_min_max(Δcost_total)

@info "lane Δcost = bilevel - gnep"
Δcost_lane = compute_Δcost(all_costs.gnep.lane, all_costs.bilevel.lane)
print_mean_min_max(Δcost_lane)

@info "control Δcost = bilevel - gnep"
Δcost_control = compute_Δcost(all_costs.gnep.control, all_costs.bilevel.control)
print_mean_min_max(Δcost_control)

@info "velocity Δcost = bilevel - gnep"
Δcost_velocity = compute_Δcost(all_costs.gnep.velocity, all_costs.bilevel.velocity)
print_mean_min_max(Δcost_velocity)

@info "terminal Δcost = bilevel - gnep"
Δcost_terminal = compute_Δcost(all_costs.gnep.terminal, all_costs.bilevel.terminal)
#print_mean_min_max(Δcost_terminal)

# best
P1_most_bilevel_adv_ind = all_costs.ind[argmax(Δcost_total.P1_abs)]
P2_most_bilevel_adv_ind = all_costs.ind[argmax(Δcost_total.P2_abs)]
#@assert(best_ind_P1 == best_ind_P2)
## worst
P1_most_gnep_adv_ind = all_costs.ind[argmin(Δcost_total.P1_abs)]
P2_most_gnep_adv_ind = all_costs.ind[argmin(Δcost_total.P2_abs)]
#@assert(worst_ind_P1 == worst_ind_P2)

#animate(probs, gnep_results[P1_most_bilevel_adv_ind]; save=true, filename="P1_most_bilevel_advantage.mp4");
#animate(probs, bilevel_results[P2_most_bilevel_adv_ind]; save=true, filename="P2_most_bilevel_advantage.mp4");
#animate(probs, gnep_results[P1_most_gnep_adv_ind]; save=true, filename="P1_most_gnep_advantage.mp4");
#animate(probs, bilevel_results[P2_most_gnep_adv_ind]; save=true, filename="P2_most_gnep_advantage.mp4");

#res = gnep_results[worst_ind_P1];
#res = bilevel_results[worst_ind_P1];
#prefs = zeros(Int, length(res));
#for key in keys(res)
#    #println("Key: $key, Pref: $(sim_results[key].lowest_preference)")
#	prefs[key] = res[key].lowest_preference;
#end
#histogram(prefs, bins=1:9, xlabel="Type", ylabel="Frequency")