#todo realized cost DONE
#todo randomized initial conditions DONE
#compare: sp, gnep, bilevel (shared brain)
#if it works, also compare bilevel (distributed brain)

using EPEC
using GLMakie
using Plots
using Dates
using JLD2

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
init_filename = "results_x0s_1000samples_2024-01-25_2315.jld2_2024-01-26_0125_100steps.jld2";
results_filename = "results_x0s_1000samples_2024-01-25_2315.jld2_2024-01-26_0125_100steps.jld2";
sample_size = 2;
time_steps = 10;
r_offset_max = 3.0; # maximum distance between P1 and P2
long_vel_max = 3.0; # maximum longitudunal velocity
lat_max = probs.params.lat_max;
r_offset_min = probs.params.r;

x0s = Dict{Int,Vector{Float64}}()

if (is_x0s_from_file)
    # WARNING params not loaded from file
    init_file = jldopen("$(data_dir)/$(init_filename)", "r")
    x0s = init_file["x0s"]
    #@infiltrate
    #Plots.scatter(x0_arr[:, 1], x0_arr[:, 2], aspect_ratio=:equal, legend=false)
    #Plots.scatter!(x0_arr[:, 5], x0_arr[:, 6], aspect_ratio=:equal, legend=false)
else
    x0s = generate_x0s(sample_size, lat_max, r_offset_min, r_offset_max, long_vel_max)
    jldsave("$(data_dir)/x0s_$(sample_size)samples_$(Dates.format(now(),"YYYY-mm-dd_HHMM")).jld2"; x0s, lat_max, r_offset_min, r_offset_max, long_vel_max)
end

gnep_results = []
bilevel_results = []
all_costs = []

if is_results_from_file
    results_file = jldopen("$(data_dir)/$(results_filename)", "r")
    gnep_results = results_file["gnep_results"]
    bilevel_results = results_file["bilevel_results"]
    all_costs = extract_costs(results_file["gnep_costs"], results_file["bilevel_costs"])
else
    gnep_results = Dict()
    bilevel_results = Dict()
    gnep_costs = Dict()
    bilevel_costs = Dict()

    start = time()
    # how to multithread?
    #using Threads
    #num_threads = Threads.nthreads()
    for (index, x0) in x0s
        @info "Solving $index: $x0"

        try
            gnep_res = solve_simulation(probs, time_steps; x0, only_want_gnep=true)
            costs = compute_realized_cost(gnep_res)
            gnep_results[index] = gnep_res
            gnep_costs[index] = costs
        catch err
            @info "gnep failed $index: $x0"
            println(err)
        end

        try
            bilevel_res = solve_simulation(probs, time_steps; x0, only_want_gnep=false)
            costs = compute_realized_cost(bilevel_res)

            bilevel_results[index] = bilevel_res
            bilevel_costs[index] = costs
        catch err
            @info "bilevel failed $index: $x0"
            println(err)
        end
    end
    elapsed = time() - start

    # save
    if is_x0s_from_file
        jldsave("$(data_dir)/results_$(init_filename)_$(Dates.format(now(),"YYYY-mm-dd_HHMM"))_$(time_steps)steps.jld2"; params=probs.params, x0s, gnep_results, gnep_costs, bilevel_results, bilevel_costs, elapsed)
    else
        jldsave("$(data_dir)/results_$(Dates.format(now(),"YYYY-mm-dd_HHMM")).jld2"; params=probs.params, x0s, gnep_results, gnep_costs, bilevel_results, bilevel_costs, elapsed)
    end

    all_costs = extract_costs(gnep_costs, bilevel_costs)
end

@info "bilevel vs gnep total cost"
print_mean_min_max(all_costs.gnep.total, all_costs.bilevel.total)

#i = rand(1:sample_size); @info i; animate(probs, bilevel_results[i]; save=false);

#@info "bilevel vs gnep lane cost"
#print_mean_min_max(all_costs.gnep.lane, all_costs.bilevel.lane)

#@info "bilevel vs gnep control cost"
#print_mean_min_max(all_costs.gnep.control, all_costs.bilevel.control)

#@info "bilevel vs gnep terminal cost"
#print_mean_min_max(all_costs.gnep.terminal, all_costs.bilevel.terminal)