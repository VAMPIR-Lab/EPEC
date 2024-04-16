using EPEC
using JLD2
using Infiltrator 
include("postprocess_helpers.jl")

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    α3=1e-1,
    β=1e-1, #.5, # sensitive to high values
    cd=0.2, #0.25,
    d=2.0, # actual road width (±)
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
    );

data_dir = "data"
x0s_filename = "x0s_100samples_2024-04-16_0001"
results_suffix = "_(x0s_100samples_2024-04-16_0001)_2024-04-16_0001_50steps";
init_file = jldopen("$(data_dir)/$(x0s_filename).jld2", "r")
x0s = init_file["x0s"];

modes = 1:10
results = Dict()

for i in modes
    file = jldopen("$(data_dir)/results_mode$(i)$(results_suffix).jld2", "r")
    results[i] = process(file["results"])
end

include("print_cost_table.jl")
include("plot_boxplot.jl")
include("plot_running_cost.jl")

# visualize
#road = Dict(0 => 0, 2 => 0, 4=>-.2, 6=>-.5, 8=>-.9, 10=>-1.7, 12=>-2.4, 14=>-2.4, 16=>-1.6, 18=>-.7, 20=>.7, 22=>1.6, 24=>2.4, 26=>2.4, 28=>1.6, 30=>.9,32=>.5,34=>.2,36=>0,38=>0, 40=>-.01);
#mode = 9;
#x0 = x0s[1];
#sim_results = solve_simulation(probs, 50; x0, road, mode);
#EPEC.animate(probs, sim_results; save=false, mode, road);

