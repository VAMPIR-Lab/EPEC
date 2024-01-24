using EPEC
using GLMakie
using Plots
include("../../examples/racing.jl")

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-2,
    α2=1e-4,
    β=1e0,
    cd=0.2,
    u_max_nominal = 1.0,
    u_max_drafting=2.0,
    box_length=5.0,
    box_width=2.0,
    lat_max=1.5
);

x0 = [1.0, 3, 0, 1, -1, 2, 0, 1.5]

@info "---  GNEP SIM ---"
sim_results_gnep = solve_simulation(probs, 200; x0, only_want_gnep=true);
@info "---  BILEVEL SIM---  "
sim_results_bilevel = solve_simulation(probs, 200; x0, only_want_gnep=false);

animate(probs, sim_results_gnep; save=true, filename="results/racing1_gnep.mp4");
animate(probs, sim_results_bilevel; save=true, filename="results/racing1_bilevel.mp4");

prefs_gnep = zeros(Int, length(sim_results_gnep))
for key in keys(prefs_gnep)
    prefs_gnep[key] = sim_results_gnep[key].lowest_preference
end
prefs_bilevel = zeros(Int, length(sim_results_bilevel))
for key in keys(prefs_bilevel)
    prefs_bilevel[key] = sim_results_bilevel[key].lowest_preference
end

Plots.plot(
	histogram(prefs_gnep, bins=1:9, xlabel="Type", ylabel="Frequency", title="gnep"),
	histogram(prefs_bilevel, bins=1:9, xlabel="Type", ylabel="Frequency", title="bilevel"),
	legend=false
)