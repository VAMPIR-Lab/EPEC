using EPEC
using LaTeXStrings
using Plots
using GLMakie
using JLD2
using LaTeXStrings
using Infiltrator
using Statistics
using StatsPlots

#include("racing.jl")
include("random_racing_helper.jl")

#probs = setup(; T=10,
#    Δt=0.1,
#    r=1.0,
#    α1=1e-3,
#    α2=1e-4,
#    α3=1e-1,
#    β=1e-1, #.5, # sensitive to high values
#    cd=0.2, #0.25,
#    d=1.5, # actual road width (±)
#    u_max_nominal=1.0,
#    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
#    box_length=5.0,
#    box_width=2.0,
#    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
#    );

data_dir = "data"
x0s_filename = "x0s_100samples_2024-04-16_0001"
results_suffix = "_(x0s_100samples_2024-04-16_0001)_2024-04-16_0001_50steps";
init_file = jldopen("$(data_dir)/$(x0s_filename).jld2", "r")
x0s = init_file["x0s"]

xdim=4
udim=2
#x0s_inds = [1, 100, 1000, 2000]
#p = Plots.plot(layout=(2,2))
#plots = []
#rad = 1
#lat = 3.0
#ymax = 2.0
#circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
#circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]

#for x0 in [x0s[i] for i in x0s_inds]
#    p = Plots.plot()
#    x, y, u, v = x0[1:4]
#    circ_x_shifted_A = circ_x .+ x
#    circ_y_shifted_A = circ_y .+ y
#    Plots.plot!(circ_x_shifted_A, circ_y_shifted_A, line=:path, color=:blue, label="")
#    Plots.quiver!([x], [y], quiver=([u], [v]), aspect_ratio=:equal, axis=([], false), color=:blue, label="", linewidth=.1)

#    x, y, u, v = x0[5:8]
#    circ_x_shifted_B = circ_x .+ x
#    circ_y_shifted_B = circ_y .+ y
#    Plots.plot!(circ_x_shifted_B, circ_y_shifted_B, line=:path, color=:red, label="")
#    Plots.quiver!([x], [y], quiver=([u], [v]), aspect_ratio=:equal, axis=([], false), color=:red, label="", linewidth=.1)
#    Plots.plot!([-lat, -lat], [-ymax, ymax], color=:black, label="")
#    Plots.plot!([+lat, +lat], [-ymax, ymax], color=:black, label="")
#    push!(plots, p)
#end

#Plots.plot(plots..., margin=1e-3*Plots.mm)

function process(results; is_trimming=false, trim_steps=100)
    costs = Dict()
    steps = Dict()

    for (index, res) in results
        len = length(res)
        steps[index] = len

        if is_trimming
            if len >= trim_steps
                #costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
                costs[index] = compute_realized_cost(res)
                #steps[index] = len
            end
        else
            costs[index] = compute_realized_cost(res)
        end
    end
    (; costs, steps)
end

modes = 1:10
results = Dict()

for i in modes
    file = jldopen("$(data_dir)/results_mode$(i)$(results_suffix).jld2", "r")
    results[i] = process(file["results"])
end

#pa_comp = Plots.plot()
#pb_comp = Plots.plot()
#@infiltrate

function get_mean_running_cost(results, i; T = 50)
    vals = [ Float64[] for _ in 1:T]
    for (index, c) in results[i].costs
        T = length(c.a.running.total)
        for t in 1:T
            push!(vals[t], c.a.running.total[t])# + c.a.running.velocity[t])
            # CI = 1.96*std(vals)/sqrt(length(vals));
        end
    end
    avgs = map(vals) do val
        mean(val)
    end
    stderrs = map(vals) do val
        1.96*std(val) / sqrt(length(val))
    end
    (avgs, stderrs)
end

avgs_1, stderrs_1 = get_mean_running_cost(results, 1)
avgs_3, stderrs_3 = get_mean_running_cost(results, 3)
avgs_9, stderrs_9 = get_mean_running_cost(results, 9)
avgs_6, stderrs_6 = get_mean_running_cost(results, 6)
avgs_10, stderrs_10 = get_mean_running_cost(results, 10)

#, yaxis=(formatter=y->string(round(Int, y / 10^-4)))
#, yaxis=(formatter=y->round(y; sigdigits=4)

#Plots.plot(layout=(2,1))

p = Plots.plot(avgs_3, ribbon = stderrs_3, fillalpha = 0.3, linewidth=3, label = "Nash competition (N-N)")
Plots.plot!(p, avgs_9, ribbon = stderrs_9, fillalpha = 0.3, linewidth=3, label = "Bilevel competition (L-F)")
annotate!([(3, 8.5e-3, Plots.text(L"\times10^{-3}", 12, :black, :center))])
Plots.plot!(p, size=(500,400), xlabel="Simulation steps", ylabel="Mean running cost per time step", yaxis=(formatter=y->round(y*1e3; sigdigits=4)))
savefig("./figures/plot_3_v_9_running_cost.pdf")

#p = Plots.plot(avgs_6, ribbon = stderrs_6, fillalpha = 0.3, color=:blue, linewidth=3, label = "Nash equilibrium (F-F)")
#Plots.plot!(p, avgs_10, ribbon = stderrs_10, fillalpha = 0.3, color=:red, linewidth=3, label = "Bilevel (L-L)")
#annotate!([(3, 8.5e-3, Plots.text(L"\times10^{-2}", 12, :black, :center))])
#Plots.plot!(p, size=(500,400), xlabel="Simulation steps", ylabel="Average total running cost", yaxis=(formatter=y->round(y*1e2; sigdigits=4)))
#savefig("./figures/plot_3_v_9_running_cost.pdf")

#Plots.plot!(avgs_1, ribbon = stderrs_1, fillalpha = 0.3, color=:blue, linewidth=3, label = "Nash equilibrium (N-N)")
#Plots.plot!(avgs_6, ribbon = stderrs_6, fillalpha = 0.3, color=:red, linewidth=3, label = "Bilevel (F-F)")
#Plots.plot!(avgs_10, ribbon = stderrs_10, fillalpha = 0.3, color=:red, linewidth=3, label = "Bilevel(L-L)")


#Plots.plot!(avgs2, ribbon = stderrs2, fillalpha = 0.3, color=:red, linewidth=3, label = "Nash equilibrium (N-N)")
#savefig("/figures/plot_1_v_3_running_cost.png")
#Plots.plot!(avgs2, color=:red, linewidth=5)
#Plots.plot!(avgs2 .+ stderrs2, color=:red, linewidth=2)
#Plots.plot!(avgs2 .- stderrs2, color=:red, linewidth=2)


#for (index, c) in results[3].costs
#    #@infiltrate
#    Plots.plot!(pa_comp, mean.(c.a.running.competitive + c.a.running.velocity), title="a competitive", label="")
#    Plots.plot!(pa_comp, mean.(c.b.running.competitive + c.b.running.velocity), title="b competitive", label="")
#end
#Plots.plot(pa_comp)
#indices = results[1].costs |> keys |> collect |> sort
#b_steps_basis = mean([results[1].steps[i] for i in indices])
#b_total_costs_basis = mean([results[1].costs[i].b.final.total for i in indices])

modes_sorted = sort(collect(keys(results)))

function process_steps(results, modes_sorted)
    steps_table_old = Dict()

    for mode in modes_sorted
        res = results[mode]
        inds = res.costs |> keys |> collect |> sort
		a_steps = [res.steps[i] for i in inds]
		b_steps = [res.steps[i] for i in inds]
        steps_table_old[mode, "a"] = a_steps
        steps_table_old[mode, "b"] = b_steps
    end

    full_steps_table = Dict()
    full_steps_table["S", "S"] = steps_table_old[1, "a"]
    full_steps_table["S", "N"] = steps_table_old[2, "a"]
    full_steps_table["S", "L"] = steps_table_old[4, "a"]
    full_steps_table["S", "F"] = steps_table_old[7, "a"]
    full_steps_table["N", "S"] = steps_table_old[2, "b"]
    full_steps_table["N", "N"] = steps_table_old[3, "a"]
    full_steps_table["N", "L"] = steps_table_old[5, "a"]
    full_steps_table["N", "F"] = steps_table_old[8, "a"]
    full_steps_table["L", "S"] = steps_table_old[4, "b"]
    full_steps_table["L", "N"] = steps_table_old[5, "b"]
    full_steps_table["L", "L"] = steps_table_old[6, "a"]
    full_steps_table["L", "F"] = steps_table_old[9, "a"]
    full_steps_table["F", "S"] = steps_table_old[7, "b"]
    full_steps_table["F", "N"] = steps_table_old[8, "b"]
    full_steps_table["F", "L"] = steps_table_old[9, "b"]
    full_steps_table["F", "F"] = steps_table_old[10, "a"]
    #display(cost_table)

    compressed_table = Dict()
    for strat in ["S", "N", "L", "F"]
        #@infiltrate
        compressed_table[strat] = (full_steps_table[strat, "S"] + full_steps_table[strat, "N"] + full_steps_table[strat, "F"] + full_steps_table[strat, "L"])/4
    end
	(;full=full_steps_table, compressed=compressed_table)
end

function process_comp_cost(results, modes_sorted)
    cost_table_old = Dict()

    for mode in modes_sorted
        res = results[mode]
        inds = res.costs |> keys |> collect |> sort
		a_steps = [res.steps[i] for i in inds]
		b_steps = [res.steps[i] for i in inds]
        a_costs = [res.costs[i].a.final.competitive + res.costs[i].a.final.velocity for i in inds]
        b_costs = [res.costs[i].b.final.competitive + res.costs[i].b.final.velocity  for i in inds]
        cost_table_old[mode, "a"] = a_costs ./ a_steps
        cost_table_old[mode, "b"] = b_costs ./ b_steps
    end

    full_table = Dict()
    full_table["S", "S"] = cost_table_old[1, "a"]
    full_table["S", "N"] = cost_table_old[2, "a"]
    full_table["S", "L"] = cost_table_old[4, "a"]
    full_table["S", "F"] = cost_table_old[7, "a"]
    full_table["N", "S"] = cost_table_old[2, "b"]
    full_table["N", "N"] = cost_table_old[3, "a"]
    full_table["N", "L"] = cost_table_old[5, "a"]
    full_table["N", "F"] = cost_table_old[8, "a"]
    full_table["L", "S"] = cost_table_old[4, "b"]
    full_table["L", "N"] = cost_table_old[5, "b"]
    full_table["L", "L"] = cost_table_old[6, "a"]
    full_table["L", "F"] = cost_table_old[9, "a"]
    full_table["F", "S"] = cost_table_old[7, "b"]
    full_table["F", "N"] = cost_table_old[8, "b"]
    full_table["F", "L"] = cost_table_old[9, "b"]
    full_table["F", "F"] = cost_table_old[10, "a"]
    #display(cost_table)

    compressed_table = Dict()
    for strat in ["S", "N", "L", "F"]
        compressed_table[strat] = full_table[strat, "S"] + full_table[strat, "N"] + full_table[strat, "F"] + full_table[strat, "L"]
    end
	(;full=full_table, compressed=compressed_table)
end

steps_table = process_steps(results, modes_sorted)
total_cost_table = process_costs(results, modes_sorted, property=:total)
lane_cost_table = process_costs(results, modes_sorted, property=:lane)
control_cost_table = process_costs(results, modes_sorted, property=:control)
velocity_cost_table = process_costs(results, modes_sorted, property=:velocity)
#comp_cost_table = process_costs(results, modes_sorted, property=:competitive)
#comp2_cost_table = process_comp_cost(results, modes_sorted)


#p = Plots.boxplot(["S" "N" "L" "F"], [total_cost_table.compressed["S"], total_cost_table.compressed["N"], total_cost_table.compressed["L"], total_cost_table.compressed["F"]], legend=false)

p = Plots.boxplot(["Nash competition, N-N" "Bilevel competition, L-F "], [
total_cost_table.full["N", "N"], 
total_cost_table.full["L", "F"]] 
#total_cost_table.full["F", "F"],
#total_cost_table.full["L", "L"]]
, legend=false, outliers=false)
annotate!([(.2, 9e-2, Plots.text(L"\times10^{-3}", 12, :black, :center))])
Plots.plot!(p, size=(500,400), xlabel="Competition type", ylabel="Mean running cost per time step", yaxis=(formatter=y->round(y*1e3; sigdigits=4)))
savefig("./figures/boxplot_running_cost.pdf")

println("		mean (±95% CI) [95% CI l, u]	std	min	max")

println("Steps:")
for (k, v) in steps_table.compressed
    print_mean_etc(v; title=k, scale=1)
end

println("Total:")
for (k, v) in total_cost_table.compressed
    print_mean_etc(v; title=k, scale=1000)
end

#println("Lane:")
#for (k, v) in lane_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Control:")
#for (k, v) in control_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Velocity:")
#for (k, v) in velocity_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Competitive:")
#for (k, v) in comp_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Combined competitive:")
#for (k, v) in comp2_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end


#println("Steps:")
#for (k, v) in steps_table.full
#    print_mean_etc(v; title=k, scale=1)
#end


println("Total:")
for (k, v) in total_cost_table.full
    print_mean_etc(v; title=k, scale=1000)
end

#println("Lane:")
#for (k, v) in lane_cost_table.full
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Control:")
#for (k, v) in control_cost_table.full
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Velocity:")
#for (k, v) in velocity_cost_table.full
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Competitive:")
#for (k, v) in comp_cost_table.full
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Combined competitive:")
#for (k, v) in comp2_cost_table.full
#    print_mean_etc(v; title=k, scale=100)
#end


#@info "terminal cost mean CI min max"
#for mode in modes_sorted
#	res = results[mode]
#	inds = res.costs |> keys |> collect |> sort
#	b_steps = [res.steps[i] for i in inds]
#	b_total_costs = [res.costs[i].b.final.terminal for i in inds]
#	a_steps = [res.steps[i] for i in inds]
#	a_total_costs = [res.costs[i].a.final.terminal for i in inds]
#	print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
#	print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
#end

#@info "velocity cost mean CI min max"
#for mode in modes_sorted
#	res = results[mode]
#	inds = res.costs |> keys |> collect |> sort
#	b_steps = [res.steps[i] for i in inds]
#	b_total_costs = [res.costs[i].b.final.velocity for i in inds]
#	a_steps = [res.steps[i] for i in inds]
#	a_total_costs = [res.costs[i].a.final.velocity for i in inds]
#	print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
#	print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
#end

#@info "control cost mean CI min max"
#for mode in modes_sorted
#	res = results[mode]
#	inds = res.costs |> keys |> collect |> sort
#	b_steps = [res.steps[i] for i in inds]
#	b_total_costs = [res.costs[i].b.final.control for i in inds]
#	a_steps = [res.steps[i] for i in inds]
#	a_total_costs = [res.costs[i].a.final.control for i in inds]
#	print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
#	print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
#end


#plot_x0s(x0s)
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

#all_costs = extract_intersected_costs(gnep_costs, gnep_costs, bilevel_costs)
#@info "Until $(trim_steps) time steps (n=$(length(all_costs.gnep.total)))"
#@info "total Δcost = bilevel - gnep"
#Δcost_total = compute_Δcost(all_costs.gnep.total, all_costs.bilevel.total)
#print_mean_min_max(Δcost_total)

#@info "lane Δcost = bilevel - gnep"
#Δcost_lane = compute_Δcost(all_costs.gnep.lane, all_costs.bilevel.lane)
#print_mean_min_max(Δcost_lane)

#@info "control Δcost = bilevel - gnep"
#Δcost_control = compute_Δcost(all_costs.gnep.control, all_costs.bilevel.control)
#print_mean_min_max(Δcost_control)

#@info "velocity Δcost = bilevel - gnep"
#Δcost_velocity = compute_Δcost(all_costs.gnep.velocity, all_costs.bilevel.velocity)
#print_mean_min_max(Δcost_velocity)

#@info "terminal Δcost = bilevel - gnep"
#Δcost_terminal = compute_Δcost(all_costs.gnep.terminal, all_costs.bilevel.terminal)
#print_mean_min_max(Δcost_terminal)

# best
#P1_most_bilevel_adv_ind = all_costs.ind[argmax(Δcost_total.P1_abs)]
#P2_most_bilevel_adv_ind = all_costs.ind[argmax(Δcost_total.P2_abs)]
#@assert(best_ind_P1 == best_ind_P2)
## worst
#P1_most_gnep_adv_ind = all_costs.ind[argmin(Δcost_total.P1_abs)]
#P2_most_gnep_adv_ind = all_costs.ind[argmin(Δcost_total.P2_abs)]
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