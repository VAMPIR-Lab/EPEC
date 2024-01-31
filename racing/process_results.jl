using EPEC
using GLMakie
using JLD2
using Plots

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

data_dir = "data"
x0s_filename = "x0s_50samples_2024-01-30_1646"
results_suffix = "_(x0s_50samples_2024-01-30_1646)_2024-01-30_1646_50steps";
init_file = jldopen("$(data_dir)/$(x0s_filename).jld2", "r")
x0s = init_file["x0s"]

function process(results; is_trimming=false, trim_steps=50)
	costs = Dict()
	steps = Dict()

	for (index, res) in results
		len = length(res)
		steps[index] = len
	
		if is_trimming
			if len >= trim_steps
				#costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
				costs[index] = compute_realized_cost(res)
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

indices = results[1].costs |> keys |> collect |> sort
b_steps_basis = mean([results[1].steps[i] for i in indices])
b_total_costs_basis = mean([results[1].costs[i].b.final.total for i in indices])
modes_sorted = sort(collect(keys(results)))

println("		mean (±95% CI)		std	min		max")
function print_mean_etc(vals; title="", header_only=false)
	CI = 1.96*std(vals)/sqrt(length(vals));
	m = mean(vals);
	m95l = m - CI;
	m95u = m + CI; 
	s = std(vals)

	println("$(title)	$(round(m; sigdigits=5)) (±$(round(CI; sigdigits=5)))	$(round(s; sigdigits=5))	$(round(minimum(vals); sigdigits=5))	$(round(maximum(vals); sigdigits=5))")
end

@info "steps mean CI min max"
for mode in modes_sorted
	res = results[mode]
	inds = res.costs |> keys |> collect |> sort
	b_steps = [res.steps[i] for i in inds]
	print_mean_min_max(b_steps; title="mode $(mode) steps")
end

@info "total cost mean CI min max"
cost_table_old = Dict()

for mode in modes_sorted
	res = results[mode]
	inds = res.costs |> keys |> collect |> sort
	b_steps = [res.steps[i] for i in inds]
	b_total_costs = [res.costs[i].b.final.total for i in inds]
	a_steps = [res.steps[i] for i in inds]
	a_total_costs = [res.costs[i].a.final.total for i in inds]
	cost_table_old[mode, "a"] = a_total_costs ./ a_steps;
	cost_table_old[mode, "b"] = b_total_costs ./ b_steps;
	#print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
	#print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
end


cost_table = Dict()
cost_table["S", "S"] = cost_table_old[1, "a"] 
cost_table["S", "N"] = cost_table_old[2, "a"] 
cost_table["S", "L"] = cost_table_old[4, "a"] 
cost_table["S", "F"] = cost_table_old[7, "a"] 
cost_table["N", "S"] = cost_table_old[2, "b"] 
cost_table["N", "N"] = cost_table_old[3, "a"] 
cost_table["N", "L"] = cost_table_old[5, "a"] 
cost_table["N", "F"] = cost_table_old[8, "a"] 
cost_table["L", "S"] = cost_table_old[4, "b"] 
cost_table["L", "N"] = cost_table_old[5, "b"] 
cost_table["L", "L"] = cost_table_old[6, "a"] 
cost_table["L", "F"] = cost_table_old[9, "a"] 
cost_table["F", "S"] = cost_table_old[7, "b"] 
cost_table["F", "N"] = cost_table_old[8, "b"] 
cost_table["F", "L"] = cost_table_old[9, "b"] 
cost_table["F", "F"] = cost_table_old[10, "a"] 
#display(cost_table)

for (k,v) in cost_table
	print_mean_min_max(v; title=k,scale=100)
end

cost_table_compressed = Dict()
for strat in ["S", "N", "L", "F"]
	cost_table_compressed[strat] = cost_table[strat, "S"] +  cost_table[strat, "N"] + cost_table[strat, "F"] + cost_table[strat, "L"]
end

for (k,v) in cost_table_compressed
	print_mean_min_max(v; title=k,scale=100)
end

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