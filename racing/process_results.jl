results_file = jldopen("$(data_dir)/$(results_filename).jld2", "r")
data_dir = "data"
res1_filename = "results_1_x0s_10samples_2024-01-29_1516_2024-01-29_1517_100steps.jld2";
res2_filename = "results_2_x0s_10samples_2024-01-29_1516_2024-01-29_1518_100steps.jld2";

res1_file = jldopen("$(data_dir)/$(res1_filename)", "r")
res2_file = jldopen("$(data_dir)/$(res2_filename)", "r")
res1 = res1_file["results"]
res2 = res2_file["results"]

function process(results; is_trimming=false, trim_steps=100)
	costs = Dict()
	steps = Dict()

	for (index, res) in results
		len = length(res)
		steps[index] = len
	
		if is_trimming
			if len >= trim_steps
				costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
			end
		else
			costs[index] = compute_realized_cost(res)
		end
	end
	(; costs, steps)
end

res1_processed = process(res1)
res2_processed = process(res2)

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