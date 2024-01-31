using Random
using Statistics

#const xdim = 4
#const udim = 2
# generate x0s
function generate_x0s(sample_size, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
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
    #Plots.scatter(a_pos0_arr[:, 1], a_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)
    #Plots.scatter!(b_pos0_arr[:, 1], b_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)

    # keep lateral velocity zero
    a_vel0_arr = hcat(zeros(sample_size), a_long_vel_max .* rand(MersenneTwister(), sample_size))
    b_vel0_arr = zeros(size(a_vel0_arr))
    # choose random velocity offset for 
    for i in 1:sample_size
        b_long_vel0_offset = -b_long_vel_delta_max + 2 * b_long_vel_delta_max .* rand(MersenneTwister())
        b_long_vel0 = a_vel0_arr[i, 2] + b_long_vel0_offset
        # reroll until b long vel is nonnegative
        while b_long_vel0 < 0
            b_long_vel0_offset = -b_long_vel_delta_max + 2 * b_long_vel_delta_max .* rand(MersenneTwister())
            b_long_vel0 = a_vel0_arr[i, 2] + b_long_vel0_offset
        end
        b_vel0_arr[i, 2] = b_long_vel0
    end

    x0_arr = hcat(a_pos0_arr, a_vel0_arr, b_pos0_arr, b_vel0_arr)
    #@infiltrate
    x0s = Dict()

    for (index, row) in enumerate(eachrow(x0_arr))
        x0s[index] = row
    end
    x0s
end

# detailed cost
function f1_breakdown(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    lane_cost_arr = zeros(T)
    control_cost_arr = zeros(T)
    velocity_cost_arr = zeros(T)
    comp_cost_arr = zeros(T)

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        lane_cost_arr[t] = α1 * xa[1]^2
        control_cost_arr[t] = α2 * ua' * ua
        velocity_cost_arr[t] = -α3 * xa[4]
        comp_cost_arr[t] = β * xb[4]
    end

    lane_cost = sum(lane_cost_arr)
    control_cost = sum(control_cost_arr)
    velocity_cost = sum(velocity_cost_arr)
    comp_cost = sum(comp_cost_arr)
    total_cost = lane_cost + control_cost + velocity_cost + comp_cost
    total_cost_arr = lane_cost_arr .+ control_cost_arr .+ velocity_cost_arr .+ comp_cost_arr
    final = (; total=total_cost, lane=lane_cost, control=control_cost, velocity=velocity_cost, competitive=comp_cost)
    running = (; total=total_cost_arr, lane=lane_cost_arr, control=control_cost_arr, velocity=velocity_cost_arr, competitive=comp_cost_arr)
    (; final, running)
end

function f2_breakdown(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    lane_cost_arr = zeros(T)
    control_cost_arr = zeros(T)
    velocity_cost_arr = zeros(T)
    comp_cost = zeros(T)

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        lane_cost_arr[t] = α1 * xb[1]^2
        control_cost_arr[t] = α2 * ub' * ub
        velocity_cost_arr[t] = -α3 * xb[4]
        comp_cost[t] = β * xa[4]
    end
    lane_cost = sum(lane_cost_arr)
    control_cost = sum(control_cost_arr)
    velocity_cost = sum(velocity_cost_arr)
    comp_cost = sum(comp_cost)
    total_cost = lane_cost + control_cost + velocity_cost + comp_cost
    total_cost_arr = lane_cost_arr .+ control_cost_arr .+ velocity_cost_arr .+ comp_cost
    final = (; total=total_cost, lane=lane_cost, control=control_cost, velocity=velocity_cost, competitive=comp_cost)
    running = (; total=total_cost_arr, lane=lane_cost_arr, control=control_cost_arr, velocity=velocity_cost_arr, competitive=comp_cost)
    (; final, running)
end

function compute_realized_cost(res)
    T = length(res)
    Xa = zeros(xdim * T)
    Ua = zeros(udim * T)
    Xb = zeros(xdim * T)
    Ub = zeros(udim * T)

    for t in eachindex(res)
        Xa[xdim*(t-1)+1:xdim*t] = res[t].x0[1:4]
        Ua[udim*(t-1)+1:udim*t] = res[t].U1[1, :]
        Xb[xdim*(t-1)+1:xdim*t] = res[t].x0[5:8]
        Ub[udim*(t-1)+1:udim*t] = res[t].U2[1, :]
    end
    Z = [Xa; Ua; Xb; Ub; zeros(8)] # making it work with f1(Z) and f2(Z)

    a_cost = probs.OP1.f(Z)
    b_cost = probs.OP2.f(Z)
    a_breakdown = f1_breakdown(Z; probs.params.α1, probs.params.α2, probs.params.α3, probs.params.β)
    b_breakdown = f2_breakdown(Z; probs.params.α1, probs.params.α2, probs.params.α3, probs.params.β)
    #a_breakdown = f1_breakdown(Z; α1=1., α2=1., α3=1., β=1.)
    #b_breakdown = f2_breakdown(Z; α1=1., α2=1., α3=1., β=1.)
    # breakdowns were copy pasted so
    #@assert(isapprox(a_cost, a_breakdown.final.total))
    #@assert(isapprox(b_cost, b_breakdown.final.total))
    (; a=a_breakdown, b=b_breakdown)
end

# statistics
function trim_by_steps(costs, steps; min_steps=10)
    trimmed_costs = Dict()

    for (index, cost) in costs
        if steps[index] >= min_steps
            trimmed_costs[index] = cost
        end
    end
    return trimmed_costs
end

function extract_intersected_costs(sp_costs, gnep_costs, bilevel_costs)
    index_arr = []
    sp_cost_arr = []
    sp_lane_cost_arr = []
    sp_control_cost_arr = []
    sp_velocity_cost_arr = []
    sp_comp_cost_arr = []
    gnep_cost_arr = []
    gnep_lane_cost_arr = []
    gnep_control_cost_arr = []
    gnep_velocity_cost_arr = []
    gnep_comp_cost_arr = []
    bilevel_cost_arr = []
    bilevel_lane_cost_arr = []
    bilevel_control_cost_arr = []
    bilevel_velocity_cost_arr = []
    bilevel_comp_cost_arr = []

    for (index, sp_cost) in sp_costs
        if haskey(gnep_costs, index) && haskey(bilevel_costs, index)
            if haskey(bilevel_costs, index)
                push!(index_arr, index)
                push!(sp_cost_arr, [sp_cost.a.final.total, sp_cost.b.final.total])
                push!(sp_lane_cost_arr, [sp_cost.a.final.lane, sp_cost.b.final.lane])
                push!(sp_control_cost_arr, [sp_cost.a.final.control, sp_cost.b.final.control])
                push!(sp_velocity_cost_arr, [sp_cost.a.final.velocity, sp_cost.b.final.velocity])
                push!(sp_comp_cost_arr, [sp_cost.a.final.competitive, sp_cost.b.final.competitive])
                push!(gnep_cost_arr, [gnep_costs[index].a.final.total, gnep_costs[index].b.final.total])
                push!(gnep_lane_cost_arr, [gnep_costs[index].a.final.lane, gnep_costs[index].b.final.lane])
                push!(gnep_control_cost_arr, [gnep_costs[index].a.final.control, gnep_costs[index].b.final.control])
                push!(gnep_velocity_cost_arr, [gnep_costs[index].a.final.velocity, gnep_costs[index].b.final.velocity])
                push!(gnep_comp_cost_arr, [gnep_costs[index].a.final.competitive, gnep_costs[index].b.final.competitive])
                push!(bilevel_cost_arr, [bilevel_costs[index].a.final.total, bilevel_costs[index].b.final.total])
                push!(bilevel_lane_cost_arr, [bilevel_costs[index].a.final.lane, bilevel_costs[index].b.final.lane])
                push!(bilevel_control_cost_arr, [bilevel_costs[index].a.final.control, bilevel_costs[index].b.final.control])
                push!(bilevel_velocity_cost_arr, [bilevel_costs[index].a.final.velocity, bilevel_costs[index].b.final.velocity])
                push!(bilevel_comp_cost_arr, [bilevel_costs[index].a.final.competitive, bilevel_costs[index].b.final.competitive])
            end
        end
    end
    sp_extracted = (total=sp_cost_arr, lane=sp_lane_cost_arr, control=sp_control_cost_arr, velocity=sp_velocity_cost_arr, comp=spcomp_cost_arr)
    gnep_extracted = (total=gnep_cost_arr, lane=gnep_lane_cost_arr, control=gnep_control_cost_arr, velocity=gnep_velocity_cost_arr, comp=gnep_comp_cost_arr)
    bilevel_extracted = (total=bilevel_cost_arr, lane=bilevel_lane_cost_arr, control=bilevel_control_cost_arr, velocity=bilevel_velocity_cost_arr, comp=bilevel_comp_cost_arr)
    cost = (ind=index_arr, sp=sp_extracted, gnep=gnep_extracted, bilevel=bilevel_extracted)
    cost
end

function compute_Δcost(baseline_cost, other_cost)
    P1_baseline_cost = [v[1] for v in baseline_cost]
    P1_other_cost = [v[1] for v in other_cost]
    P2_baseline_cost = [v[2] for v in baseline_cost]
    P2_other_cost = [v[2] for v in other_cost]

    # bilevel vs gnep
    P1_abs_Δcost = P1_other_cost .- P1_baseline_cost
    P2_abs_Δcost = P2_other_cost .- P2_baseline_cost
    P1_rel_Δcost = P1_abs_Δcost ./ abs.(P1_baseline_cost)
    P2_rel_Δcost = P2_abs_Δcost ./ abs.(P2_baseline_cost)

    Δcost = (; P1_abs=P1_abs_Δcost, P2_abs=P2_abs_Δcost, P1_rel=P1_rel_Δcost, P2_rel=P2_rel_Δcost)
    Δcost
end

function process_costs(results, modes_sorted; property=:total)
    cost_table_old = Dict()

    for mode in modes_sorted
        res = results[mode]
        inds = res.costs |> keys |> collect |> sort
		a_steps = [res.steps[i] for i in inds]
		b_steps = [res.steps[i] for i in inds]
        a_costs = [getindex(res.costs[i].a.final, property) for i in inds]
        b_costs = [getindex(res.costs[i].b.final, property)  for i in inds]
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


function print_mean_etc(vals; title="", scale=1.0, sigdigits = 4)
    vals = vals.*scale
	CI = 1.96*std(vals)/sqrt(length(vals));
	m = mean(vals);
	m95l = m - CI;
	m95u = m + CI; 
	s = std(vals)

	println("$(title)	$(round(m; sigdigits)) (±$(round(CI; sigdigits))) [$(round(m95l; sigdigits)), $(round(m95u; sigdigits))]	$(round(s; sigdigits))	$(round(minimum(vals); sigdigits))	$(round(maximum(vals); sigdigits))")
end

function print_mean_min_max(Δcost)
    println("		mean		stderrmin			max")
    println("P1 Δcost abs :  $(mean(Δcost.P1_abs))  $(std(Δcost.P1_abs)/sqrt(length(Δcost.P1_abs)))  $(minimum(Δcost.P1_abs))  $(maximum(Δcost.P1_abs))")
    println("P2 Δcost abs :  $(mean(Δcost.P2_abs))  $(std(Δcost.P2_abs)/sqrt(length(Δcost.P2_abs)))  $(minimum(Δcost.P2_abs))  $(maximum(Δcost.P2_abs))")
    println("P1 Δcost rel%:  $(mean(Δcost.P1_rel) * 100)  $(std(Δcost.P1_rel)/sqrt(length(Δcost.P1_rel)) * 100)  $(minimum(Δcost.P1_rel) * 100)  $(maximum(Δcost.P1_rel) * 100)")
    println("P2 Δcost rel%:  $(mean(Δcost.P2_rel) * 100)  $(std(Δcost.P2_rel)/sqrt(length(Δcost.P2_rel)) * 100)  $(minimum(Δcost.P2_rel) * 100)  $(maximum(Δcost.P2_rel) * 100)")
end

function plot_running_costs(costs; T=10, is_cumulative=false, sup_title="", alpha=0.2)
    pa_lane = Plots.plot()
    pb_lane = Plots.plot()
    pa_control = Plots.plot()
    pb_control = Plots.plot()
    pa_velocity = Plots.plot()
    pb_velocity = Plots.plot()
    pa_terminal = Plots.plot()
    pb_terminal = Plots.plot()
    pa_total = Plots.plot()
    pb_total = Plots.plot()

    for (index, c) in costs
        idx = 1:min(T, length(c.a.running.lane))

        if is_cumulative
            Plots.plot!(pa_lane, alpha=alpha, cumsum(c.a.running.lane[idx]), title="a lane", label="")
            Plots.plot!(pb_lane, alpha=alpha, cumsum(c.b.running.lane[idx]), title="b lane", label="")
            Plots.plot!(pa_control, alpha=alpha, cumsum(c.a.running.control[idx]), title="a control", label="")
            Plots.plot!(pb_control, alpha=alpha, cumsum(c.b.running.control[idx]), title="b control", label="")
            Plots.plot!(pa_velocity, alpha=alpha, cumsum(c.a.running.velocity[idx]), title="a velocity", label="")
            Plots.plot!(pb_velocity, alpha=alpha, cumsum(c.b.running.velocity[idx]), title="b velocity", label="")
            Plots.plot!(pa_terminal, alpha=alpha, cumsum(c.a.running.terminal[idx]), title="a terminal", label="")
            Plots.plot!(pb_terminal, alpha=alpha, cumsum(c.b.running.terminal[idx]), title="b terminal", label="")
            Plots.plot!(pa_total, alpha=alpha, cumsum(c.a.running.total[idx]), title="a total", label="")
            Plots.plot!(pb_total, alpha=alpha, cumsum(c.b.running.total[idx]), title="b total", label="")
        else
            Plots.plot!(pa_lane, alpha=alpha, c.a.running.lane[idx], title="a lane", label="")
            Plots.plot!(pb_lane, alpha=alpha, c.b.running.lane[idx], title="b lane", label="")
            Plots.plot!(pa_control, alpha=alpha, c.a.running.control[idx], title="a control", label="")
            Plots.plot!(pb_control, alpha=alpha, c.b.running.control[idx], title="b control", label="")
            Plots.plot!(pa_velocity, alpha=alpha, c.a.running.velocity[idx], title="a velocity", label="")
            Plots.plot!(pb_velocity, alpha=alpha, c.b.running.velocity[idx], title="b velocity", label="")
            Plots.plot!(pa_terminal, alpha=alpha, c.a.running.terminal[idx], title="a terminal", label="")
            Plots.plot!(pb_terminal, alpha=alpha, c.b.running.terminal[idx], title="b terminal", label="")
            Plots.plot!(pa_total, alpha=alpha, c.a.running.total[idx], title="a total", label="")
            Plots.plot!(pb_total, alpha=alpha, c.b.running.total[idx], title="b total", label="")
        end
    end

    #Plots.plot(pa_lane, pa_control, pa_velocity, pa_terminal, pa_total, pb_lane, pb_control, pb_velocity, pb_terminal, pb_total, layout=(2, 5))
    #Plots.plot!(title)

    title = Plots.plot(title="$(sup_title)", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    Plots.plot(title, pa_lane, pa_control, pa_velocity, pa_terminal, pa_total, pb_lane, pb_control, pb_velocity, pb_terminal, pb_total, layout=@layout([A{0.01h}; (2, 5)]))
end

function plot_x0s(data_dict; lat=2.0, ymax=3.0, rad=0.5)
    plots = []
    circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]

    for (key, values) in data_dict
        p = Plots.plot()
        x, y, u, v = values[1:4]
        circ_x_shifted_A = circ_x .+ x
        circ_y_shifted_A = circ_y .+ y
        Plots.plot!(circ_x_shifted_A, circ_y_shifted_A, line=:path, color=:blue, label="")
        Plots.quiver!([x], [y], quiver=([u], [v]), aspect_ratio=:equal, axis=([], false), color=:blue, label="", linewidth=.1)

        x, y, u, v = values[5:8]
        circ_x_shifted_B = circ_x .+ x
        circ_y_shifted_B = circ_y .+ y
        Plots.plot!(circ_x_shifted_B, circ_y_shifted_B, line=:path, color=:red, label="")
        Plots.quiver!([x], [y], quiver=([u], [v]), aspect_ratio=:equal, axis=([], false), color=:red, label="", linewidth=.1)
        Plots.plot!([-lat, -lat], [-ymax, ymax], color=:black, label="")
        Plots.plot!([+lat, +lat], [-ymax, ymax], color=:black, label="")
        push!(plots, p)
    end
    Plots.plot(plots..., margin=1e-3*Plots.mm)
end
