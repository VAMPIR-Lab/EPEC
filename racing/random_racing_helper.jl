using Random
using Statistics

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

    lane_cost = 0.0
    control_cost = 0.0
    velocity_cost = 0.0
    terminal_cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        lane_cost += α1 * xa[1]^2
        control_cost += α2 * ua' * ua
        velocity_cost += -α3 * xa[4]
        terminal_cost += β * xb[2] 
    end
    cost = lane_cost + control_cost + velocity_cost + terminal_cost
    (; cost, lane_cost, control_cost, velocity_cost, terminal_cost)
end

function f2_breakdown(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    lane_cost = 0.0
    control_cost = 0.0
    velocity_cost = 0.0
    terminal_cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        lane_cost += α1 * xb[1]^2
        control_cost += α2 * ub' * ub
        velocity_cost += -α3 * xb[4]
        terminal_cost += β * xa[2]
    end
    cost = lane_cost + control_cost + velocity_cost + terminal_cost
    (; cost, lane_cost, control_cost, velocity_cost, terminal_cost)
end

function compute_realized_cost(res)
    z_arr = zeros(length(res), 12)

    for t in eachindex(res)
        z_arr[t, 1:4] = res[t].x0[1:4]
        z_arr[t, 5:6] = res[t].U1[1, :]
        z_arr[t, 7:10] = res[t].x0[5:8]
        z_arr[t, 11:12] = res[t].U2[1, :]
    end
    Z = [z_arr[:]; zeros(8)] # making it work with f1(Z) and f2(Z)
    a_cost = probs.OP1.f(Z)
    b_cost = probs.OP2.f(Z)
    a_cost_breakdown = f1_breakdown(Z; probs.params.α1, probs.params.α2, probs.params.α3, probs.params.β)
    b_cost_breakdown = f2_breakdown(Z; probs.params.α1, probs.params.α2, probs.params.α3, probs.params.β)
    # breakdowns were copy pasted so
    @assert(isapprox(a_cost, a_cost_breakdown.cost))
    @assert(isapprox(b_cost, b_cost_breakdown.cost))
    (; a_cost, b_cost, a_cost_breakdown, b_cost_breakdown)
end

function find_common_ind(dicts...)
    common_keys = intersect(keys(dict1), keys(dict2), keys(dict3))

end

# statistics
function extract_costs(sp_costs, gnep_costs, bilevel_costs)
    index_arr = []
    sp_cost_arr = []
    sp_lane_cost_arr = []
    sp_control_cost_arr = []
    sp_velocity_cost_arr = []
    sp_terminal_cost_arr = []
    gnep_cost_arr = []
    gnep_lane_cost_arr = []
    gnep_control_cost_arr = []
    gnep_velocity_cost_arr = []
    gnep_terminal_cost_arr = []
    bilevel_cost_arr = []
    bilevel_lane_cost_arr = []
    bilevel_control_cost_arr = []
    bilevel_velocity_cost_arr = []
    bilevel_terminal_cost_arr = []

    for (index, sp_cost) in sp_costs
        if haskey(gnep_costs, index) && haskey(bilevel_costs, index)
            if haskey(bilevel_costs, index)
                push!(index_arr, index)
                push!(sp_cost_arr, [sp_cost.a_cost, sp_cost.b_cost])
                push!(sp_lane_cost_arr, [sp_cost.a_cost_breakdown.lane_cost, sp_cost.b_cost_breakdown.lane_cost])
                push!(sp_control_cost_arr, [sp_cost.a_cost_breakdown.control_cost, sp_cost.b_cost_breakdown.control_cost])
                push!(sp_velocity_cost_arr, [sp_cost.a_cost_breakdown.velocity_cost, sp_cost.b_cost_breakdown.velocity_cost])
                push!(sp_terminal_cost_arr, [sp_cost.a_cost_breakdown.terminal_cost, sp_cost.b_cost_breakdown.terminal_cost])
                push!(gnep_cost_arr, [gnep_costs[index].a_cost, gnep_costs[index].b_cost])
                push!(gnep_lane_cost_arr, [gnep_costs[index].a_cost_breakdown.lane_cost, gnep_costs[index].b_cost_breakdown.lane_cost])
                push!(gnep_control_cost_arr, [gnep_costs[index].a_cost_breakdown.control_cost, gnep_costs[index].b_cost_breakdown.control_cost])
                push!(gnep_velocity_cost_arr, [gnep_costs[index].a_cost_breakdown.velocity_cost, gnep_costs[index].b_cost_breakdown.velocity_cost])
                push!(gnep_terminal_cost_arr, [gnep_costs[index].a_cost_breakdown.terminal_cost, gnep_costs[index].b_cost_breakdown.terminal_cost])
                push!(bilevel_cost_arr, [bilevel_costs[index].a_cost, bilevel_costs[index].b_cost])
                push!(bilevel_lane_cost_arr, [bilevel_costs[index].a_cost_breakdown.lane_cost, bilevel_costs[index].b_cost_breakdown.lane_cost])
                push!(bilevel_control_cost_arr, [bilevel_costs[index].a_cost_breakdown.control_cost, bilevel_costs[index].b_cost_breakdown.control_cost])
                push!(bilevel_velocity_cost_arr, [bilevel_costs[index].a_cost_breakdown.velocity_cost, bilevel_costs[index].b_cost_breakdown.velocity_cost])
                push!(bilevel_terminal_cost_arr, [bilevel_costs[index].a_cost_breakdown.terminal_cost, bilevel_costs[index].b_cost_breakdown.terminal_cost])
            end
        end
    end
    sp_extracted = (total=sp_cost_arr, lane=sp_lane_cost_arr, control=sp_control_cost_arr, velocity=sp_velocity_cost_arr, terminal=sp_terminal_cost_arr)
    gnep_extracted = (total=gnep_cost_arr, lane=gnep_lane_cost_arr, control=gnep_control_cost_arr, velocity=gnep_velocity_cost_arr, terminal=gnep_terminal_cost_arr)
    bilevel_extracted = (total=bilevel_cost_arr, lane=bilevel_lane_cost_arr, control=bilevel_control_cost_arr, velocity=bilevel_velocity_cost_arr, terminal=bilevel_terminal_cost_arr)
    cost = (ind=index_arr, sp=sp_extracted, gnep=gnep_extracted, bilevel=bilevel_extracted)
    cost
end

function compute_player_Δcost(baseline_cost, other_cost)
    P1_baseline_cost = [v[1] for v in baseline_cost]
    P1_other_cost = [v[1] for v in other_cost]
    P2_baseline_cost = [v[2] for v in baseline_cost]
    P2_other_cost = [v[2] for v in other_cost]

    # bilevel vs gnep
    P1_abs_Δcost = P1_other_cost .- P1_baseline_cost
    P2_abs_Δcost = P2_other_cost .- P2_baseline_cost
    P1_rel_Δcost = P1_abs_Δcost ./ abs.(P1_baseline_cost)
    P2_rel_Δcost = P2_abs_Δcost ./ abs.(P2_baseline_cost)

    P1_max_ind = argmax(P1_abs_Δcost)
    P1_min_ind = argmin(P1_abs_Δcost)
    P2_max_ind = argmax(P2_abs_Δcost)
    P2_min_ind = argmin(P2_abs_Δcost)
    Δcost = (; P1_abs=P1_abs_Δcost, P2_abs=P2_abs_Δcost, P1_rel=P1_rel_Δcost, P2_rel=P2_rel_Δcost, P1_max_ind, P1_min_ind, P2_max_ind, P2_min_ind)
    Δcost
end

function print_mean_min_max(P1_cost_diff, P2_cost_diff, P1_rel_cost_diff, P2_rel_cost_diff)
    println("		mean 		        stderr 		    min		        max")
    println("P1 Δcost abs :  $(mean(P1_cost_diff))  $(std(P1_cost_diff)/sqrt(length(P1_cost_diff)))  $(minimum(P1_cost_diff))  $(maximum(P1_cost_diff))")
    println("P2 Δcost abs :  $(mean(P2_cost_diff))  $(std(P2_cost_diff)/sqrt(length(P1_cost_diff)))  $(minimum(P2_cost_diff))  $(maximum(P2_cost_diff))")
    println("P1 Δcost rel%:  $(mean(P1_rel_cost_diff)*100)  $(std(P1_rel_cost_diff)/sqrt(length(P1_cost_diff))*100)  $(minimum(P1_rel_cost_diff)*100)  $(maximum(P1_rel_cost_diff)*100)")
    println("P2 Δcost rel%:  $(mean(P2_rel_cost_diff)*100)  $(std(P2_rel_cost_diff)/sqrt(length(P1_cost_diff))*100)  $(minimum(P2_rel_cost_diff)*100)  $(maximum(P2_rel_cost_diff)*100)")
end