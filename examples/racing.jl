# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# xⁱₖ := [p1 p2 v1 v2]
# xₖ = dyn(xₖ₋₁, uₖ; Δt) (pointmass dynamics)
const xdim = 4
const udim = 2

# P1 wants to make forward progress and stay in center of lane.
function f1(Z; α1=1.0, α2=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    #@infiltrate
    running_cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        running_cost += α1 * xa[1]^2 + α2 * ua' * ua
    end
    @inbounds xa = @view(Xa[xdim*(T-1)+1:xdim*T])
    @inbounds xb = @view(Xb[xdim*(T-1)+1:xdim*T])
    terminal_cost = β * (xb[2] - 2 * xa[2])
    cost = running_cost + terminal_cost
end

# P2 wants to make forward progress and stay in center of lane.
function f2(Z; α1=1.0, α2=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    running_cost = 0.0

    for t in 1:T
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        running_cost += α1 * xb[1]^2 + α2 * ub' * ub
    end
    @inbounds xa = @view(Xa[xdim*(T-1)+1:xdim*T])
    @inbounds xb = @view(Xb[xdim*(T-1)+1:xdim*T])
    terminal_cost = β * (xa[2] - 2 * xb[2])
    cost = running_cost + terminal_cost
end

function pointmass(x, u, Δt, cd)
    Δt2 = 0.5 * Δt * Δt
    a1 = u[1] - cd * x[3]
    a2 = u[2] - cd * x[4]
    [x[1] + Δt * x[3] + Δt2 * a1,
        x[2] + Δt * x[4] + Δt2 * a2,
        x[3] + Δt * a1,
        x[4] + Δt * a2]
end

function dyn(X, U, x0, Δt, cd)
    T = Int(length(X) / 4)
    x = x0
    mapreduce(vcat, 1:T) do t
        xx = X[(t-1)*4+1:t*4]
        u = U[(t-1)*2+1:t*2]
        diff = xx - pointmass(x, u, Δt, cd)
        x = xx
        diff
    end
end

function col(Xa, Xb, r)
    T = Int(length(Xa) / 4)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xb = @view(Xb[(t-1)*4+1:t*4])
        delta = xa[1:2] - xb[1:2]
        [delta' * delta - r^2,]
    end
end

function responsibility(Xa, Xb)
    T = Int(length(Xa) / 4)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xb = @view(Xb[(t-1)*4+1:t*4])
        h = [xb[2] - xa[2],] # h is positive when xa is behind xb in second coordinate
    end
end

function accel_bounds_1(Xa, Xb, u_max_nominal, u_max_drafting, box_length, box_width)
    T = Int(length(Xa) / 4)
    d = mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xb = @view(Xb[(t-1)*4+1:t*4])
        [xa[1] - xb[1] xa[2] - xb[2]]
    end
    @assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    u_max_1 = du * sigmoid.(d[:, 2] .+ box_length, 10.0, 0) .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:, 2], 10.0, 0) .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    (u_max_1, u_max_2, u_max_3, u_max_4)
end
function accel_bounds_2(Xa, Xb, u_max_nominal, u_max_drafting, box_length, box_width)
    T = Int(length(Xa) / 4)
    d = mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xb = @view(Xb[(t-1)*4+1:t*4])
        [xb[1] - xa[1] xb[2] - xa[2]]
    end
    @assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    u_max_1 = du * sigmoid.(d[:, 2] .+ box_length, 10.0, 0) .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:, 2], 10.0, 0) .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    (u_max_1, u_max_2, u_max_3, u_max_4)
end

function sigmoid(x, a, b)
    xx = x * a + b
    1.0 / (1.0 + exp(-xx))
end

# lower bound function -- above zero whenever h ≥ 0, below zero otherwise
function l(h; a=5.0, b=4.5)
    sigmoid(h, a, b) - sigmoid(0, a, b)
end

function g1(Z,
    Δt=0.1,
    r=1.0,
    cd=1.0,
    u_max_nominal=2.0,
    u_max_drafting=5.0,
    box_length=3.0,
    box_width=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])
    @inbounds x0a = @view(Z[12*T+1:12*T+4])
    @inbounds x0b = @view(Z[12*T+5:12*T+8])

    g_dyn = dyn(Xa, Ua, x0a, Δt, cd)
    g_col = col(Xa, Xb, r)
    h_col = responsibility(Xa, Xb)
    u_max_1, u_max_2, u_max_3, u_max_4 = accel_bounds_1(Xa,
        Xb,
        u_max_nominal,
        u_max_drafting,
        box_length,
        box_width)
    long_accel = @view(Ua[2:2:end])
    lat_accel = @view(Ua[1:2:end])
    lat_pos = @view(Xa[1:4:end])

    [g_dyn
        g_col - l.(h_col)
        lat_accel
        long_accel - u_max_1
        long_accel - u_max_2
        long_accel - u_max_3
        long_accel - u_max_4
        lat_pos]
end

function g2(Z,
    Δt=0.1,
    r=1.0,
    cd=1.0,
    u_max_nominal=2.0,
    u_max_drafting=5.0,
    box_length=3.0,
    box_width=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])
    @inbounds x0a = @view(Z[12*T+1:12*T+4])
    @inbounds x0b = @view(Z[12*T+5:12*T+8])

    g_dyn = dyn(Xb, Ub, x0b, Δt, cd)
    g_col = col(Xa, Xb, r)
    h_col = -responsibility(Xa, Xb)
    u_max_1, u_max_2, u_max_3, u_max_4 = accel_bounds_2(Xa,
        Xb,
        u_max_nominal,
        u_max_drafting,
        box_length,
        box_width)
    long_accel = @view(Ub[2:2:end])
    lat_accel = @view(Ub[1:2:end])
    lat_pos = @view(Xb[1:4:end])

    [g_dyn
        g_col - l.(h_col)
        lat_accel
        long_accel - u_max_1
        long_accel - u_max_2
        long_accel - u_max_3
        long_accel - u_max_4
        lat_pos]
end

function setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=0.01,
    α2=0.001,
    β = 1e3,
    cd=0.2,
    u_max_nominal=2.0,
    u_max_drafting=5.0,
    box_length=3.0,
    box_width=1.0,
    lat_max=5.0)

    lb = [fill(0., 4 * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4 * T); fill(-lat_max, T)]
    ub = [fill(0., 4 * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, 4 * T); fill(+lat_max, T)]

    f1_pinned = (z -> f1(z; α1, α2, β))
    f2_pinned = (z -> f2(z; α1, α2, β))
    g1_pinned = (z -> g1(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width))
    g2_pinned = (z -> g2(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width))

    OP1 = OptimizationProblem(12 * T + 8, 1:6*T, f1_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(12 * T + 8, 1:6*T, f2_pinned, g2_pinned, lb, ub)

	sp_a = EPEC.create_epec((1,0), OP1)
	sp_b = EPEC.create_epec((1,0), OP2)
    gnep = [OP1 OP2]
    bilevel = [OP1; OP2]

    function extract_gnep(θ)
        Z = θ[gnep.x_inds]
        @inbounds Xa = @view(Z[1:4*T])
        @inbounds Ua = @view(Z[4*T+1:6*T])
        @inbounds Xb = @view(Z[6*T+1:10*T])
        @inbounds Ub = @view(Z[10*T+1:12*T])
        @inbounds x0a = @view(Z[12*T+1:12*T+4])
        @inbounds x0b = @view(Z[12*T+5:12*T+8])
        (; Xa, Ua, Xb, Ub, x0a, x0b)
    end
    function extract_bilevel(θ)
        Z = θ[bilevel.x_inds]
        @inbounds Xa = @view(Z[1:4*T])
        @inbounds Ua = @view(Z[4*T+1:6*T])
        @inbounds Xb = @view(Z[6*T+1:10*T])
        @inbounds Ub = @view(Z[10*T+1:12*T])
        @inbounds x0a = @view(Z[12*T+1:12*T+4])
        @inbounds x0b = @view(Z[12*T+5:12*T+8])
        (; Xa, Ua, Xb, Ub, x0a, x0b)
    end
    problems = (; sp_a, sp_b, gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2, params=(; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting))
end

function solve_seq(probs, x0)
    init = zeros(probs.gnep.top_level.n)
    X = init[probs.gnep.x_inds]
    #T = Int(length(X) / 12)
    T = probs.params.T
    Δt = probs.params.Δt
    cd = probs.params.cd
    Xa = []
    Ua = []
    Xb = []
    Ub = []
    xa = x0[1:4]
    xb = x0[5:8]
    for t in 1:T
        ua = cd * xa[3:4]
        ub = cd * xb[3:4]
        xa = pointmass(xa, ua, Δt, cd)
        xb = pointmass(xb, ub, Δt, cd)
        append!(Ua, ua)
        append!(Ub, ub)
        append!(Xa, xa)
        append!(Xb, xb)
    end
    init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
    @infiltrate
	#show_me(init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
	init = [init; x0]    


	@info "Solving singleplayer: a"
    #!!! 348 vs 568 breaks it
    sp_a_init = zeros(probs.gnep.top_level.n)
    sp_a_init[probs.sp_a.x_inds] = [Xa; Ua]
    sp_a_init[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60] = [Xb; Ub]
    sp_a_init[probs.sp_a.top_level.n+61:probs.sp_a.top_level.n+68] = x0;
    #sp_a_init = [sp_a_init; x0]
    θ_sp_a = solve(probs.sp_a, sp_a_init)
    #@infiltrate
    # !!! Ordering of x_w changes because:
    # 1p: [x1=>1:60 λ1,s1=>61:280, w=>281:348]
    # 2p: [x1,x2=>1:120, λ1,λ2,s1,s2=>121:560 w=>561:568]
    # Solution: parametrize view statements?
	#θ_sp_a = solve(probs.sp_a, sp_a_init)
    #θ_sp_a =  solve(probs.sp_a, init)
    #show_me([safehouse.θ_out[probs.sp_a.x_inds]; safehouse.θ_out[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60]], safehouse.w[probs.sp_a.top_level.n+1:end]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    #show_me([θ_sp_a[probs.sp_a.x_inds]; θ_sp_a[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    return
    @info "Solving gnep.."
    θg = solve(probs.gnep, init)
    θb = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
    θb[probs.bilevel.x_inds] = θg[probs.gnep.x_inds]
    θb[probs.bilevel.inds["λ", 1]] = θg[probs.gnep.inds["λ", 1]]
    θb[probs.bilevel.inds["s", 1]] = θg[probs.gnep.inds["s", 1]]
    θb[probs.bilevel.inds["λ", 2]] = θg[probs.gnep.inds["λ", 2]]
    θb[probs.bilevel.inds["s", 2]] = θg[probs.gnep.inds["s", 2]]
    θb[probs.bilevel.inds["w", 0]] = θg[probs.gnep.inds["w", 0]]

    
    @info "Solving bilevel.."
    #@info probs.bilevel.inds["λ", 1]
    θ = solve(probs.bilevel, θb)
    #θ = solve(probs.bilevel, init)
    Z = probs.extract_bilevel(θ)
    #Z = probs.extract_gnep(θg)
    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    gd = col(Z.Xa, Z.Xb, probs.params.r)
    h = responsibility(Z.Xa, Z.Xb)
    gd_both = [gd - l.(h) gd - l.(-h) gd]
    (; P1, P2, gd_both, h, U1, U2)
end

function solve_simulation(probs, T; x0=[0, 0, 0, 7, 0.1, -2.21, 0, 7])
    results = Dict()
    for t = 1:T
        @info "Simulation step $t"
        r = solve_seq(probs, x0)
        x0a = r.P1[1, :]
        x0b = r.P2[1, :]
        results[t] = (; x0, r.P1, r.P2, r.U1, r.U2, r.gd_both, r.h)
        x0 = [x0a; x0b]
    end
    results
end

function animate(probs, sim_results; save=false)
    rad = sqrt(probs.params.r) / 2
    lat = probs.params.lat_max + rad
    (f, ax, XA, XB, lat) = visualize(; rad=rad, lat=lat)
    display(f)
    T = length(sim_results)

    if save
        record(f, "test.mp4", 1:T; framerate=20) do t
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = string(t)
        end
    else
        for t in 1:T
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = string(t)
            sleep(1e-2)
        end
    end
end

function visualize(; rad=0.5, lat=6.0)
    f = Figure(resolution=(1000, 1000))
    ax = Axis(f[1, 1], aspect=DataAspect())

    lines!(ax, [-lat, -lat], [-10.0, 300.0], color=:black)
    lines!(ax, [+lat, +lat], [-10.0, 300.0], color=:black)

    XA = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:10)
    XB = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:10)

    circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]
    lines!(ax, @lift(circ_x .+ $(XA[0][1])), @lift(circ_y .+ $(XA[0][2])), color=:blue, linewidth=5)
    lines!(ax, @lift(circ_x .+ $(XB[0][1])), @lift(circ_y .+ $(XB[0][2])), color=:red, linewidth=5)

    for t in 1:10
        lines!(ax, @lift(circ_x .+ $(XA[t][1])), @lift(circ_y .+ $(XA[t][2])), color=:blue, linewidth=2, linestyle=:dash)
        lines!(ax, @lift(circ_x .+ $(XB[t][1])), @lift(circ_y .+ $(XB[t][2])), color=:red, linewidth=2, linestyle=:dash)
    end

    return (f, ax, XA, XB, lat)
end

function update_visual!(ax, XA, XB, x0, P1, P2; T=10, lat=6.0)
    XA[0][1][] = x0[1]
    XA[0][2][] = x0[2]
    XB[0][1][] = x0[5]
    XB[0][2][] = x0[6]

    for l in 1:T
        XA[l][1][] = P1[l, 1]
        XA[l][2][] = P1[l, 2]
        XB[l][1][] = P2[l, 1]
        XB[l][2][] = P2[l, 2]
    end

    xlims!(ax, -2 * lat, 2 * lat)
    ylims!(ax, x0[6] - lat, maximum([P1[T, 2], P2[T, 2]]) + lat)
end

function show_me(θ, x0; T=10, t=0, lat_pos_max=1.0)
    x_inds = 1:12*T
    function extract(θ; x_inds=x_inds, T=T)
        Z = θ[x_inds]
        @inbounds Xa = @view(Z[1:4*T])
        @inbounds Ua = @view(Z[4*T+1:6*T])
        @inbounds Xb = @view(Z[6*T+1:10*T])
        @inbounds Ub = @view(Z[10*T+1:12*T])
        (; Xa, Ua, Xb, Ub)
    end
    Z = extract(θ)

    (f, ax, XA, XB, lat) = visualize(; lat=lat_pos_max)
    display(f)

    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    update_visual!(ax, XA, XB, x0, P1, P2; T=T, lat=lat_pos_max)

    if t > 0
        ax.title = string(t)
    end
end