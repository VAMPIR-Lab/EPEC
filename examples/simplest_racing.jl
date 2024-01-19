module simplest_racing

using EPEC
using GLMakie

# problem size
# instantenous velocity change (̇x = u)
# xa := [xa1 xa2], ua := [ua1 ua2]
# xb := [xb1 xb2], ub := [ub1 ub2]
# x[k+1] = dyn(x[k], u[k]; Δt) (pointmass dynamics)
# Xa = [xa[1] | ... xa[T]] ∈ R^(2T)
# Ua = [ua[1] | ... ua[T]] ∈ R^(2T)
# Xb = [xb[1] | ... xb[T]] ∈ R^(2T)
# Ub = [ub[1] | ... ub[T]] ∈ R^(2T)
# z = [Xa | Ua | Xb | Ub | xa0 | xb0] ∈ R^(8T + 4)
const xdim = 2
const udim = 2

function view_z(z)
    T = Int((length(z) - 2 * xdim) / (2 * (xdim + udim)))
    @inbounds Xa = @view(z[1:xdim*T])
    @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
    @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
    @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
    @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
    @inbounds (; T, Xa, Ua, Xb, Ub, x0a, x0b)
end

# euler integration
# x[k+1] = x[k] + Δt * u
function euler(x, u, Δt)
    x .+ Δt .* u
end

function dyn(T, X, U, x0, Δt)
    x_prev = x0

    mapreduce(vcat, 1:T) do t
        x = X[xdim*(t-1)+1:xdim*t]
        u = U[udim*(t-1)+1:udim*t]
        diff = x - euler(x_prev, u, Δt)
        x_prev = x
        diff
    end
end

function col(T, Xa, Xb, r)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        delta = xa - xb
        delta' * delta - r^2
    end
end

function responsibility(T, Xa, Xb)
    mapreduce(vcat, 1:T) do t
        @inbounds xa2 = @view(Xa[xdim*t])
        @inbounds xb2 = @view(Xb[xdim*t])
        h = xb2 - xa2 # h is positive when xa is behind xb in the second coordinate
    end
end

function sigmoid(x, a, b)
    xx = x * a + b
    1.0 / (1.0 + exp(-xx))
end

# lower bound function -- above zero whenever h ≥ 0, below zero otherwise
function l(h; a=5.0, b=4.5)
    sigmoid(h, a, b) - sigmoid(0, a, b)
end

# P1 constraints: dynamics, collision, velocity
function g1(z;
    Δt=0.1,
    r=1.0)
    #T, Xa, Ua, Xb, x0a = view_z(z) # why doesn't this work?? view(z) copy pasted:
    T = Int((length(z) - 2 * xdim) / (2 * (xdim + udim)))
    @inbounds Xa = @view(z[1:xdim*T])
    @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
    @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
    @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
    @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
    (; T, Xa, Ua, Xb, Ub, x0a, x0b)

    g_dyn = dyn(T, Xa, Ua, x0a, Δt)
    g_col = col(T, Xa, Xb, r)
    h_col = responsibility(T, Xa, Xb)

    long_vel = @view(Ua[2:udim:end])
    lat_vel = @view(Ua[1:udim:end])
    lat_pos = @view(Xa[1:xdim:end])

    [g_dyn
        g_col - l.(h_col)
        lat_vel
        long_vel
        lat_pos]
end

function g2(z;
    Δt=0.1,
    r=1.0)
    #(; T, Xa, Xb, Ub, x0b) = view_z(z) # why doesn't this work??  view(z) copy pasted:
    T = Int((length(z) - 2 * xdim) / (2 * (xdim + udim)))
    @inbounds Xa = @view(z[1:xdim*T])
    @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
    @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
    @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
    @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])

    g_dyn = dyn(T, Xb, Ub, x0b, Δt)
    g_col = col(T, Xa, Xb, r)
    h_col = -responsibility(T, Xa, Xb)

    long_vel = @view(Ub[2:udim:end])
    lat_vel = @view(Ub[1:udim:end])
    lat_pos = @view(Xb[1:xdim:end])

    [g_dyn
        g_col - l.(h_col)
        lat_vel
        long_vel .- 2.0
        lat_pos]
end

# P1 wants to maximize lead wrt to P2 at the end of the horizon, minimize offset from the centerline and effort 
function f1(z; α1=1.0, α2=0.0, β=1.0)
    (; T, Xa, Ua, Xb) = view_z(z)
    running_cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        running_cost += α1 * xa[1]^2 + α2 * ua' * ua
    end
    terminal_cost = β * (Xb[xdim*T] - 2 * Xa[xdim*T])
    cost = running_cost + terminal_cost
end

# P2 wants to maximize lead wrt to P1 at the end of the horizon, minimize offset from the centerline and effort 
function f2(z; α1=1.0, α2=0.0, β=1.0)
    (; T, Xa, Xb, Ub) = view_z(z)
    running_cost = 0.0

    for t in 1:T
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        running_cost += α1 * xb[1]^2 + α2 * ub' * ub
    end
    terminal_cost = β * (Xa[xdim*T] - 2 * Xb[xdim*T])
    cost = running_cost + terminal_cost
end

function setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-2,
    α2=0e0,
    β=1e1,
    lat_vel_max=1.0,
    long_vel_max=10.0,
    lat_pos_max=1.0)

    lb = [fill(0.0, xdim * T); fill(0.0, T); fill(-lat_vel_max, T); fill(-long_vel_max, T); fill(-lat_pos_max, T)]
    ub = [fill(0.0, xdim * T); fill(Inf, T); fill(+lat_vel_max, T); fill(+long_vel_max, T); fill(+lat_pos_max, T)]
    f1_pinned = (z -> f1(z; α1, α2, β))
    f2_pinned = (z -> f2(z; α1, α2, β))
    g1_pinned = (z -> g1(z; Δt, r))
    g2_pinned = (z -> g2(z; Δt, r))

    OP1 = OptimizationProblem(2 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f1_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(2 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f2_pinned, g2_pinned, lb, ub)

    gnep = [OP1 OP2]
    bilevel = [OP1; OP2]

    function extract_gnep(θ)
        z = θ[gnep.x_inds]
        #(; Xa, Ua, Xb, Ub, x0a, x0b) = view_z(z) #InexactError: Int64(9.5)??  view(z) copy pasted:
        @inbounds Xa = @view(z[1:xdim*T])
        @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
        @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
        @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
        @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
        @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
        (; T, Xa, Ua, Xb, Ub, x0a, x0b)
    end

    function extract_bilevel(θ)
        z = θ[bilevel.x_inds]
        #(; Xa, Ua, Xb, Ub, x0a, x0b) = view_z(z) # InexactError: Int64(9.5)??  view(z) copy pasted:
        @inbounds Xa = @view(z[1:xdim*T])
        @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
        @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
        @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
        @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
        @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
        (; T, Xa, Ua, Xb, Ub, x0a, x0b)
    end
    problems = (; gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2, params=(; T, Δt, r, lat_pos_max))
end

function solve_seq(probs, x0)
    T = probs.params.T
    Δt = probs.params.Δt
    Xa = []
    Ua = []
    Xb = []
    Ub = []
    xa = x0[1:xdim]
    xb = x0[xdim+1:2*xdim]

    for t in 1:T
        ua = [0; 1]
        ub = [0; 1]
        xa = euler(xa, ua, Δt)
        xb = euler(xb, ub, Δt)
        append!(Ua, ua)
        append!(Ub, ub)
        append!(Xa, xa)
        append!(Xb, xb)
    end
    init = zeros(probs.gnep.top_level.n)
    init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
    init = [init; x0]

    #show_me(init, x0; x_inds=1:120, T=T, t=0) 
    #@infiltrate
    θg = solve(probs.gnep, init)
    θb = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
    θb[probs.bilevel.x_inds] = θg[probs.gnep.x_inds]
    θb[probs.bilevel.inds["λ", 1]] = θg[probs.gnep.inds["λ", 1]]
    θb[probs.bilevel.inds["s", 1]] = θg[probs.gnep.inds["s", 1]]
    θb[probs.bilevel.inds["λ", 2]] = θg[probs.gnep.inds["λ", 2]]
    θb[probs.bilevel.inds["s", 2]] = θg[probs.gnep.inds["s", 2]]
    θb[probs.bilevel.inds["w", 0]] = θg[probs.gnep.inds["w", 0]]

    θ = solve(probs.bilevel, θb)
    #θ = solve(probs.bilevel, init)
    Z = probs.extract_bilevel(θ)
    #Z = probs.extract_gnep(θ)
    #Z = probs.extract_gnep(θg)
    P1 = [Z.Xa[1:xdim:end] Z.Xa[2:xdim:end]]
    U1 = [Z.Ua[1:udim:end] Z.Ua[2:udim:end]]
    P2 = [Z.Xb[1:xdim:end] Z.Xb[2:xdim:end]]
    U2 = [Z.Ub[1:udim:end] Z.Ub[2:udim:end]]

    gd = col(T, Z.Xa, Z.Xb, probs.params.r)
    h = responsibility(T, Z.Xa, Z.Xb)
    gd_both = [gd - l.(h) gd - l.(-h) gd]
    (; P1, P2, gd_both, h, U1, U2)
end

function solve_simulation(probs, T; x0=[0, 0, 0, 1])
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
    lat = probs.params.lat_pos_max + rad
    (f, ax, XA, XB, lat) = visualize(; rad=rad, lat=lat)
    display(f)
    T = length(sim_results)

    if save
        record(f, "jockeying_animation.mp4", 1:T; framerate=10) do t
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = string(t)
            sleep(0.2)
        end
    else
        for t in 1:T
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = string(t)
            sleep(0.2)
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
    XB[0][1][] = x0[3]
    XB[0][2][] = x0[4]

    for l in 1:T
        XA[l][1][] = P1[l, 1]
        XA[l][2][] = P1[l, 2]
        XB[l][1][] = P2[l, 1]
        XB[l][2][] = P2[l, 2]
    end

    xlims!(ax, -2 * lat, 2 * lat)
    ylims!(ax, x0[4] - lat, maximum([P1[T, 2], P2[T, 2]]) + lat)
end

function show_me(θ, x0; T=10, t=0, lat_pos_max=1.0)
    x_inds = 1:2*(xdim+udim)*T
    function extract(θ; x_inds=x_inds, T=T)
        Z = θ[x_inds]
        @inbounds Xa = @view(Z[1:xdim*T])
        @inbounds Ua = @view(Z[xdim*T+1:(xdim+udim)*T])
        @inbounds Xb = @view(Z[(xdim+udim)*T+1:(2*xdim+udim)*T])
        @inbounds Ub = @view(Z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
        (; Xa, Ua, Xb, Ub)
    end
    Z = extract(θ)

    (f, ax, XA, XB, lat) = visualize(; lat=lat_pos_max)
    display(f)

    P1 = [Z.Xa[1:xdim:end] Z.Xa[2:xdim:end]]
    U1 = [Z.Ua[1:udim:end] Z.Ua[2:udim:end]]
    P2 = [Z.Xb[1:xdim:end] Z.Xb[2:xdim:end]]
    U2 = [Z.Ub[1:udim:end] Z.Ub[2:udim:end]]

    update_visual!(ax, XA, XB, x0, P1, P2; T=T, lat=lat_pos_max)

    if t > 0
        ax.title = string(t)
    end
end

#function setup(;
#    x0=[0.0; 2.0; 0.0; 0.0],
#    sim_steps=50,
#    T=10,
#    Δt=0.1,
#    r=1.0,
#    α1=1e-2,
#    α2=0e0,
#    β=1e1,
#    lat_vel_max=1.0,
#    long_vel_max=10.0,
#    lat_pos_max=1.0
#)
#    probs = setup_probs(;T=T);
#    #(; P1, P2, gd_both, h, U1, U2) = simplest_racing.solve_seq(probs, x0);
#    # in case exfiltrated:
#    # before x0
#    #show_me(safehouse.x, safehouse.w)
#    # after
#    #simplest_racing.show_me(safehouse.θ_out, safehouse.w; T=10)

#    sim_results = solve_simulation(probs, sim_steps; x0);
#    animate(probs, sim_results; save=false);
#end
end