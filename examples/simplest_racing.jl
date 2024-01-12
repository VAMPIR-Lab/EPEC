# instantenous velocity change (̇x = u)

# xa := [xa1 xa2], ua := [ua1 ua2]
# xb := [xb1 xb2], ub := [ub1 ub2]
# x[k+1] = dyn(x[k], u[k]; Δt) (pointmass dynamics)

# Xa = [xa[1] | ... xa[T]] ∈ R^(2T)
# Ua = [ua[1] | ... ua[T]] ∈ R^(2T)
# Xb = [xb[1] | ... xb[T]] ∈ R^(2T)
# Ub = [ub[1] | ... ub[T]] ∈ R^(2T)
# z = [Xa | Ua | Xb | Ub | xa0 | xb0] ∈ R^(8T + 4)
function view_z(z; xdim=2, udim=2)
    T = Int((length(z) - 2 * xdim) / 2 * (xdim + udim))
    @inbounds Xa = @view(z[1:xdim*T])
    @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
    @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
    @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
    @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
    (; Xa, Ua, Xb, Ub, x0a, x0b)
end

# euler integration
# x[k+1] = x[k] + Δt * u
function euler(x, u, Δt)
    x .+ Δt .* u
end

function dyn(X, U, x0, Δt; xdim=2, udim=2)
    T = Int(length(X) / xdim)
    x_prev = x0

    mapreduce(vcat, 1:T) do t
        x = X[xdim*(t-1)+1:xdim*t]
        u = U[udim*(t-1)+1:udim*t]
        diff = x - euler(x_prev, u, Δt)
        x_prev = x
        diff
    end
end

function col(Xa, Xb, r; xdim=2)
    T = Int(length(Xa) / xdim)

    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        delta = xa - xb
        delta' * delta - r^2
    end
end

function responsibility(Xa, Xb; xdim=2)
    T = Int(length(Xa) / xdim)

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
    (; Xa, Ua, Xb, x0a) = view(z)

    g_dyn = dyn(Xa, Ua, x0a, Δt)
    g_col = col(Xa, Xb, r)
    h_col = responsibility(Xa, Xb)

    long_vel = @view(Ua[2:2:end])
    lat_vel = @view(Ua[1:2:end])
    lat_pos = @view(Xa[1:4:end])

    [g_dyn
        g_col - l.(h_col)
        lat_vel
        long_vel
        lat_pos]
end

function g2(z;
    Δt=0.1,
    r=1.0)
    (; Xa, Xb, Ub, x0b) = view(z)

    g_dyn = dyn(Xb, Ub, x0b, Δt)
    g_col = col(Xa, Xb, r)
    h_col = -responsibility(Xa, Xb)

    long_vel = @view(Ub[2:2:end])
    lat_vel = @view(Ub[1:2:end])
    lat_pos = @view(Xb[1:4:end])

    [g_dyn
        g_col - l.(h_col)
        lat_vel
        long_vel
        lat_pos]
end

# P1 wants to maximize lead wrt to P2 at the end of the horizon, minimize offset from the centerline and effort 
function f1(z; α1=1.0, α2=0.0, β=1.0)
    (; Xa, Ua, Xb) = view_z(z)
    running_cost = 0.0

    for t in 1:T
        @inbounds xa1 = @view(Xa[xdim*(t-1)+1])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        running_cost += α1 * xa1^2 + α2 * ua' * ua
    end
    terminal_cost = β * (Xb[xdim*T] - Xa[xdim*T])
    cost = running_cost + terminal_cost
end

# P2 wants to maximize lead wrt to P1 at the end of the horizon, minimize offset from the centerline and effort 
function f2(z; α1=1.0, α2=0.0, β=1.0)
    (; Xa, Xb, Ub) = view_z(z)
    running_cost = 0.0

    for t in 1:T
        @inbounds xb1 = @view(Xb[xdim*(t-1)+1])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        running_cost += α1 * xb1^2 + α2 * ub' * ub
    end
    terminal_cost = β * (Xa[xdim*T] - Xb[xdim*T])
    cost = running_cost + terminal_cost
end

function setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=0.01,
    α2=0.001,
    β=1.0,
    v_max=1.0,
    lat_max=5.0)

    lb = [fill(0.0, 4 * T); fill(0.0, T); fill(-v_max, T); fill(-v_max, T); fill(-lat_max, T)]
    ub = [fill(0.0, 4 * T); fill(Inf, T); fill(+v_max, T); fill(+v_max, T); fill(+lat_max, T)]

    f1_pinned = (z -> f1(z; α1, α2, β))
    f2_pinned = (z -> f2(z; α1, α2, β))
    g1_pinned = (z -> g1(z, Δt, r))
    g2_pinned = (z -> g2(z, Δt, r))

    OP1 = OptimizationProblem(2 * (xdim + udim) * T + 2*xdim, 1:6*T, f1_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(2 * (xdim + udim) * T + 2*xdim, 1:6*T, f2_pinned, g2_pinned, lb, ub)

    gnep = [OP1 OP2]
    bilevel = [OP1; OP2]

    function view_z(z; xdim=2, udim=2)
        T = Int((length(z) - 2 * xdim) / 2 * (xdim + udim))
        @inbounds Xa = @view(z[1:xdim*T])
        @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
        @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
        @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
        @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
        @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
        (; Xa, Ua, Xb, Ub, x0a, x0b)
    end
    function extract_gnep(θ)
        z = θ[gnep.x_inds]
        (; Xa, Ua, Xb, Ub, x0a, x0b) = view_z(z)
    end

    function extract_bilevel(θ)
        z = θ[bilevel.x_inds]
        (; Xa, Ua, Xb, Ub, x0a, x0b) = view_z(z)
    end
    problems = (; gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2, params=(; T, Δt, r))
end