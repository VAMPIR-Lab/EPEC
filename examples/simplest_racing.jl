# instantenous velocity change (̇x = u)

# xa := [xa1 xa2], ua := [ua1 ua2]
# xb := [xb1 xb2], ub := [ub1 ub2]
# x[k+1] = dyn(x[k], u[k]; Δt) (pointmass dynamics)

# Xa = [xa[1] | ... xa[T]] ∈ R^(2T)
# Ua = [ua[1] | ... ua[T]] ∈ R^(2T)
# Xb = [xb[1] | ... xb[T]] ∈ R^(2T)
# Ub = [ub[1] | ... ub[T]] ∈ R^(2T)
# z = [Xa | Ua | Xb | Ub | xa0 | xb0] ∈ R^(8T + 4)

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

# P1 wants to maximize lead wrt to P2 at the end of the horizon, minimize offset from the centerline and effort 
function f1(z; α1=1.0, α2=0.0, β=1.0, xdim=2, udim=2)
    T = Int((length(z) - 2 * xdim) / 2 * (xdim + udim))
    @inbounds Xa = @view(z[1:xdim*T])
    @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
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
    xdim = 2
    udim = 2
    T = Int((length(z) - 2 * xdim) / 2 * (xdim + udim))
    @inbounds Xa = @view(z[1:xdim*T])
    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
    @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
    running_cost = 0.0

    for t in 1:T
        @inbounds xb1 = @view(Xb[xdim*(t-1)+1])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        running_cost += α1 * xb1^2 + α2 * ub' * ub
    end
    terminal_cost = β * (Xa[xdim*T] - Xb[xdim*T])
    cost = running_cost + terminal_cost
end