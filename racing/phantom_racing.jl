# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# τⁱ := [xⁱ₁ ... xⁱₜ | uⁱ₁ ... uⁱₜ]
# Z := [τ¹ | τ² | ϕ¹ | ϕ²]

function view_z(z)
    xdim = 4
    udim = 2
    T = Int((length(z) - 2 * xdim) / (4 * (xdim + udim)))
    indices = Dict()
    idx = 0
    for (len, name) in zip([xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim, xdim], ["Xa", "Ua", "Xb", "Ub", "Xap", "Uap", "Xbp", "Ubp", "x0a", "x0b"])
        indices[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds Xa = @view(z[indices["Xa"]])
    @inbounds Ua = @view(z[indices["Ua"]])
    @inbounds Xb = @view(z[indices["Xb"]])
    @inbounds Ub = @view(z[indices["Ub"]])
    @inbounds Xap = @view(z[indices["Xap"]])
    @inbounds Uap = @view(z[indices["Uap"]])
    @inbounds Xbp = @view(z[indices["Xbp"]])
    @inbounds Ubp = @view(z[indices["Ubp"]])
    @inbounds x0a = @view(z[indices["x0a"]])
    @inbounds x0b = @view(z[indices["x0b"]])
    (T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices)
end

# ego: a
# Each player wants to make forward progress and stay in center of lane.
function f(T, Xa, Ua, Xb; α1, α2, α3, β)
    xdim = 4
    udim = 2
    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        cost += α1 * xa[1]^2 + α2 * ua' * ua - α3 * xa[4] + β * xb[4]
    end
    cost
end

function f_a(z; α1, α2, α3, β, γ)
    T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    # γ f(τ_a, τ_b) + (1 - γ) f(τ_a, ϕ_b)
    γ * f(T, Xa, Ua, Xb; α1, α2, α3, β) + (1.0 - γ) * f(T, Xa, Ua, Xbp; α1, α2, α3, β)
end

function f_b(z; α1, α2, α3, β, γ)
    T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    # γ f(τ_a, τ_b) + (1 - γ) f(ϕ_a, τ_b)
    γ * f(T, Xb, Ub, Xa; α1, α2, α3, β) + (1.0 - γ) * f(T, Xb, Ub, Xap; α1, α2, α3, β)
end

function f_ap(z; α1, α2, α3, β)
    T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    f(T, Xap, Uap, Xbp; α1, α2, α3, β)
end

function f_bp(z; α1, α2, α3, β)
    T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    f(T, Xbp, Ubp, Xap; α1, α2, α3, β)
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
    xdim = 4
    udim = 2
    T = Int(length(X) / 4)
    x = x0
    mapreduce(vcat, 1:T) do t
        xx = X[(t-1)*xdim+1:t*xdim]
        u = U[(t-1)*udim+1:t*udim]
        diff = xx - pointmass(x, u, Δt, cd)
        x = xx
        diff
    end
end

function col(Xa, Xb, r)
    xdim = 4
    udim = 2
    T = Int(length(Xa) / 4)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*xdim+1:t*xdim])
        @inbounds xb = @view(Xb[(t-1)*xdim+1:t*xdim])
        delta = xa[1:2] - xb[1:2]
        [delta' * delta - r^2,]
    end
end

function responsibility(Xa, Xb)
    xdim = 4
    udim = 2
    T = Int(length(Xa) / 4)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*xdim+1:t*xdim])
        @inbounds xb = @view(Xb[(t-1)*xdim+1:t*xdim])
        h = [xb[2] - xa[2],] # h is positive when xa is behind xb in second coordinate
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

# ego: a
function g(Xa, Ua, Xb, x0a; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    xdim = 4
    udim = 2
    g_dyn = dyn(Xa, Ua, x0a, Δt, cd)
    g_col = col(Xa, Xb, r)
    h_col = responsibility(Xa, Xb)

    T = Int(length(Xa) / 4)
    d = mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*xdim+1:t*xdim])
        @inbounds xb = @view(Xb[(t-1)*xdim+1:t*xdim])
        [xa[1] - xb[1] xa[2] - xb[2]]
    end
    #@assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    u_max_1 = du * sigmoid.(d[:, 2] .+ box_length, 10.0, 0) .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:, 2], 10.0, 0) .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal

    long_accel = @view(Ua[udim:udim:end])
    lat_accel = @view(Ua[1:udim:end])
    lat_pos = @view(Xa[1:xdim:end])
    long_vel = @view(Xa[xdim:xdim:end])

    [g_dyn
        g_col - l.(h_col) .- col_buffer
        lat_accel
        long_accel - u_max_1
        long_accel - u_max_2
        long_accel - u_max_3
        long_accel - u_max_4
        long_accel
        long_vel
        lat_pos]
end

function g_a_all(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices =view_z(z)

    [g(Xa, Ua, Xb, x0a; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g(Xa, Ua, Xbp, x0a; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end
function g_b_all(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    [g(Xb, Ub, Xa, x0a; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g(Xb, Ub, Xap, x0a; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end


function setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=0.01,
    α2=0.001,
    α3=0.0,
    β=1e3,
    γ=1,
    cd=0.2,
    u_max_nominal=2.0,
    u_max_drafting=5.0,
    u_max_braking=2 * u_max_drafting,
    box_length=3.0,
    box_width=1.0,
    lat_max=5.0,
    min_long_vel=-5.0,
    col_buffer=r / 5)
    xdim = 4
    udim = 2

    f_a_pinned = (z -> f_a(z; α1, α2, α3, β, γ))
    f_b_pinned = (z -> f_b(z; α1, α2, α3, β, γ))
    f_ap_pinned = (z -> f_ap(z; α1, α2, α3, β))
    f_bp_pinned = (z -> f_bp(z; α1, α2, α3, β))
    g_a_all_pinned = (z -> g_a_all(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    g_b_all_pinned = (z -> g_b_all(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))

    lb = [fill(0.0, xdim * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, xdim * T); fill(-u_max_braking, T); fill(min_long_vel, T); fill(-lat_max, T)]
    ub = [fill(0.0, xdim * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, xdim * T); fill(Inf, T); fill(Inf, T); fill(+lat_max, T)]
    lb = [lb; lb]
    ub = [ub; ub]

    OP_a = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_a_pinned, g_a_all_pinned, lb, ub)
    OP_b = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_b_pinned, g_b_all_pinned, lb, ub)
    OP_ap = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_ap_pinned, g_a_all_pinned, lb, ub)
    OP_bp = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_bp_pinned, g_b_all_pinned, lb, ub)

    #sp_a = EPEC.create_epec((1, 0), OP_a)
    #gnep = [OP_a OP_b]
    #bilevel = [OP_a; OP_b]
    phantom = EPEC.create_epec((2, 2), OP_a, OP_b, OP_ap, OP_bp)

    problems = (; phantom, OP_a, OP_b, OP_ap, OP_bp, params=(; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting, u_max_braking, α1, α2, α3, β, box_length, box_width, min_long_vel, col_buffer))
end

