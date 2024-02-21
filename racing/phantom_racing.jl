# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# τⁱ := [xⁱ₁ ... xⁱₜ | uⁱ₁ ... uⁱₜ]
# Z := [τ¹ | τ² | ϕ¹ | ϕ²]

function view_Z(Z; xdim=4, udim=2)
    T = Int((length(Z) - 2 * xdim) / (4 * (xdim + udim)))
    indices = Dict()
    idx = 0
    for (len, name) in zip([xdim*T, udim*T, xdim*T, udim*T,  xdim*T, udim*T, xdim*T, udim*T, xdim, xdim], ["Xa", "Ua", "Xb", "Ub", "Xpa", "Upa", "Xpb", "Upb", "x0a", "x0b"])
        indices[name] = (idx+1):(idx+len)
        idx += len
    end 
    @inbounds Xa = @view(Z[indices["Xa"]])
    @inbounds Ua = @view(Z[indices["Ua"]])
    @inbounds Xb = @view(Z[indices["Xb"]])
    @inbounds Ub = @view(Z[indices["Ub"]])
    @inbounds Xpa = @view(Z[indices["Xpa"]])
    @inbounds Upa = @view(Z[indices["Upa"]])
    @inbounds Xpb = @view(Z[indices["Xpb"]])
    @inbounds Upb = @view(Z[indices["Upb"]])
    @inbounds x0a = @view(Z[indices["x0a"]])
    @inbounds x0b = @view(Z[indices["x0b"]])
    (T, Xa, Ua, Xb, Ub, Xpa, Upa, Xpb, Upb, x0a, x0b, indices)
end

function f1(Z; γ=1.0, α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T, Xa, Ua, Xb, Ub, Xpa, Upa, Xpb, Upb, x0a, x0b, indices = view_Z(Z)

    γ * fa(T, Xa, Ua, Xb; α1, α2, α3, β) + (1.0 - γ) * fa(T, Xa, Ua, Xpb; α1, α2, α3, β)
end

function f2(Z; γ=1.0, α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T, Xa, Ua, Xb, Ub, Xpa, Upa, Xpb, Upb, x0a, x0b, indices = view_Z(Z)

    γ * fb(T, Xb, Ub, Xa; α1, α2, α3, β) + (1.0 - γ) * fb(T, Xb, Ub, Xpa; α1, α2, α3, β)
end

# P1 wants to make forward progress and stay in center of lane.
function fa(T, Xa, Ua, Xb; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        cost += α1 * xa[1]^2 + α2 * ua' * ua - α3 * xa[4] + β * xb[4]
    end
    cost
end

# P2 wants to make forward progress and stay in center of lane.
function fb(T, Xb, Ub, Xa; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        cost += α1 * xb[1]^2 + α2 * ub' * ub - α3 * xb[4] + β * xa[4]
    end
    cost
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
    box_width=1.0,
    col_buffer=r / 5)
    T, Xa, Ua, Xb, Ub, Xpa, Upa, Xpb, Upb, x0a, x0b, indices = view_Z(Z)

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
    long_vel = @view(Xa[4:4:end])

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

function g2(Z,
    Δt=0.1,
    r=1.0,
    cd=1.0,
    u_max_nominal=2.0,
    u_max_drafting=5.0,
    box_length=3.0,
    box_width=1.0,
    col_buffer=r / 5)
    T, Xa, Ua, Xb, Ub, Xpa, Upa, Xpb, Upb, x0a, x0b, indices = view_Z(Z)

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
    long_vel = @view(Xb[4:4:end])

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

function setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=0.01,
    α2=0.001,
    α3=0.0,
    β=1e3,
    cd=0.2,
    u_max_nominal=2.0,
    u_max_drafting=5.0,
    u_max_braking=2 * u_max_drafting,
    box_length=3.0,
    box_width=1.0,
    lat_max=5.0,
    min_long_vel=-5.0,
    col_buffer=r / 5)

    lb = [fill(0.0, 4 * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4 * T); fill(-u_max_braking, T); fill(min_long_vel, T); fill(-lat_max, T)]
    ub = [fill(0.0, 4 * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, 4 * T); fill(Inf, T); fill(Inf, T); fill(+lat_max, T)]

    f1_pinned = (z -> f1(z; α1, α2, α3, β))
    f2_pinned = (z -> f2(z; α1, α2, α3, β))
    g1_pinned = (z -> g1(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    g2_pinned = (z -> g2(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))

    OP1 = OptimizationProblem(12 * T + 8, 1:6*T, f1_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(12 * T + 8, 1:6*T, f2_pinned, g2_pinned, lb, ub)
    OP1_phantom = OptimizationProblem(12 * T + 8, 1:6*T, f1_pinned, g1_pinned, lb, ub)
    OP2_phantom = OptimizationProblem(12 * T + 8, 1:6*T, f2_pinned, g2_pinned, lb, ub)

    EPEC.create_epec((2, 2), OP1, OP2, OP1_phantom, OP2_phantom)

    #sp_a = EPEC.create_epec((1, 0), OP1)
    #gnep = [OP1 OP2]
    #bilevel = [OP1; OP2]
    phantom = [OP1, OP2; OP2_phantom, OP1_phantom]

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
    problems = (; sp_a, gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2, params=(; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting, u_max_braking, α1, α2, α3, β, box_length, box_width, min_long_vel, col_buffer))
end