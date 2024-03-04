# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# xⁱₖ := [p1 p2 v1 v2]
# xₖ = dyn(xₖ₋₁, uₖ; Δt) (pointmass dynamics)
const xdim = 4
const udim = 2

#function view_z(z)
#    T = Int((length(z) - 2 * xdim) / (2 * (xdim + udim)))
#    @inbounds Xa = @view(z[1:xdim*T])
#    @inbounds Ua = @view(z[xdim*T+1:(xdim+udim)*T])
#    @inbounds Xb = @view(z[(xdim+udim)*T+1:(2*xdim+udim)*T])
#    @inbounds Ub = @view(z[(2*xdim+udim)*T+1:2*(xdim+udim)*T])
#    @inbounds x0a = @view(z[2*(xdim+udim)*T+1:2*(xdim+udim)*T+xdim])
#    @inbounds x0b = @view(z[2*(xdim+udim)*T+xdim+1:2*(xdim+udim)*T+2*xdim])
#    (T, Xa, Ua, Xb, Ub, x0a, x0b)
#end

function view_z(z)
    xdim = 4
    udim = 2
    T = Int((length(z) - 2 * xdim) / (4 * (xdim + udim))) # 2 real players, 4 players total
    indices = Dict()
    idx = 0
    for (len, name) in zip([xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim, xdim], ["Xa", "Ua", "Xb", "Ub", "Xap", "Uap", "Xbp", "Ubp", "x0a", "x0b"])
        indices[name] = (idx+1):(idx+len)
        idx += len
    end
    #for (len, name) in zip([xdim * T, udim * T, xdim * T, udim * T, xdim, xdim], ["Xa", "Ua", "Xb", "Ub", "x0a", "x0b"])
    #    indices[name] = (idx+1):(idx+len)
    #    idx += len
    #end
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
    #(T, Xa, Ua, Xb, Ub, x0a, x0b, indices)
end

# P1 wants to make forward progress and stay in center of lane.
function f1(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    #T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    #@inbounds Xa = @view(Z[1:4*T])
    #@inbounds Ua = @view(Z[4*T+1:6*T])
    #@inbounds Xb = @view(Z[6*T+1:10*T])
    #@inbounds Ub = @view(Z[10*T+1:12*T])

    T, Xa, Ua, Xb, Ub, x0a, x0b = view_z(Z)

    #@infiltrate
    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        cost += α1 * xa[1]^2 + α2 * ua' * ua - α3 * xa[4] + β * xb[4]
        #cost += α1 * xa[1]^2 + α2 * ua' * ua - α3 * xa[4] + β * xb[4]
    end
    cost
end

# P2 wants to make forward progress and stay in center of lane.
function f2(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    #T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    #@inbounds Xa = @view(Z[1:4*T])
    #@inbounds Ua = @view(Z[4*T+1:6*T])
    #@inbounds Xb = @view(Z[6*T+1:10*T])
    #@inbounds Ub = @view(Z[10*T+1:12*T])
    T, Xa, Ua, Xb, Ub, x0a, x0b = view_z(Z)

    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        cost += α1 * xb[1]^2 + α2 * ub' * ub - α3 * xb[4] + β * xa[4]
        #cost += α1 * xb[1]^2 + α2 * ub' * ub - α3 * xb[4] + β * xa[4]
    end
    cost
end

## P1 wants to make forward progress and stay in center of lane.
#function f1(Z; α1=1.0, α2=0.0, β=1.0)
#    #T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
#    #@inbounds Xa = @view(Z[1:4*T])
#    #@inbounds Ua = @view(Z[4*T+1:6*T])
#    #@inbounds Xb = @view(Z[6*T+1:10*T])
#    #@inbounds Ub = @view(Z[10*T+1:12*T])

#    T, Xa, Ua, Xb, Ub, x0a, x0b = view_z(Z)

#    #@infiltrate
#    cost = 0.0

#    for t in 1:T
#        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
#        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
#        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
#        #cost += α1 * xa[1]^2 + α2 * ua' * ua + β * (xb[4] - xa[4])
#        cost += α1 * xa[1]^2 + α2 * ua' * ua + β * xb[4] - β * xa[4]
#    end
#    cost
#end

## P2 wants to make forward progress and stay in center of lane.
#function f2(Z; α1=1.0, α2=0.0, β=1.0)
#    #T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
#    #@inbounds Xa = @view(Z[1:4*T])
#    #@inbounds Ua = @view(Z[4*T+1:6*T])
#    #@inbounds Xb = @view(Z[6*T+1:10*T])
#    #@inbounds Ub = @view(Z[10*T+1:12*T])
#    T, Xa, Ua, Xb, Ub, x0a, x0b = view_z(Z)

#    cost = 0.0

#    for t in 1:T
#        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
#        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
#        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
#        #cost += α1 * xb[1]^2 + α2 * ub' * ub + β * (xa[4] - xb[4])
#        cost += α1 * xb[1]^2 + α2 * ub' * ub + β * xa[4] - β * xb[4]
#    end
#    cost
#end


function f_a(z; α1, α2, α3, β, γ)
    #T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    # γ f(τ_a, τ_b) + (1 - γ) f(τ_a, ϕ_b)
    γ * f1(z; α1, α2, α3, β) + (1.0 - γ) * f1(z; α1, α2, α3, β)
end

function f_b(z; α1, α2, α3, β, γ)
    #T, Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp, x0a, x0b, indices = view_z(z)

    # γ f(τ_a, τ_b) + (1 - γ) f(ϕ_a, τ_b)
    γ * f2(z; α1, α2, α3, β) + (1.0 - γ) * f2(z; α1, α2, α3, β)
end

function f_ap(z; α1, α2, α3, β)
    f1(z; α1, α2, α3, β)
end

function f_bp(z; α1, α2, α3, β)
    f2(z; α1, α2, α3, β)
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
    #T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    #@inbounds Xa = @view(Z[1:4*T])
    #@inbounds Ua = @view(Z[4*T+1:6*T])
    #@inbounds Xb = @view(Z[6*T+1:10*T])
    #@inbounds Ub = @view(Z[10*T+1:12*T])
    #@inbounds x0a = @view(Z[12*T+1:12*T+4])
    #@inbounds x0b = @view(Z[12*T+5:12*T+8])
    T, Xa, Ua, Xb, Ub, x0a, x0b = view_z(Z)


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
    #T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    #@inbounds Xa = @view(Z[1:4*T])
    #@inbounds Ua = @view(Z[4*T+1:6*T])
    #@inbounds Xb = @view(Z[6*T+1:10*T])
    #@inbounds Ub = @view(Z[10*T+1:12*T])
    #@inbounds x0a = @view(Z[12*T+1:12*T+4])
    #@inbounds x0b = @view(Z[12*T+5:12*T+8])
    T, Xa, Ua, Xb, Ub, x0a, x0b = view_z(Z)

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
    α1=1e-3,
    α2=1e-4,
    α3=1e-1,
    β=1e-1, #.5, # sensitive to high values
    cd=0.2, #0.25,
    γ=1.0,
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=2.0,
    lat_max=2.0,
    u_max_braking=2 * u_max_drafting,
    min_long_vel=-5.0,
    col_buffer=r / 5)

    xdim = 4
    udim = 2
    lb = [fill(0.0, 4 * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4 * T); fill(-u_max_braking, T); fill(min_long_vel, T); fill(-lat_max, T)]
    ub = [fill(0.0, 4 * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, 4 * T); fill(Inf, T); fill(Inf, T); fill(+lat_max, T)]

    #f1_pinned = (z -> f1(z; α1, α2, α3, β))
    #f2_pinned = (z -> f2(z; α1, α2, α3, β))
    f_a_pinned = (z -> f_a(z; α1, α2, α3, β, γ))
    f_b_pinned = (z -> f_b(z; α1, α2, α3, β, γ))
    f_ap_pinned = (z -> f_ap(z; α1, α2, α3, β))
    f_bp_pinned = (z -> f_bp(z; α1, α2, α3, β))

    g1_pinned = (z -> g1(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    g2_pinned = (z -> g2(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))

    OP1 = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_a_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_b_pinned, g2_pinned, lb, ub)
    OP3 = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_ap_pinned, g1_pinned, lb, ub)
    OP4 = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f_bp_pinned, g2_pinned, lb, ub)


    sp_a = EPEC.create_epec((1, 0), OP1)
    gnep = [OP1 OP2 OP3 OP4]
    bilevel = [OP1; OP2]
    phantom = [OP1 OP2; OP3 OP4]

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
    function extract_phantom(θ)
        xdim = 4
        udim = 2
        z = θ[phantom.x_inds]
        T = Int((length(z)) / (4 * (xdim + udim))) # 2 real players, 4 players total
        indices = Dict()
        idx = 0
        for (len, name) in zip([xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T], ["Xa", "Ua", "Xb", "Ub", "Xap", "Uap", "Xbp", "Ubp"])
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
        (; Xa, Ua, Xb, Ub, Xap, Uap, Xbp, Ubp)
    end

    problems = (; sp_a, gnep, bilevel, phantom, extract_gnep, extract_bilevel, extract_phantom, OP1, OP2, params=(; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting, u_max_braking, α1, α2, α3, β, box_length, box_width, min_long_vel, col_buffer))
end

function attempt_solve(prob, init)
    success = true
    result = init
    try
        result = solve(prob, init)
    catch err
        println(err)
        success = false
    end
    (success, result)
end

function solve_seq(probs, x0)
    dummy_init = zeros(probs.gnep.top_level.n)
    X = dummy_init[probs.gnep.x_inds]
    #T = Int(length(X) / 12)
    T = probs.params.T
    Δt = probs.params.Δt
    cd = probs.params.cd
    Xa = []
    Ua = []
    Xb = []
    Ub = []
    x0a = x0[1:4]
    x0b = x0[5:8]
    xa = x0a
    xb = x0b
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
    dummy_init = [Xa; Ua; Xb; Ub]
    Z = (; Xa, Ua, Xb, Ub, x0a, x0b)
    #show_me(dummy_init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    phantom_init = zeros(probs.phantom.top_level.n)
    phantom_init[probs.phantom.x_inds] = [Xa; Ua; Xb; Ub; Xa; Ua; Xb; Ub]
    phantom_init = [phantom_init; x0]

    @info "nep"
    gnep_init = zeros(probs.gnep.top_level.n)
    gnep_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub; Xa; Ua; Xb; Ub]
    #gnep_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
    gnep_init = [gnep_init; x0]
    success, θ = attempt_solve(probs.gnep, gnep_init)

    @info "phantom pain"
    phantom_success, θ_phantom = attempt_solve(probs.phantom, phantom_init)

    if phantom_success
        #@info "bilevel success 1"
        Z = probs.extract_phantom(θ_phantom)
    else
        want_gnep = true
    end

    #@info "Solving gnep.."
    #gnep_init = zeros(probs.gnep.top_level.n)
    #gnep_init[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; θ_sp_b[probs.sp_a.x_inds]]
    #gnep_init[probs.bilevel.inds["λ", 1]] = θ_sp_a[probs.gnep.inds["λ", 1]]
    #gnep_init[probs.bilevel.inds["s", 1]] = θ_sp_a[probs.gnep.inds["s", 1]]
    #gnep_init[probs.bilevel.inds["λ", 2]] = θ_sp_b[probs.gnep.inds["λ", 1]]
    #gnep_init[probs.bilevel.inds["s", 2]] = θ_sp_b[probs.gnep.inds["s", 1]]
    #gnep_init = [gnep_init; x0]
    ##show_me(gnep_init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    #θ_gnep = gnep_init # fall back
    #try
    #    θ_gnep = solve(probs.gnep, gnep_init)
    #    #show_me(θ_gnep, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    #catch err
    #    println(err)
    #    @info "Fell back to gnep init.."
    #end

    #@info "Solving bilevel a.."
    #bilevel_init = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
    #bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
    #bilevel_init[probs.bilevel.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
    #bilevel_init[probs.bilevel.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
    #bilevel_init[probs.bilevel.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
    #bilevel_init[probs.bilevel.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
    #bilevel_init[probs.bilevel.inds["w", 0]] = θ_gnep[probs.gnep.inds["w", 0]]

    #θ_bilevel = bilevel_init # fall back

    #try
    #    θ_bilevel = solve(probs.bilevel, bilevel_init)
    #    #show_me(θ_bilevel, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    #    Z = probs.extract_bilevel(θ_bilevel)
    #catch err
    #    println(err)
    #    @info "Fell back to gnep init.."
    #end

    #Z = probs.extract_bilevel(θ_bilevel)
    #P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    #U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    #P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    #U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]


    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    #gd = col(Z.Xa, Z.Xb, probs.params.r)
    #h = responsibility(Z.Xa, Z.Xb)
    #gd_both = [gd - l.(h) gd - l.(-h) gd]
    (; P1, P2, U1, U2)
end

# Solve mode:
#						P1:						
#				SP  NE   P1-leader  P1-follower
#			 SP 1              
# P2:		 NE 2   3
#	  P2-Leader 4   5   6 
#   P2-Follower 7   8   9		    10
#
function solve_simulation(probs, T; x0=[0, 0, 0, 7, 0.1, -2.21, 0, 7])
    lat_max = probs.params.lat_max
    status = "ok"
    x0a = x0[1:4]
    x0b = x0[5:8]

    results = Dict()
    for t = 1:T
        #@info "Sim timestep $t:"
        print("Sim timestep $t: ")
        # check initial condition feasibility
        is_x0_infeasible = false

        if col(x0a, x0b, probs.params.r)[1] <= 0 - 1e-4
            status = "Infeasible initial condition: Collision"
            is_x0_infeasible = true
        elseif x0a[1] < -lat_max - 1e-4 || x0a[1] > lat_max + 1e-4 || x0b[1] < -lat_max - 1e-4 || x0b[1] > lat_max + 1e-4
            status = "Infeasible initial condition: Out of lanes"
            is_x0_infeasible = true
        elseif x0a[4] < probs.params.min_long_vel - 1e-4 || x0b[4] < probs.params.min_long_vel - 1e-4
            status = "Infeasible initial condition: Invalid velocity"
            is_x0_infeasible = true
        end

        if is_x0_infeasible
            # currently status isn't saved
            print(status)
            print("\n")
            results[t] = (; x0, P1=repeat(x0', 10, 1), P2=repeat(x0', 10, 1))
            break
        end

        res = solve_seq(probs, x0)
        r_P1 = res.P1
        r_U1 = res.U1
        r_P2 = res.P2
        r_U2 = res.U2
        r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        print("\n")

        # clamp controls and check feasibility
        xa = r.P1[1, :]
        xb = r.P2[1, :]
        ua = r.U1[1, :]
        ub = r.U2[1, :]

        ua_maxes = accel_bounds_1(xa,
            xb,
            probs.params.u_max_nominal,
            probs.params.u_max_drafting,
            probs.params.box_length,
            probs.params.box_width)

        ub_maxes = accel_bounds_2(xa,
            xb,
            probs.params.u_max_nominal,
            probs.params.u_max_drafting,
            probs.params.box_length,
            probs.params.box_width)

        ua[1] = minimum([maximum([ua[1], -probs.params.u_max_nominal]), probs.params.u_max_nominal])
        ub[1] = minimum([maximum([ub[1], -probs.params.u_max_nominal]), probs.params.u_max_nominal])
        ua[2] = minimum([maximum([ua[2], -probs.params.u_max_braking]), ua_maxes[1][1], ua_maxes[2][1], ua_maxes[3][1], ua_maxes[4][1]])
        ub[2] = minimum([maximum([ub[2], -probs.params.u_max_braking]), ub_maxes[1][1], ub_maxes[2][1], ub_maxes[3][1], ub_maxes[4][1]])

        x0a = pointmass(xa, ua, probs.params.Δt, probs.params.cd)
        x0b = pointmass(xb, ub, probs.params.Δt, probs.params.cd)

        results[t] = (; x0, r.P1, r.P2, r.U1, r.U2)
        x0 = [x0a; x0b]
    end
    results
end
