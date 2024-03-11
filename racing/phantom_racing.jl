# OP₁	OP₂
#  |	|
# OPᵦ   OPₐ 
# ---
# Information:
# τ₂, OPᵦ -> OP₁
# τ₁, OPₐ -> OP₂
# τ₂ -> OPₐ
# τ₁ -> OPᵦ
#
# ---
# γ∈[0,1]: phantom cost ratio
# Trajectories:
# τ₁ of OP₁: player 1
# τ₂ of OP₂: player 2
# τₐ of OPᵦ: phantom of player 1
# τᵦ of OPₐ: phantom of player 2
#
# ---
# OP1: 
#
# min 	γ * fₐ(τ₁, τᵦ) + (1 - γ) * fₐ(τ₁, τ₂)
# τ₁,τᵦ
# 	s.t. 
#	(1 - γ) * gₐ(τ₁, τ₂) ≥ 0
#	gₐ(τ₁, τᵦ) ≥ 0
# 	τᵦ ∈ argmin 	fᵦ(τ₁, τᵦ),
#			τᵦ
# 		s.t.
#		gₐ(τ₁, τᵦ) ≥ 0
#
# Assume γ=0 -> only Player 1 Vs. Player 2 cost (no phantom cost), all constraints
# Assume γ=1 -> only Player 1 leader Vs. Phantom 2 follower cost , only phantom constraints
#
# ---
# OP2:
#
# min  γ * fᵦ(τₐ, τ₂) + (1 - γ) * fᵦ(τ₁, τ₂) 
# τₐ,τ₂ 
# 	s.t. 
#	(1 - γ) * gᵦ(τ₁, τ₂) ≥ 0
# 	gᵦ(τₐ, τ₂) ≥ 0
# 	τₐ ∈ argmin fₐ(τₐ, τ₂),
#			τₐ
# 		s.t.	
# 		gᵦ(τₐ, τ₂) ≥ 0
#
# Assume γ=0 -> only Player 2 Vs. Player 1 cost (no phantom cost), all constraints
# Assume γ=1 -> only Player 2 leader Vs. Phantom 1 follower cost, only phantom constraints
#
# ---
# OPa: 
#
# min	fₐ(τₐ, τ₂)
# τₐ
#	s.t.
# 	gₐ(τₐ, τ₂) ≥ 0
#
# ---
# OPb:
#
# min	fᵦ(τ₁, τᵦ)
# τᵦ
# 	s.t.
# 	gᵦ(τ₁, τᵦ) ≥ 0
#
# ---
# modes:
# 1. gnep: 
#	OPₐ		OPᵦ
#
# 2. bilevel 1 (γ=1):
#	OP₁	
#	|
#	OPᵦ
#
# 3. bilevel 2 (γ=1):
#	OP₂
#	|
#	OPₐ
#
# 4. phantom (γ<1):
#	OP₁		OP₂
#	|		|
#	OPᵦ		OPₐ 


# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# τⁱ := [xⁱ₁ ... xⁱₜ | uⁱ₁ ... uⁱₜ]
# Z := [τ¹ | τ² | ϕ¹ | ϕ²]
function view_z(z)
    xdim = 4
    udim = 2
    T = Int((length(z) - 2 * xdim) / (4 * (xdim + udim))) # 2 real players, 4 players total
    inds = Dict()
    idx = 0
    for (len, name) in zip([xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim, xdim], ["X1", "U1", "X2", "U2", "Xa", "Ua", "Xb", "Ub", "x01", "x02"])
        inds[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds X1 = @view(z[inds["X1"]])
    @inbounds U1 = @view(z[inds["U1"]])
    @inbounds X2 = @view(z[inds["X2"]])
    @inbounds U2 = @view(z[inds["U2"]])
    @inbounds Xa = @view(z[inds["Xa"]])
    @inbounds Ua = @view(z[inds["Ua"]])
    @inbounds Xb = @view(z[inds["Xb"]])
    @inbounds Ub = @view(z[inds["Ub"]])
    @inbounds x01 = @view(z[inds["x01"]])
    @inbounds x02 = @view(z[inds["x02"]])
    (T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds)
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

function accel_bounds(Xa, Xb, u_max_nominal, u_max_drafting, box_length, box_width)
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

###########
# f and g #
###########
# ego player wants to make forward progress with respect to oppponent, and stay in center of lane
# e = ego
# o = opponent
function f_ego(T, Xe, Ue, Xo; α1, α2, β)
    xdim = 4
    udim = 2
    cost = 0.0

    for t in 1:T
        @inbounds xe = @view(Xe[xdim*(t-1)+1:xdim*t])
        @inbounds xo = @view(Xo[xdim*(t-1)+1:xdim*t])
        @inbounds ue = @view(Ue[udim*(t-1)+1:udim*t])
        cost += α1 * xe[1]^2 + α2 * ue' * ue + β * (xo[4] - xe[4])
    end
    cost
end

function f1(z; α1, α2, β, γ)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # γ * fₐ(τ₁, τᵦ) + (1 - γ) * fₐ(τ₁, τ₂)
    γ * f_ego(T, X1, U1, Xb; α1, α2, β) + (1.0 - γ) * f_ego(T, X1, U1, X2; α1, α2, β)
end

function f2(z; α1, α2, β, γ)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # γ * fᵦ(τₐ, τ₂) + (1 - γ) * fᵦ(τ₁, τ₂) 
    γ * f_ego(T, X2, U2, Xa; α1, α2, β) + (1.0 - γ) * f_ego(T, X2, U2, X1; α1, α2, β)
end

function fa(z; α1, α2, β)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # fₐ(τₐ, τ₂)
    f_ego(T, Xa, Ua, X2; α1, α2, β)
end

function f2(z; α1, α2, β)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # fᵦ(τ₁, τᵦ)
    f_ego(T, Xb, Ub, X1; α1, α2, β)
end


# e = ego
# o = opponent
function g_ego(Xe, Ue, Xo, x0e; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    xdim = 4
    udim = 2

    g_dyn = dyn(Xe, Ue, x0e, Δt, cd)
    g_col = col(Xe, Xo, r)
    h_col = responsibility(Xe, Xo)
    u_max_1, u_max_2, u_max_3, u_max_4 = accel_bounds(Xe,
        Xo,
        u_max_nominal,
        u_max_drafting,
        box_length,
        box_width)
    long_accel = @view(Ue[udim:udim:end])
    lat_accel = @view(Ue[1:udim:end])
    lat_pos = @view(Xe[1:xdim:end])
    long_vel = @view(Xe[xdim:xdim:end])

    [
        g_dyn
        g_col - l.(h_col) .- col_buffer
        lat_accel
        long_accel - u_max_1
        long_accel - u_max_2
        long_accel - u_max_3
        long_accel - u_max_4
        long_accel
        long_vel
        lat_pos
    ]
end

function g1(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer, γ)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # (1 - γ) * gₐ(τ₁, τ₂) ≥ 0
    # gₐ(τ₁, τᵦ) ≥ 0
    [(1 - γ) .* g_ego(X1, U1, X2, x01; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g_ego(X1, U1, Xb, x01; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end

function g2(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer, γ)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # (1 - γ) * gᵦ(τ₁, τ₂) ≥ 0
    # gᵦ(τₐ, τ₂) ≥ 0
    [(1 - γ) .* g_ego(X2, U2, X1, x02; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g_ego(X2, U2, Xa, x02; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end

function ga(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # gₐ(τₐ, τ₂) ≥ 0
    g_ego(Xa, Ua, X2, x01; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

function gb(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z(z)
    # gᵦ(τ₁, τᵦ) ≥ 0
    g_ego(Xb, Ub, X1, x02; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

function setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    β=1e-1, #.5, # sensitive to high values
    γ=0.5,
    cd=0.2, #0.25,
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

    f1_pinned = (z -> f1(z; α1, α2, β, γ))
    f2_pinned = (z -> f2(z; α1, α2, β, γ))
    fa_pinned = (z -> fa(z; α1, α2, β))
    fb_pinned = (z -> fb(z; α1, α2, β))
    g1_pinned = (z -> g1(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer, γ))
    g2_pinned = (z -> g2(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer, γ))
    ga_pinned = (z -> ga(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    gb_pinned = (z -> gb(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    f1_bilevel = (z -> f1(z; α1, α2, β, γ=1.0))
    f2_bilevel = (z -> f2(z; α1, α2, β, γ=1.0))
    g1_bilevel = (z -> g1(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer, γ=1.0))
    g2_bilevel = (z -> g2(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer, γ=1.0))

    lb = [fill(0.0, 4 * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4 * T); fill(-u_max_braking, T); fill(min_long_vel, T); fill(-lat_max, T)]
    ub = [fill(0.0, 4 * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, 4 * T); fill(Inf, T); fill(Inf, T); fill(+lat_max, T)]
    lb_all = [lb; lb]
    ub_all = [ub; ub]

    OP1 = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f1_pinned, g1_pinned, lb_all, ub_all)
    OP2 = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f2_pinned, g2_pinned, lb_all, ub_all)
    OPa = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fa_pinned, ga_pinned, lb, ub)
    OPb = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fb_pinned, gb_pinned, lb, ub)
    OP1_bilevel = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f1_bilevel, g1_bilevel, lb_all, ub_all)
    OP2_bilevel = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, f2_bilevel, g2_bilevel, lb_all, ub_all)
    # modes:
    # 1. gnep: 
    #	OPₐ		OPᵦ
    gnep = [OPa OPb]

    # 2. bilevel 1 (γ=1):
    #	OP₁	
    #	|
    #	OPᵦ
    bilevel1 = [OP1_bilevel; OPb]

    # 3. bilevel 2 (γ=1):
    #	OP₂
    #	|
    #	OPₐ
    bilevel2 = [OP2_bilevel OPa]

    # 4. phantom (γ<1):
    #	OP₁		OP₂
    #	|		|
    #	OPᵦ		OPₐ 
    phantom = EPEC.create_epec((2, 2), OP1, OP2, OPa, OPb) # order is fixed

    function extract(θ, x_inds)
        z = θ[x_inds]
		T, X1, U1, X2, U2, Xa, Ua, Xb, Ub, x01, x02, inds = view_z([z; zeros(2 * xdim)])
        (; X1, U1, X2, U2, Xa, Ua, Xb, Ub)
    end

    params = (; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting, u_max_braking, α1, α2, β, box_length, box_width, min_long_vel, col_buffer)

    (; gnep, bilevel1, bilevel2, phantom, params, extract, OP1, OP2, OPa, OPb, OP1_bilevel, OP2_bilevel)
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

	
    @info "gnep"
    gnep_init = zeros(probs.gnep.top_level.n)
    gnep_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
    gnep_init = [gnep_init; x0]
    gnep_success, θ_gnep = attempt_solve(probs.gnep, gnep_init)

	@info "phantom pain"
    phantom_init = zeros(probs.phantom.top_level.n)
    phantom_init[probs.phantom.x_inds] = [Xa; Ua; Xb; Ub; Xa; Ua; Xb; Ub]
    phantom_init = [phantom_init; x0]

    if (gnep_success)
		phantom_init[probs.phantom.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
		phantom_init[probs.phantom.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
		phantom_init[probs.phantom.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
		phantom_init[probs.phantom.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
		phantom_init[probs.phantom.inds["λ", 3]] = θ_gnep[probs.gnep.inds["λ", 2]]
		phantom_init[probs.phantom.inds["s", 3]] = θ_gnep[probs.gnep.inds["s", 2]]
		phantom_init[probs.phantom.inds["λ", 4]] = θ_gnep[probs.gnep.inds["λ", 1]]
		phantom_init[probs.phantom.inds["s", 4]] = θ_gnep[probs.gnep.inds["s", 1]]
    end
  
    phantom_success, θ_phantom = attempt_solve(probs.phantom, phantom_init)
    show_me(probs.extract(θ_phantom, probs.phantom.x_inds), x0)

    if phantom_success
        @info "phantom success"
        Z = probs.extract(θ_phantom, probs.phantom.x_inds)
    else
        @infiltrate
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

    PA = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    UA = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    PB = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    UB = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    PAp = [Z.Xap[1:4:end] Z.Xap[2:4:end] Z.Xap[3:4:end] Z.Xap[4:4:end]]
    UAp = [Z.Xbp[1:4:end] Z.Xbp[2:4:end] Z.Xbp[3:4:end] Z.Xbp[4:4:end]]
    PBp = [Z.Xbp[1:4:end] Z.Xbp[2:4:end] Z.Xbp[3:4:end] Z.Xbp[4:4:end]]
    UBp = [Z.Ubp[1:2:end] Z.Ubp[2:2:end]]
    #gd = col(Z.Xa, Z.Xb, probs.params.r)
    #h = responsibility(Z.Xa, Z.Xb)
    #gd_both = [gd - l.(h) gd - l.(-h) gd]
    (; PA, PB, PAp, PBp, UA, UB, UAp, UBp)
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
        r_PA = res.PA
        r_UA = res.UA
        r_PB = res.PB
        r_UB = res.UB
        r_PAp = res.PAp
        r_UAp = res.UAp
        r_PBp = res.PBp
        r_UBp = res.UBp
        r = (; PA=r_PA, UA=r_UA, PB=r_PB, UB=r_UB, PAp=r_PAp, UAp=r_UAp, PBp=r_PBp, UBp=r_UBp)
        print("\n")

        # clamp controls and check feasibility
        xa = r.PA[1, :]
        xb = r.PB[1, :]
        ua = r.UA[1, :]
        ub = r.UB[1, :]

        ua_maxes = accel_bounds(xa,
            xb,
            probs.params.u_max_nominal,
            probs.params.u_max_drafting,
            probs.params.box_length,
            probs.params.box_width)

        ub_maxes = accel_bounds(xb,
            xa,
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

        results[t] = (; x0, r.PA, r.PB, r.PAp, r.PBp, r.UA, r.UB, r.UAp, r.UBp)
        x0 = [x0a; x0b]
    end
    results
end
