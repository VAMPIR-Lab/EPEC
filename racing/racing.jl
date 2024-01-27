# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# xⁱₖ := [p1 p2 v1 v2]
# xₖ = dyn(xₖ₋₁, uₖ; Δt) (pointmass dynamics)
const xdim = 4
const udim = 2

# P1 wants to make forward progress and stay in center of lane.
function f1(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    #@infiltrate
    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ua = @view(Ua[udim*(t-1)+1:udim*t])
        cost += α1 * xa[1]^2 + α2 * ua' * ua - α3 * xa[4] + β * xb[2]
    end
    cost
end

# P2 wants to make forward progress and stay in center of lane.
function f2(Z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T = Int((length(Z) - 8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])

    cost = 0.0

    for t in 1:T
        @inbounds xa = @view(Xa[xdim*(t-1)+1:xdim*t])
        @inbounds xb = @view(Xb[xdim*(t-1)+1:xdim*t])
        @inbounds ub = @view(Ub[udim*(t-1)+1:udim*t])
        cost += α1 * xb[1]^2 + α2 * ub' * ub - α3 * xb[4] + β * xa[2]
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
    long_vel = @view(Xa[4:4:end])

    [g_dyn
        g_col - l.(h_col)
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
    long_vel = @view(Xb[4:4:end])

    [g_dyn
        g_col - l.(h_col)
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
    min_long_vel=-5.0)

    lb = [fill(0.0, 4 * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4 * T); fill(-u_max_braking, T); fill(min_long_vel, T); fill(-lat_max, T)]
    ub = [fill(0.0, 4 * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, 4 * T); fill(Inf, T); fill(Inf, T); fill(+lat_max, T)]

    f1_pinned = (z -> f1(z; α1, α2, α3, β))
    f2_pinned = (z -> f2(z; α1, α2, α3, β))
    g1_pinned = (z -> g1(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width))
    g2_pinned = (z -> g2(z, Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width))

    OP1 = OptimizationProblem(12 * T + 8, 1:6*T, f1_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(12 * T + 8, 1:6*T, f2_pinned, g2_pinned, lb, ub)

    sp_a = EPEC.create_epec((1, 0), OP1)
    sp_b = EPEC.create_epec((1, 0), OP2)
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
    problems = (; sp_a, sp_b, gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2, params=(; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting, u_max_braking, α1, α2, α3, β, box_length, box_width, min_long_vel))
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

function solve_seq_adaptive(probs, x0; only_want_gnep=false, only_want_sp=false, try_bilevel_first=false, try_gnep_first=true)
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
        ua = [0; 0]#-cd * xa[3:4]
        ub = [0; 0]#-cd * xb[3:4]
        xa = pointmass(xa, ua, Δt, cd)
        xb = pointmass(xb, ub, Δt, cd)
        append!(Ua, ua)
        append!(Ub, ub)
        append!(Xa, xa)
        append!(Xb, xb)
    end
    # dummy init
    Z = (; Xa, Ua, Xb, Ub, x0a, x0b)
    valid_Z = Dict()
    valid_Z[8] = Z
    #dummy_init = zeros(probs.gnep.top_level.n)
    #dummy_init = [Xa; Ua; Xb; Ub]
    #show_me(dummy_init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    bilevel_success = false
    gnep_success = false
    sp_success = false
    preference_id = 0

    θ_bilevel = []
    θ_gnep = []
    θ_sp_a = []
    θ_sp_b = []

    if only_want_gnep
        want_gnep = true
        want_bilevel = false
    else
        want_gnep = false
        want_bilevel = true
    end

    if !try_bilevel_first && !try_gnep_first
        want_sp = true
    else
        want_sp = false
    end

    if only_want_sp
        want_sp = true
        want_gnep = false
        want_bilevel = false
        try_bilevel_first = false
        try_gnep_first = false
    end

    # preference order
    # 1. bilevel
    # 2. gnep->bilevel
    # 3. sp->gnep->bilevel
    # 4. sp->bilevel
    # 5. gnep
    # 6. sp->gnep
    # 7. sp
    # 8. dummy
    if try_bilevel_first && want_bilevel
        # initialized from dummy:
        bilevel_init = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
        bilevel_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
        bilevel_init = [bilevel_init; x0]
        #@info "(1) bilevel..."
        bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

        if bilevel_success
            #@info "bilevel success 1"
            valid_Z[1] = probs.extract_bilevel(θ_bilevel)
        else
            want_gnep = true
        end
    elseif want_bilevel
        want_gnep = true
    end

    if try_gnep_first && want_gnep
        # initialized from dummy:
        gnep_init = zeros(probs.gnep.top_level.n)
        gnep_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
        gnep_init = [gnep_init; x0]
        #@info "(5) gnep..."
        gnep_success, θ_gnep = attempt_solve(probs.gnep, gnep_init)

        if gnep_success
            #@info "gnep success 5"
            valid_Z[5] = probs.extract_gnep(θ_gnep)

            if want_bilevel
                # initialized from gnep:
                bilevel_init = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
                bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
                bilevel_init[probs.bilevel.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
                bilevel_init[probs.bilevel.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
                bilevel_init[probs.bilevel.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
                bilevel_init[probs.bilevel.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
                bilevel_init[probs.bilevel.inds["w", 0]] = θ_gnep[probs.gnep.inds["w", 0]]
                #@info "(2) gnep->bilevel..."
                bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

                if bilevel_success
                    #@info "gnep->bilevel success 2"
                    valid_Z[2] = probs.extract_bilevel(θ_bilevel)
                else
                    want_sp = true
                end
            end
        else
            want_sp = true
        end
    end

    if want_sp
        # initialized from dummy:
        # [!] need to be changed in problems.jl so this isn't such a mess
        # !!! 348 vs 568 breaks it
        # !!! Ordering of x_w changes because:
        # 1p: [x1=>1:60 λ1,s1=>61:280, w=>281:348]
        # 2p: [x1,x2=>1:120, λ1,λ2,s1,s2=>121:560 w=>561:568
        sp_a_init = zeros(probs.gnep.top_level.n)
        sp_a_init[probs.sp_a.x_inds] = [Xa; Ua]
        sp_a_init[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+6*T] = [Xb; Ub]
        sp_a_init[probs.sp_a.top_level.n+6*T+1:probs.sp_a.top_level.n+6*T+8] = x0 # right now parameters are expected to be contiguous
        #sp_a_init = [sp_a_init; x0]; 

        #@info "(7a) sp_a..."
        θ_sp_a_success, θ_sp_a = attempt_solve(probs.sp_a, sp_a_init)
        #show_me([θ_sp_a[probs.sp_a.x_inds]; θ_sp_a[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        # if it fails:
        #show_me([safehouse.θ_out[probs.sp_a.x_inds]; safehouse.w[1:60]], safehouse.w[61:68]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        # swapping b for a:
        sp_b_init = zeros(probs.gnep.top_level.n)
        sp_b_init[probs.sp_b.x_inds] = [Xb; Ub]
        sp_b_init[probs.sp_b.top_level.n+1:probs.sp_b.top_level.n+6*T] = [Xa; Ua]
        sp_b_init[probs.sp_b.top_level.n+6*T+1:probs.sp_b.top_level.n+6*T+8] = [x0[5:8]; x0[1:4]]
        #θ_sp_b = solve(probs.sp_b, sp_b_init) # doesn't work because x_w = [xb xa x0]

        #@info "(7b) sp_b..."
        θ_sp_b_success, θ_sp_b = attempt_solve(probs.sp_a, sp_b_init)
        #show_me([θ_sp_b[probs.sp_b.top_level.n+1:probs.sp_b.top_level.n+60]; θ_sp_b[probs.sp_b.x_inds]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        # if it fails:
        #show_me([safehouse.w[1:60]; safehouse.θ_out[probs.sp_b.x_inds]], [safehouse.w[65:68]; safehouse.w[61:64]]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)    

        sp_success = θ_sp_a_success && θ_sp_b_success

        if sp_success
            #@info "sp success 7"
            gnep_like = zeros(probs.gnep.top_level.n)
            gnep_like[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; θ_sp_b[probs.sp_a.x_inds]]
            gnep_like = [gnep_like; x0]
            valid_Z[7] = probs.extract_gnep(gnep_like)

            if want_gnep
                # initialized from θ_sp_a and θ_sp_b:
                gnep_init = zeros(probs.gnep.top_level.n)
                gnep_init[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; θ_sp_b[probs.sp_a.x_inds]]
                gnep_init[probs.bilevel.inds["λ", 1]] = θ_sp_a[probs.gnep.inds["λ", 1]]
                gnep_init[probs.bilevel.inds["s", 1]] = θ_sp_a[probs.gnep.inds["s", 1]]
                gnep_init[probs.bilevel.inds["λ", 2]] = θ_sp_b[probs.gnep.inds["λ", 1]]
                gnep_init[probs.bilevel.inds["s", 2]] = θ_sp_b[probs.gnep.inds["s", 1]]
                gnep_init = [gnep_init; x0]
                #@info "(6) sp->gnep..."
                gnep_success, θ_gnep = attempt_solve(probs.gnep, gnep_init)

                if gnep_success
                    #@info "sp->gnep success 6"
                    want_gnep = false
                    valid_Z[6] = probs.extract_gnep(θ_gnep)

                    if want_bilevel
                        # initialized from gnep which was initialized from sp:
                        bilevel_init = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
                        bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
                        bilevel_init[probs.bilevel.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
                        bilevel_init[probs.bilevel.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
                        bilevel_init[probs.bilevel.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
                        bilevel_init[probs.bilevel.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
                        bilevel_init[probs.bilevel.inds["w", 0]] = θ_gnep[probs.gnep.inds["w", 0]]
                        #@info "(3) sp->gnep->bilevel..."
                        bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

                        if bilevel_success
                            #@info "sp->gnep->bilevel success 3"
                            valid_Z[3] = probs.extract_bilevel(θ_bilevel)
                        end
                    end
                else
                    if want_bilevel
                        # initialized from sp:
                        bilevel_init = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
                        bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
                        bilevel_init[probs.bilevel.inds["λ", 1]] = θ_sp_a[probs.gnep.inds["λ", 1]]
                        bilevel_init[probs.bilevel.inds["s", 1]] = θ_sp_a[probs.gnep.inds["s", 1]]
                        bilevel_init[probs.bilevel.inds["λ", 2]] = θ_sp_b[probs.gnep.inds["λ", 1]]
                        bilevel_init[probs.bilevel.inds["s", 2]] = θ_sp_b[probs.gnep.inds["s", 1]]
                        bilevel_init[probs.bilevel.inds["w", 0]] = x0
                        #bilevel_init = [bilevel_init; x0]
                        #@info "(4) sp->bilevel..."
                        bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

                        if bilevel_success
                            #@info "sp->bilevel success 4"
                            valid_Z[4] = probs.extract_bilevel(θ_bilevel)
                        end
                    end
                end
            end
        end
    end

    sorted_Z = sort(collect(valid_Z), by=x -> x[1])
    lowest_preference, Z = sorted_Z[1] # best pair

    if lowest_preference < 8
        @info "Success $lowest_preference"
    else
        @info "Fail $lowest_preference"
    end


    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    gd = col(Z.Xa, Z.Xb, probs.params.r)
    h = responsibility(Z.Xa, Z.Xb)
    gd_both = [gd - l.(h) gd - l.(-h) gd]
    (; P1, P2, gd_both, h, U1, U2, lowest_preference, sorted_Z)
end

function solve_simulation(probs, T; x0=[0, 0, 0, 7, 0.1, -2.21, 0, 7], only_want_gnep=false, only_want_sp=false)
    lat_max = probs.params.lat_max
    status = "ok"
    x0a = x0[1:4]
    x0b = x0[5:8]

    results = Dict()
    for t = 1:T
        @info "Sim timestep $t:"
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
            @info(status)
            break
        end

        r = solve_seq_adaptive(probs, x0; only_want_gnep=only_want_gnep, only_want_sp=only_want_sp)

        #costs = [OP.f(z) for OP in probs.gnep.OPs]
        lowest_preference, Z = r.sorted_Z[1]
        z = [Z.Xa; Z.Ua; Z.Xb; Z.Ub; x0]
        feasible_arr = [[OP.l .- 1e-4 .<= OP.g(z) .<= OP.u .+ 1e-4] for OP in probs.gnep.OPs]
        feasible = all(all(feasible_arr[i][1]) for i in 1:2) # I think this is fine


        if !feasible || any(r.P1[:, 4] .< probs.params.min_long_vel - 1e-4) || any(r.P2[:, 4] .< probs.params.min_long_vel - 1e-4) || any(r.P1[:, 1] .< -lat_max - 1e-4) || any(r.P2[:, 1] .< -lat_max - 1e-4) || any(r.P1[:, 1] .> 1e-4 + lat_max) || any(r.P2[:, 1] .> lat_max + 1e-4)
            if (feasible)
                # this must never trigger
                @infiltrate
            end
            status = "Invalid solution"
            @info "Invalid solution"
        end
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

        results[t] = (; x0, r.P1, r.P2, r.U1, r.U2, r.gd_both, r.h, r.lowest_preference, r.sorted_Z)
        x0 = [x0a; x0b]
    end
    results
end

function animate(probs, sim_results; save=false, filename="test.mp4", sleep_duration=1e-2)
    rad = sqrt(probs.params.r) / 2
    lat = probs.params.lat_max + rad
    (f, ax, XA, XB, lat) = visualize(; rad=rad, lat=lat)
    display(f)
    T = length(sim_results)

    if save
        record(f, filename, 1:T; framerate=20) do t
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = string(t)
        end
    else
        for t in 1:T
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = string(t)
            sleep(sleep_duration)
        end
    end
end

function visualize(; rad=0.5, lat=6.0)
    f = Figure(resolution=(1000, 1000))
    ax = Axis(f[1, 1], aspect=DataAspect())

    GLMakie.lines!(ax, [-lat, -lat], [-10.0, 1000.0], color=:black)
    GLMakie.lines!(ax, [+lat, +lat], [-10.0, 1000.0], color=:black)

    XA = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:10)
    XB = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:10)

    circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]
    GLMakie.lines!(ax, @lift(circ_x .+ $(XA[0][1])), @lift(circ_y .+ $(XA[0][2])), color=:blue, linewidth=5)
    GLMakie.lines!(ax, @lift(circ_x .+ $(XB[0][1])), @lift(circ_y .+ $(XB[0][2])), color=:red, linewidth=5)

    for t in 1:10
        GLMakie.lines!(ax, @lift(circ_x .+ $(XA[t][1])), @lift(circ_y .+ $(XA[t][2])), color=:blue, linewidth=2, linestyle=:dash)
        GLMakie.lines!(ax, @lift(circ_x .+ $(XB[t][1])), @lift(circ_y .+ $(XB[t][2])), color=:red, linewidth=2, linestyle=:dash)
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

    GLMakie.xlims!(ax, -2 * lat, 2 * lat)
    GLMakie.ylims!(ax, x0[6] - lat, maximum([P1[T, 2], P2[T, 2]]) + lat)
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

#
#
#
# deprecated:

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
    dummy_init = [Xa; Ua; Xb; Ub]
    #show_me(dummy_init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    @info "Solving singleplayer a.."
    #!!! 348 vs 568 breaks it
    # !!! Ordering of x_w changes because:
    # 1p: [x1=>1:60 λ1,s1=>61:280, w=>281:348]
    # 2p: [x1,x2=>1:120, λ1,λ2,s1,s2=>121:560 w=>561:568
    sp_a_init = zeros(probs.gnep.top_level.n)
    sp_a_init[probs.sp_a.x_inds] = [Xa; Ua]
    sp_a_init[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60] = [Xb; Ub]
    sp_a_init[probs.sp_a.top_level.n+61:probs.sp_a.top_level.n+68] = x0 # right now parameters are expected to be contiguous
    #sp_a_init = [sp_a_init; x0]; 
    θ_sp_a = solve(probs.sp_a, sp_a_init)
    #show_me([θ_sp_a[probs.sp_a.x_inds]; θ_sp_a[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    # if it fails:
    #show_me([safehouse.θ_out[probs.sp_a.x_inds]; safehouse.w[1:60]], safehouse.w[61:68]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    @info "Solving singleplayer b.."
    # swapping b for a
    sp_b_init = zeros(probs.gnep.top_level.n)
    sp_b_init[probs.sp_b.x_inds] = [Xb; Ub]
    sp_b_init[probs.sp_b.top_level.n+1:probs.sp_b.top_level.n+60] = [Xa; Ua]
    sp_b_init[probs.sp_b.top_level.n+61:probs.sp_b.top_level.n+68] = [x0[5:8]; x0[1:4]]
    #θ_sp_b = solve(probs.sp_b, sp_b_init) # doesn't work because x_w = [xb xa x0], need to be changed in problems.jl
    θ_sp_b = solve(probs.sp_a, sp_b_init)
    #show_me([θ_sp_b[probs.sp_b.top_level.n+1:probs.sp_b.top_level.n+60]; θ_sp_b[probs.sp_b.x_inds]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    # if it fails:
    #show_me([safehouse.w[1:60]; safehouse.θ_out[probs.sp_b.x_inds]], [safehouse.w[65:68]; safehouse.w[61:64]]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    @info "Solving gnep.."
    gnep_init = zeros(probs.gnep.top_level.n)
    gnep_init[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; θ_sp_b[probs.sp_a.x_inds]]
    gnep_init[probs.bilevel.inds["λ", 1]] = θ_sp_a[probs.gnep.inds["λ", 1]]
    gnep_init[probs.bilevel.inds["s", 1]] = θ_sp_a[probs.gnep.inds["s", 1]]
    gnep_init[probs.bilevel.inds["λ", 2]] = θ_sp_b[probs.gnep.inds["λ", 1]]
    gnep_init[probs.bilevel.inds["s", 2]] = θ_sp_b[probs.gnep.inds["s", 1]]
    gnep_init = [gnep_init; x0]
    #show_me(gnep_init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    θ_gnep = gnep_init # fall back
    try
        θ_gnep = solve(probs.gnep, gnep_init)
        #show_me(θ_gnep, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
    catch err
        println(err)
        @info "Fell back to gnep init.."
    end

    @info "Solving bilevel.."
    bilevel_init = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
    bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
    bilevel_init[probs.bilevel.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
    bilevel_init[probs.bilevel.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
    bilevel_init[probs.bilevel.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
    bilevel_init[probs.bilevel.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
    bilevel_init[probs.bilevel.inds["w", 0]] = θ_gnep[probs.gnep.inds["w", 0]]

    θ_bilevel = bilevel_init # fall back

    try
        θ_bilevel = solve(probs.bilevel, bilevel_init)
        #show_me(θ_bilevel, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        Z = probs.extract_bilevel(θ_bilevel)
    catch err
        println(err)
        @info "Fell back to gnep init.."
    end

    Z = probs.extract_bilevel(θ_bilevel)
    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    gd = col(Z.Xa, Z.Xb, probs.params.r)
    h = responsibility(Z.Xa, Z.Xb)
    gd_both = [gd - l.(h) gd - l.(-h) gd]
    (; P1, P2, gd_both, h, U1, U2, dummy_init, gnep_init, bilevel_init)
end
