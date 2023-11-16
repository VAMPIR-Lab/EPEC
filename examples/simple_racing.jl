# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# xⁱₖ := [p1 p2 v1 v2]
# xₖ = dyn(xₖ₋₁, uₖ; Δt) (pointmass dynamics)

# P1 wants to make forward progress and stay in center of lane.
function f1(Z; α1 = 1.0, α2 = 0.0)
    T = Int((length(Z)-8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])
    
    cost = 0.0
    for t in 1:T
        @inbounds xat = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xbt = @view(Xb[(t-1)*4+1:t*4])
        @inbounds ut = @view(Ua[(t-1)*2+1:t*2]) 
        cost += xbt[2]-xat[2] + α1*xat[1]^2 + α2 * ut'*ut
    end
    cost
end

# P2 wants to make forward progress and stay in center of lane.
function f2(Z; α1 = 1.0, α2 = 0.0)
    T = Int((length(Z)-8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])
    
    cost = 0.0
    for t in 1:T
        @inbounds xat = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xbt = @view(Xb[(t-1)*4+1:t*4])
        @inbounds ut = @view(Ub[(t-1)*2+1:t*2]) 
        cost += xat[2]-xbt[2] + α1*xbt[1]^2 + α2 * ut'*ut
    end
    cost
end

function pointmass(x, u, Δt, cd)
    Δt2 = 0.5*Δt*Δt
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
        delta = xa[1:2]-xb[1:2] 
        [delta'*delta - r^2,]
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
        [xa[1]-xb[1] xa[2]-xb[2]]
    end
    @assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    u_max_1 = du * sigmoid.(d[:,2] .+ box_length, 10.0, 0)  .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:,2], 10.0, 0)  .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:,1] .+ box_width/2, 10.0, 0)  .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:,1] .+ box_width/2, 10.0, 0)  .+ u_max_nominal
    (u_max_1, u_max_2, u_max_3, u_max_4)
end
function accel_bounds_2(Xa, Xb, u_max_nominal, u_max_drafting, box_length, box_width)
    T = Int(length(Xa) / 4)
    d = mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*4+1:t*4])
        @inbounds xb = @view(Xb[(t-1)*4+1:t*4])
        [xb[1]-xa[1] xb[2]-xa[2]]
    end
    @assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    u_max_1 = du * sigmoid.(d[:,2] .+ box_length, 10.0, 0)  .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:,2], 10.0, 0)  .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:,1] .+ box_width/2, 10.0, 0)  .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:,1] .+ box_width/2, 10.0, 0)  .+ u_max_nominal
    (u_max_1, u_max_2, u_max_3, u_max_4)
end

function sigmoid(x, a, b)
    xx = x*a+b
    1.0 / (1.0 + exp(-xx))
end

# lower bound function -- above zero whenever h ≥ 0, below zero otherwise
function l(h; a=5.0, b=4.5)
    sigmoid(h, a, b) - sigmoid(0, a, b)
end

function g1(Z; 
        Δt = 0.1, 
        r = 1.0, 
        cd = 1.0, 
        u_max_nominal=5.0, 
        u_max_drafting=3.0,
        box_length=3.0, 
        box_width=1.0)
    T = Int((length(Z)-8) / 12) # 2*(state_dim + control_dim) = 12
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

    [g_dyn; 
     g_col - l.(h_col); 
     lat_accel; 
     long_accel-u_max_1; 
     long_accel-u_max_2; 
     long_accel-u_max_3; 
     long_accel-u_max_4; 
     lat_pos]
end

function g2(Z; 
        Δt = 0.1, 
        r = 1.0, 
        cd = 1.0, 
        u_max_nominal=3.0, 
        u_max_drafting=5.0,
        box_length=3.0, 
        box_width=1.0)
    T = Int((length(Z)-8) / 12) # 2*(state_dim + control_dim) = 12
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

    [g_dyn; 
     g_col - l.(h_col); 
     lat_accel; 
     long_accel-u_max_1; 
     long_accel-u_max_2; 
     long_accel-u_max_3; 
     long_accel-u_max_4; 
     lat_pos]
end

function setup(; T=10, 
                 Δt = 0.1, 
                 r=1.0, 
                 α1 = 0.01,
                 α2 = 0.001,
                 cd = 0.5,
                 u_max_nominal = 3.0, 
                 u_max_drafting = 5.0,
                 box_length=3.0,
                 box_width=1.0,
                 lat_max = 2.0)

    lb = [fill(0.0, 4*T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4*T); fill(-lat_max, T)]
    ub = [fill(0.0, 4*T); fill(Inf, T); fill(+u_max_nominal, T); fill( 0.0, 4*T); fill(+lat_max, T)]

    f1_pinned = (z -> f1(z; α1, α2))
    f2_pinned = (z -> f2(z; α1, α2))
    g1_pinned = (z -> g1(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width))
    g2_pinned = (z -> g2(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width))

    OP1 = OptimizationProblem(12*T+8, 1:6*T, f1_pinned, g1, lb, ub)
    OP2 = OptimizationProblem(12*T+8, 1:6*T, f2_pinned, g2, lb, ub)

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
    problems = (; gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2, params=(; Δt, r, cd, lat_max))
end

function solve_seq(probs, x0)
    init = zeros(probs.gnep.top_level.n)
    X = init[probs.gnep.x_inds]
    T = Int(length(X) / 12)
    Δt = probs.params.Δt
    cd = probs.params.cd
    Xa = []
    Ua = []
    Xb = []
    Ub = []
    xa = x0[1:4]
    xb = x0[5:8]
    for t in 1:T
        ua = cd*xa[3:4]
        ub = cd*xb[3:4]
        xa = pointmass(xa, ua, Δt, cd)
        xb = pointmass(xb, ub, Δt, cd)
        append!(Ua, ua)
        append!(Ub, ub)
        append!(Xa, xa)
        append!(Xb, xb)
    end
    init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
    init = [init; x0]

    θg = solve(probs.gnep, init)
    θb = zeros(probs.bilevel.top_level.n + probs.bilevel.top_level.n_param)
    θb[probs.bilevel.x_inds] = θg[probs.gnep.x_inds]
    θb[probs.bilevel.inds["λ", 1]] = θg[probs.gnep.inds["λ", 1]]
    θb[probs.bilevel.inds["s", 1]] = θg[probs.gnep.inds["s", 1]]
    θb[probs.bilevel.inds["λ", 2]] = θg[probs.gnep.inds["λ", 2]]
    θb[probs.bilevel.inds["s", 2]] = θg[probs.gnep.inds["s", 2]]
    θb[probs.bilevel.inds["w", 0]] = θg[probs.gnep.inds["w", 0]]
    θ = solve(probs.bilevel, θb)
    Z = probs.extract_bilevel(θ)
    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    #P1 = [x0[1:4]'; P1]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]
    #P2 = [x0[5:8]'; P2]
   
    #u_max_11, u_max_12, u_max_13, u_max_14 = accel_bounds_1(Z.Xa, 
    #                                                    Z.Xb, 
    #                                                    3.0, 
    #                                                    5.0, 
    #                                                    3.0, 
    #                                                    1.0) 
    #u_max_21, u_max_22, u_max_23, u_max_24 = accel_bounds_2(Z.Xa, 
    #                                                    Z.Xb, 
    #                                                    3.0, 
    #                                                    5.0, 
    #                                                    3.0, 
    #                                                    1.0) 
    #g2val = g2([Z.Xa; Z.Ua; Z.Xb; Z.Ub; x0]; cd=probs.params.cd)
    gd = col(Z.Xa, Z.Xb, probs.params.r)
    h = responsibility(Z.Xa, Z.Xb)
    gd_both = [gd-l.(h) gd-l.(-h) gd]
    (; P1, P2, gd_both, h, U1, U2)#, u_max_11, u_max_12, u_max_13, u_max_14, u_max_21, u_max_22, u_max_23, u_max_24)
end

function solve_simulation(probs, x0, T)
    results = Dict()
    for t = 1:T
        @info "Simulation step $t"
        r = solve_seq(probs, x0)
        x0a = r.P1[1,:]
        x0b = r.P2[1,:]
        results[t] = (; x0, r.P1, r.P2, r.U1, r.U2, r.gd_both, r.h)
        x0 = [x0a; x0b]
    end
    results
end

function visualize(probs, sim_results)
    T = length(sim_results)
    f = Figure(resolution = (1000, 1000))
    ax = Axis(f[1,1], aspect = DataAspect())
    rad = sqrt(probs.params.r) / 2
    lat = probs.params.lat_max + rad
    lines!(ax, [-lat, -lat], [-10.0, 30.0], color=:black)
    lines!(ax, [+lat, +lat], [-10.0, 30.0], color=:black)

    xa1 = Observable(sim_results[1].x0[1])
    xa2 = Observable(sim_results[1].x0[2])
    xb1 = Observable(sim_results[1].x0[5])
    xb2 = Observable(sim_results[1].x0[6])

    circ_x = [rad*cos(t) for t in 0:0.1:2π]
    circ_y = [rad*sin(t) for t in 0:0.1:2π]
    lines!(ax, @lift(circ_x .+ $xa1), @lift(circ_y .+ $xa2), color=:blue)
    lines!(ax, @lift(circ_x .+ $xb1), @lift(circ_y .+ $xb2), color=:red)
    
    display(f)
    for t in 2:T
        xa1[] = sim_results[t].x0[1]
        xa2[] = sim_results[t].x0[2]
        xb1[] = sim_results[t].x0[5]
        xb2[] = sim_results[t].x0[6]
        sleep(0.1)
    end
end

