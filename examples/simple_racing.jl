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

function pointmass(x, u, Δt)
    Δt2 = 0.5*Δt*Δt
    [x[1] + Δt * x[3] + Δt2 * u[1],
     x[2] + Δt * x[4] + Δt2 * u[2],
     x[3] + Δt * u[1],
     x[4] + Δt * u[2]]
end

function dyn(X, U, x0, Δt)
    T = Int(length(X) / 4)
    x = x0
    mapreduce(vcat, 1:T) do t
        xx = X[(t-1)*4+1:t*4]
        u = U[(t-1)*2+1:t*2]
        diff = xx - pointmass(x, u, Δt)
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

function sigmoid(x, a, b)
    xx = x*a+b
    1.0 / (1.0 + exp(-xx))
end

# lower bound function -- above zero whenever h ≥ 0, below zero otherwise
function l(h; a=5.0, b=4.5)
    sigmoid(h, a, b) - sigmoid(0, a, b)
end

function g1(Z; Δt = 0.1, r = 1.0)
    T = Int((length(Z)-8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])
    @inbounds x0a = @view(Z[12*T+1:12*T+4])
    @inbounds x0b = @view(Z[12*T+5:12*T+8])

    g_dyn = dyn(Xa, Ua, x0a, Δt)
    g_col = col(Xa, Xb, r)
    h_col = responsibility(Xa, Xb)   
    h_col = -ones(length(g_col))
    [g_dyn; g_col - l.(h_col); Ua; Xa]
end

function g2(Z; Δt = 0.1, r = 1.0)
    T = Int((length(Z)-8) / 12) # 2*(state_dim + control_dim) = 12
    @inbounds Xa = @view(Z[1:4*T])
    @inbounds Ua = @view(Z[4*T+1:6*T])
    @inbounds Xb = @view(Z[6*T+1:10*T])
    @inbounds Ub = @view(Z[10*T+1:12*T])
    @inbounds x0a = @view(Z[12*T+1:12*T+4])
    @inbounds x0b = @view(Z[12*T+5:12*T+8])

    g_dyn = dyn(Xb, Ub, x0b, Δt)
    g_col = col(Xa, Xb, r)
    h_col = -responsibility(Xa, Xb)   
    h_col = ones(length(g_col))

    [g_dyn; g_col - l.(h_col); Ub; Xb]
end

function setup(; T=10, 
                 Δt = 0.1, 
                 r=1.0, 
                 α1 = 0.01,
                 α2 = 0.01,
                 u_max_1 = 5.0, 
                 u_max_2 = 10.0, 
                 v_max_1 = 2.0, 
                 v_max_2 = 3.0, 
                 lat_max = 0.75)
    lb1 = [fill(0.0, 4*T); fill(0.0, T); fill(-u_max_1, 2*T); repeat([-lat_max, -Inf, -v_max_1, -v_max_1], T)]
    ub1 = [fill(0.0, 4*T); fill(Inf, T); fill(+u_max_1, 2*T); repeat([+lat_max, +Inf, +v_max_1, +v_max_1], T)]
    lb2 = [fill(0.0, 4*T); fill(0.0, T); fill(-u_max_2, 2*T); repeat([-lat_max, -Inf, -v_max_2, -v_max_2], T)]
    ub2 = [fill(0.0, 4*T); fill(Inf, T); fill(+u_max_2, 2*T); repeat([+lat_max, +Inf, +v_max_2, +v_max_2], T)]

    f1_pinned = (z -> f1(z; α1, α2))
    f2_pinned = (z -> f2(z; α1, α2))

    OP1 = OptimizationProblem(12*T+8, 1:6*T, f1_pinned, g1, lb1, ub1)
    OP2 = OptimizationProblem(12*T+8, 1:6*T, f2_pinned, g2, lb2, ub2)

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
    problems = (; gnep, bilevel, extract_gnep, extract_bilevel, OP1, OP2)
end

function solve_seq(probs, x0)
    init = zeros(probs.gnep.top_level.n)
    X = init[probs.gnep.x_inds]
    T = Int(length(X) / 12)
    Xa = []
    Ua = []
    Xb = []
    Ub = []
    xa = x0[1:4]
    xb = x0[5:8]
    for t in 1:T
        ua = zeros(2)
        ub = zeros(2)
        xa = pointmass(xa, ua, 0.1)
        xb = pointmass(xb, ub, 0.1)
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
    #P1 = [x0[1:4]'; P1]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    #P2 = [x0[5:8]'; P2]
    
    gd = col(Z.Xa, Z.Xb, 1.0)
    h = responsibility(Z.Xa, Z.Xb)
    gd_both = [gd-l.(h) gd-l.(-h) gd]
    (; P1, P2, gd_both, h)
end

function solve_simulation(probs, x0, T)
    results = Dict()
    for t = 1:T
        @info "Simulation step $t"
        r = solve_seq(probs, x0)
        x0a = r.P1[1,:]
        x0b = r.P2[1,:]
        results[t] = (; x0, r.P1, r.P2, r.gd_both, r.h)
        x0 = [x0a; x0b]
    end
    results
end




