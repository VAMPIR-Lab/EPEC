# 2 players: A, B
# corresponding phantoms: a, b

# 2 Players -------

# 1. Nash equilibrium: 
#   OP-A     OP-B

# 2. A-leader bilevel:
#   OP-A 
#    ^
#   OP-B

# 3. B-leader bilevel:
#   OP-B 
#    ^
#   OP-A

# 3 Players -------

# 4. only b-phantom:
#   OP-Ab     OP-B
#    ^
#   OP-b

# 5. only a-phantom:
#   OP-A    OP-Ba
#            ^
#           OP-a

# 4 Players -------

# 6. both phantoms:
#	OP-Ab	OP-Ba
#    ^       ^
#	OP-b	OP-a 

# 1. OP-A: 
# min	fA(τA,τB)
# τA
# s.t. lA ≤ gA(τA,τB) ≤ uA

# 2. OP-B:
# min	fB(τA,τB)
# τB
# s.t. lB ≤ gB(τA,τB) ≤ uB

# γ∈[0,1]: phantom cost ratio
# Note:
# γ=0   => only real costs
# γ=1   => only phantom costs

# 3. OP-Ab: 
# min 	γ fA(τA,τb) + (1-γ) fA(τA,τB)
# τA,τb
# s.t. lA ≤ gA(τA,τb) ≤ uA 
#      lA ≤ gA(τA,τB) ≤ uA 

# 4. OP-Ba: 
# min 	γ fB(τa,τB) + (1-γ) fB(τA,τB)
# τa,τB
# s.t. lB ≤ gB(τa,τB) ≤ uB
#      lB ≤ gB(τA,τB) ≤ uB

# 5. OP-a: 
# min	fA(τa,τB)
# τa
# s.t.  lA ≤ gA(τa,τB) ≤ uA 

# 6. OP-b: 
# min	fB(τA,τb)
# τb
# s.t.  lB ≤ gB(τA,τb) ≤ uB  

###################
# z decomposition #
###################
# Trajectory of player i:
# τⁱ := [xⁱ₁ ... xⁱₜ | uⁱ₁ ... uⁱₜ]
# for n players:
# z := [τ¹ | ... | τⁿ | params]

# pdim: number of players
# xdim: number of state variables for each player (same for all)
# udim: number of control variables for each player (same for all)
# T: number of time steps

# 2 real players (A, B)
function view_z_AB(z)
    pdim = 2
    xdim = 4
    udim = 2
    T = Int((length(z) - 2 * xdim) / (pdim * (xdim + udim)))

    inds = Dict()
    idx = 0
    for (len, name) in zip(
        [xdim * T, udim * T, xdim * T, udim * T, xdim, xdim],
        ["XA", "UA", "XB", "UB", "x0A", "x0B"])
        inds[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds XA = @view(z[inds["XA"]])
    @inbounds UA = @view(z[inds["UA"]])
    @inbounds XB = @view(z[inds["XB"]])
    @inbounds UB = @view(z[inds["UB"]])
    @inbounds x0A = @view(z[inds["x0A"]])
    @inbounds x0B = @view(z[inds["x0B"]])
    (XA, UA, XB, UB, x0A, x0B, T, inds)
end

# 2 real players (A, B), 1 phantom player (b)
function view_z_ABb(z)
    pdim = 3
    xdim = 4
    udim = 2
    T = Int((length(z) - 2 * xdim) / (pdim * (xdim + udim)))

    inds = Dict()
    idx = 0
    for (len, name) in zip(
        [xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim, xdim],
        ["XA", "UA", "XB", "UB", "Xb", "Ub", "x0A", "x0B"])
        inds[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds XA = @view(z[inds["XA"]])
    @inbounds UA = @view(z[inds["UA"]])
    @inbounds XB = @view(z[inds["XB"]])
    @inbounds UB = @view(z[inds["UB"]])
    @inbounds Xb = @view(z[inds["Xb"]])
    @inbounds Ub = @view(z[inds["Ub"]])
    @inbounds x0A = @view(z[inds["x0A"]])
    @inbounds x0B = @view(z[inds["x0B"]])
    (XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds)
end

# 2 real players (A, B), 2 phantoms (a, b)
function view_z_ABab(z)
    pdim = 4
    xdim = 4
    udim = 2
    T = Int((length(z) - 2 * xdim) / (pdim * (xdim + udim)))

    inds = Dict()
    idx = 0
    for (len, name) in zip(
        [xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim, xdim],
        ["XA", "UA", "XB", "UB", "Xa", "Ua", "Xb", "Ub", "x0A", "x0B"])
        inds[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds XA = @view(z[inds["XA"]])
    @inbounds UA = @view(z[inds["UA"]])
    @inbounds XB = @view(z[inds["XB"]])
    @inbounds UB = @view(z[inds["UB"]])
    @inbounds Xa = @view(z[inds["Xa"]])
    @inbounds Ua = @view(z[inds["Ua"]])
    @inbounds Xb = @view(z[inds["Xb"]])
    @inbounds Ub = @view(z[inds["Ub"]])
    @inbounds x0A = @view(z[inds["x0A"]])
    @inbounds x0B = @view(z[inds["x0B"]])
    (XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds)
end

# 2 real players (A, B), 1 phantom player (a)
#function view_z_ABa(z)
#    pdim = 3
#    xdim = 4
#    udim = 2
#    T = Int((length(z) - 2 * xdim) / (pdim * (xdim + udim)))

#    inds = Dict()
#    idx = 0
#    for (len, name) in zip(
#        [xdim * T, udim * T, xdim * T, udim * T, xdim * T, udim * T, xdim, xdim],
#        ["XA", "UA", "XB", "UB", "Xa", "Ua", "x0A", "x0B"])
#        inds[name] = (idx+1):(idx+len)
#        idx += len
#    end
#    @inbounds XA = @view(z[inds["XA"]])
#    @inbounds UA = @view(z[inds["UA"]])
#    @inbounds XB = @view(z[inds["XB"]])
#    @inbounds UB = @view(z[inds["UB"]])
#    @inbounds Xa = @view(z[inds["Xa"]])
#    @inbounds Ua = @view(z[inds["Ua"]])
#    @inbounds x0A = @view(z[inds["x0A"]])
#    @inbounds x0B = @view(z[inds["x0B"]])
#    (XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds)
#end

################
# base f and g #
################
# X is ego, ego player wants to make forward progress with respect to oppponent, and stay in center of lane
function f_ego(X, U, X_opp; α1, α2, β)
    xdim = 4
    udim = 2
    cost = 0.0
    T = Int(length(X) / xdim)

    for t in 1:T
        @inbounds x = @view(X[xdim*(t-1)+1:xdim*t])
        @inbounds u = @view(U[udim*(t-1)+1:udim*t])
        @inbounds x_opp = @view(X_opp[xdim*(t-1)+1:xdim*t])
        cost += α1 * x[1]^2 + α2 * u' * u + β * (x_opp[4] - x[4])
    end
    cost
end

function pointmass(x, u; Δt, cd)
    Δt2 = 0.5 * Δt * Δt
    a1 = u[1] - cd * x[3]
    a2 = u[2] - cd * x[4]
    [x[1] + Δt * x[3] + Δt2 * a1,
        x[2] + Δt * x[4] + Δt2 * a2,
        x[3] + Δt * a1,
        x[4] + Δt * a2]
end

function dyn(X, U, x0; Δt, cd)
    xdim = 4
    udim = 2
    T = Int(length(X) / xdim)
    x = x0
    mapreduce(vcat, 1:T) do t
        xx = X[(t-1)*xdim+1:t*xdim]
        u = U[(t-1)*udim+1:t*udim]
        diff = xx - pointmass(x, u; Δt, cd)
        x = xx
        diff
    end
end

function col(X1, X2; r)
    xdim = 4
    T = Int(length(X1) / xdim)
    mapreduce(vcat, 1:T) do t
        @inbounds x1 = @view(X1[(t-1)*xdim+1:t*xdim])
        @inbounds x2 = @view(X2[(t-1)*xdim+1:t*xdim])
        delta = x1[1:2] - x2[1:2]
        [delta' * delta - r^2,]
    end
end

# X is ego
function responsibility_ego(X, X_opp)
    xdim = 4
    T = Int(length(X) / xdim)
    mapreduce(vcat, 1:T) do t
        @inbounds x = @view(X[(t-1)*xdim+1:t*xdim])
        @inbounds x_opp = @view(X_opp[(t-1)*xdim+1:t*xdim])
        h = [x_opp[2] - x[2],] # ego h is positive when ego is behind opponent in the second coordinate
    end
end

function sigmoid(x; a, b)
    xx = x * a + b
    1.0 / (1.0 + exp(-xx))
end

# lower bound function -- above zero whenever h ≥ 0, below zero otherwise
function l(h; a=5.0, b=4.5)
    sigmoid(h; a, b) - sigmoid(0; a, b)
end

# X is ego
function u_max_ego(X, X_opp; u_max_nominal, u_max_drafting, box_length, box_width)
    xdim = 4
    T = Int(length(X) / xdim)
    d = mapreduce(vcat, 1:T) do t
        @inbounds x = @view(X[(t-1)*xdim+1:t*xdim])
        @inbounds x_opp = @view(X_opp[(t-1)*xdim+1:t*xdim])
        [x[1] - x_opp[1] x[2] - x_opp[2]]
    end
    @assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    u_max_1 = du * sigmoid.(d[:, 2] .+ box_length; a=10.0, b=0) .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:, 2]; a=10.0, b=0) .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:, 1] .+ box_width / 2; a=10.0, b=0) .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:, 1] .+ box_width / 2; a=10.0, b=0) .+ u_max_nominal
    (u_max_1, u_max_2, u_max_3, u_max_4)
end

# X is ego, constraints are: dynamics, collision, max accelerations, max backwards velocity, max lateral position
function g_ego(X, U, x0, X_opp; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    xdim = 4
    udim = 2

    g_dyn = dyn(X, U, x0; Δt, cd)
    g_col = col(X, X_opp; r)
    h_col = responsibility_ego(X, X_opp)
    u_max_1, u_max_2, u_max_3, u_max_4 = u_max_ego(X, X_opp; u_max_nominal, u_max_drafting, box_length, box_width)
    long_accel = @view(U[udim:udim:end])
    lat_accel = @view(U[1:udim:end])
    lat_pos = @view(X[1:xdim:end])
    long_vel = @view(X[xdim:xdim:end])

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

# 2 Players -------
# 1. OP-A
# 2. OP-B

########
# OP-A #
########
# ego: A
# opp: B

# fA(τA, τB)
function fA(z; α1, α2, β)
    XA, UA, XB, UB, x0A, x0B, T, inds = view_z_AB(z)
    f_ego(XA, UA, XB; α1, α2, β)
end
# gA(τA,τB)
function gA(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, x0A, x0B, T, inds = view_z_AB(z)
    g_ego(XA, UA, x0A, XB; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

########
# OP-B #
########
# ego: B
# opp: A

# fB(τA, τB)
function fB(z; α1, α2, β)
    XA, UA, XB, UB, x0A, x0B, T, inds = view_z_AB(z)
    f_ego(XB, UB, XA; α1, α2, β)
end
# gB(τA,τB)
function gB(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, x0A, x0B, T, inds = view_z_AB(z)
    g_ego(XB, UB, x0B, XA; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

# 3 Players -------
# 1. OP-Ab-only-b
# 2. OP-Ba-only-a
# 3. OP-A-only-a
# 4. OP-B-only-b
# 5. OP-a-only-a
# 6. OP-b-only-b

#################
# OP-Ab only b  #
#################
# ego: A
# opp: b, B

# fAb = γ fA(τA,τb) + (1-γ) fA(τA,τB)
function fAb_only_b(z; α1, α2, β, γ)
    XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb(z)
    γ * f_ego(XA, UA, Xb; α1, α2, β) + (1.0 - γ) * f_ego(XA, UA, XB; α1, α2, β)
end
# gAb = 
#  gA(τA, τb)
#  gA(τA, τB)
function gAb_only_b(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb(z)
    [g_ego(XA, UA, x0A, Xb; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g_ego(XA, UA, x0A, XB; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end

###############
# OP-B only b #
###############
# ego: B
# opp: A

# fB(τA, τB)
function fB_only_b(z; α1, α2, β)
    XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb(z)
    f_ego(XB, UB, XA; α1, α2, β)
end
# gB(τA,τB)
function gB_only_b(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb(z)
    g_ego(XB, UB, x0B, XA; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

###############
# OP-b only b #
###############
# ego: b
# opp: A

# fb = fB(τA, τb)
function fb_only_b(z; α1, α2, β)
    XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb(z)
    f_ego(Xb, Ub, XA; α1, α2, β)
end
# gb = gB(τA,τb)
function gb_only_b(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb(z)
    g_ego(Xb, Ub, x0B, XA; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

################
# OP-Ba only a #
################
# ego: B
# opp: a, A

# fBa = γ fB(τa,τB) + (1-γ) fB(τA,τB) 
#function fBa_only_a(z; α1, α2, β, γ)
#    XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds = view_z_ABa(z)
#    γ * f_ego(XB, UB, Xa; α1, α2, β) + (1.0 - γ) * f_ego(XB, UB, XA; α1, α2, β)
#end
# gBa = 
#  gB(τa, τB)
#  gA(τA, τB)
#function gBa_only_a(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
#    XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds = view_z_ABa(z)
#    [g_ego(XB, UB, x0B, Xa; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
#        g_ego(XB, UB, x0B, XA; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
#end

###############
# OP-A only a #
###############
# ego: A
# opp: B

# fA(τA, τB)
#function fA_only_a(z; α1, α2, β)
#    XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds = view_z_ABa(z)
#    f_ego(XA, UA, XB; α1, α2, β)
#end
# gA(τA,τB)
#function gA_only_a(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
#    XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds = view_z_ABa(z)
#    g_ego(XA, UA, x0A, XB; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
#end

###############
# OP-a only a #
###############
# ego: a
# opp: B

# fa = fA(τa, τB)
#function fa_only_a(z; α1, α2, β)
#    XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds = view_z_ABa(z)
#    f_ego(Xa, Ua, XB; α1, α2, β)
#end
#-- ga = gA(τa,τB)
#function ga_only_a(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
#    XA, UA, XB, UB, Xa, Ua, x0A, x0B, T, inds = view_z_ABa(z)
#    g_ego(Xa, Ua, x0A, XB; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
#end

# 4 Players -------
# 1. OP-Ab
# 2. OP-Ba
# 3. OP-b
# 4. OP-a

#########
# OP-Ab #
#########
# ego: A
# opp: b, B

# fAb = γ fA(τA,τb) + (1-γ) fA(τA,τB)
function fAb(z; α1, α2, β, γ)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    γ * f_ego(XA, UA, Xb; α1, α2, β) + (1.0 - γ) * f_ego(XA, UA, XB; α1, α2, β)
end
# gAb = 
#  gA(τA, τb)
#  gA(τA, τB)
function gAb(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    [g_ego(XA, UA, x0A, Xb; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g_ego(XA, UA, x0A, XB; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end

#########
# OP-Ba #
#########
# ego: B
# opp: a, A

# fBa = γ fB(τa,τB) + (1-γ) fB(τA,τB) 
function fBa(z; α1, α2, β, γ)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    γ * f_ego(XB, UB, Xa; α1, α2, β) + (1.0 - γ) * f_ego(XB, UB, XA; α1, α2, β)
end
# gBa = 
#  gB(τa, τB)
#  gA(τA, τB)
function gBa(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    [g_ego(XB, UB, x0B, Xa; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
        g_ego(XB, UB, x0B, XA; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)]
end

########
# OP-a #
########
# ego: a
# opp: B

# fa = fA(τa, τB)
function fa(z; α1, α2, β)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    f_ego(Xa, Ua, XB; α1, α2, β)
end
#-- ga = gA(τa,τB)
function ga(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    g_ego(Xa, Ua, x0A, XB; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

########
# OP-b #
########
# ego: b
# opp: A

# fb = fB(τA, τb)
function fb(z; α1, α2, β)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    f_ego(Xb, Ub, XA; α1, α2, β)
end
# gb = gB(τA,τb)
function gb(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab(z)
    g_ego(Xb, Ub, x0B, XA; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
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

    lb = [fill(0.0, 4 * T); fill(0.0, T); fill(-u_max_nominal, T); fill(-Inf, 4 * T); fill(-u_max_braking, T); fill(min_long_vel, T); fill(-lat_max, T)]
    ub = [fill(0.0, 4 * T); fill(Inf, T); fill(+u_max_nominal, T); fill(0.0, 4 * T); fill(Inf, T); fill(Inf, T); fill(+lat_max, T)]
    lb_all = [lb; lb]
    ub_all = [ub; ub]

    # 2 Players -------
    fA_pin = (z -> fA(z; α1, α2, β))
    fB_pin = (z -> fB(z; α1, α2, β))

    gA_pin = (z -> gA(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    gB_pin = (z -> gB(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))

    OP_A = OptimizationProblem(2 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fA_pin, gA_pin, lb, ub)
    OP_B = OptimizationProblem(2 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fB_pin, gB_pin, lb, ub)

    # 3 Players -------  
    fAb_only_b_pin = (z -> fAb_only_b(z; α1, α2, β, γ))
    fB_only_b_pin = (z -> fB_only_b(z; α1, α2, β))
    fb_only_b_pin = (z -> fb_only_b(z; α1, α2, β))
    #fBa_only_a_pin = (z -> fBa_only_a(z; α1, α2, β, γ))
    #fA_only_a_pin = (z -> fA_only_a(z; α1, α2, β))
    #fa_only_a_pin = (z -> fa_only_a(z; α1, α2, β))

    gAb_only_b_pin = (z -> gAb_only_b(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    gB_only_b_pin = (z -> gB_only_b(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    gb_only_b_pin = (z -> gb_only_b(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    #gBa_only_a_pin = (z -> gBa_only_a(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    #gA_only_a_pin = (z -> gA_only_a(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    #ga_only_a_pin = (z -> ga_only_a(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))


    OP_Ab_only_b = OptimizationProblem(
        3 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fAb_only_b_pin, gAb_only_b_pin, lb_all, ub_all)
    OP_B_only_b = OptimizationProblem(
        3 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fB_only_b_pin, gB_only_b_pin, lb, ub)
    OP_b_only_b = OptimizationProblem(
        3 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fb_only_b_pin, gb_only_b_pin, lb, ub)
    #OP_Ba_only_a = OptimizationProblem(
    #    3 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fBa_only_a_pin, gBa_only_a_pin, lb_all, ub_all)
    #OP_A_only_a = OptimizationProblem(
    #    3 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fA_only_a_pin, gA_only_a_pin, lb, ub)
    #OP_a_only_a = OptimizationProblem(
    #    3 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fa_only_a_pin, ga_only_a_pin, lb, ub)

    # 4 Players -------
    fAb_pin = (z -> fAb(z; α1, α2, β, γ))
    fBa_pin = (z -> fBa(z; α1, α2, β, γ))
    fa_pin = (z -> fa(z; α1, α2, β))
    fb_pin = (z -> fb(z; α1, α2, β))

    gAb_pin = (z -> gAb(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    gBa_pin = (z -> gBa(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    ga_pin = (z -> ga(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))
    gb_pin = (z -> gb(z; Δt, r, cd, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer))

    OP_Ab = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fAb_pin, gAb_pin, lb_all, ub_all)
    OP_Ba = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fBa_pin, gBa_pin, lb_all, ub_all)
    OP_a = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fa_pin, ga_pin, lb, ub)
    OP_b = OptimizationProblem(4 * (xdim + udim) * T + 2 * xdim, 1:(xdim+udim)*T, fb_pin, gb_pin, lb, ub)

    # 2 Players -------
    # 1. Nash equilibrium: 
    nash = [OP_A OP_B]
    # 2. A-leader bilevel:
    A_leader = [OP_A; OP_B]
    # 3. B-leader bilevel:
    #B_leader = [OP_B; OP_A]
    # 3 Players -------
    # 4. only b-phantom:
    only_b_phantom = EPEC.create_epec((2, 1), OP_Ab_only_b, OP_B_only_b, OP_b_only_b)
    # 5. only a-phantom:
    #only_a_phantom = EPEC.create_epec((2, 1), OP_A_only_a, OP_Ba_only_a, OP_a_only_a)
    # 4 Players -------
    # 6. both phantoms:
    phantom = EPEC.create_epec((2, 2), OP_Ab, OP_Ba, OP_a, OP_b)

    OPs = (; A=OP_A, B=OP_B, Ab_only_b=OP_Ab_only_b, B_only_b=OP_B_only_b, b_only_b=OP_b_only_b, Ab=OP_Ab, Ba=OP_Ba, a=OP_a, b=OP_b)

    function extract_AB(θ, x_inds, x0)
        z = θ[x_inds]
        XA, UA, XB, UB, x0A, x0B, T, inds = view_z_AB([z; x0])
        (; XA, UA, XB, UB)
    end

    function extract_ABb(θ, x_inds, x0)
        z = θ[x_inds]
        XA, UA, XB, UB, Xb, Ub, x0A, x0B, T, inds = view_z_ABb([z; x0])
        (; XA, UA, XB, UB, Xb, Ub)
    end

    function extract_ABab(θ, x_inds, x0)
        z = θ[x_inds]
        XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABab([z; x0])
        (; XA, UA, XB, UB, Xa, Ua, Xb, Ub)
    end
    
    #function extract_ABa(θ, x_inds, x0)
    #    z = θ[x_inds]
    #    XA, UA, XB, UB, Xa, Ua, Xb, Ub, x0A, x0B, T, inds = view_z_ABa([z; x0])
    #    (; XA, UA, XB, UB, Xa, Ua, Xb, Ub)
    #end

    params = (; T, Δt, r, cd, lat_max, u_max_nominal, u_max_drafting, u_max_braking, α1, α2, β, box_length, box_width, min_long_vel, col_buffer)

    (; nash, A_leader, only_b_phantom, phantom, params, extract_AB, extract_ABb, extract_ABab, OPs)
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

# initializations:
# nash:
# dummy->gnep
# bilevel:
# gnep->A_leader
# gnep->B_leader
# phantoms:
# gnep->only_a_phantom
# gnep->only_b_phantom
# only_a_phantom + only_b_phantom->both_phantoms
function get_dummy_sol(x0; T, Δt, cd)
    XA = []
    UA = []
    XB = []
    UB = []
    x0A = x0[1:4]
    x0B = x0[5:8]
    xA = x0A
    xA = x0B

    for t in 1:T
        uA = [0; 0]
        uB = [0; 0]
        xA = pointmass(xA, uA; Δt, cd)
        xb = pointmass(xA, uB; Δt, cd)
        append!(UA, uA)
        append!(UB, uB)
        append!(XA, xA)
        append!(XB, xb)
    end
    (XA, UA, XB, UB)
end

function solve_nash_seq(probs, x0)
    XA, UA, XB, UB = get_dummy_sol(x0; probs.params.T, probs.params.Δt, probs.params.cd)

    init = zeros(probs.nash.top_level.n)
    init[probs.nash.x_inds] = [XA; UA; XB; UB]
    init = [init; x0]
    success, θ = attempt_solve(probs.nash, init)

    if success
        τ = probs.extract_AB(θ, probs.nash.x_inds, x0)
    else
        τ = (; XA, UA, XB, UB)
    end

    (τ, success)
end

function solve_A_leader_seq(probs, x0; dummy_init_first=false)
    XA, UA, XB, UB = get_dummy_sol(x0; probs.params.T, probs.params.Δt, probs.params.cd)

    init = zeros(probs.A_leader.top_level.n)
    init[probs.A_leader.x_inds] = [XA; UA; XB; UB]
    init = [init; x0]

    if dummy_init_first
        success, θ = attempt_solve(probs.A_leader, init)
    else
        success = false
    end

    # initialize with nash    
    if !success
        nash_init = zeros(probs.nash.top_level.n)
        nash_init[probs.nash.x_inds] = [XA; UA; XB; UB]
        nash_init = [nash_init; x0]
        nash_success, θ_nash = attempt_solve(probs.nash, nash_init)

        if nash_success
            init[probs.A_leader.x_inds] = θ_nash[probs.nash.x_inds]
            init[probs.A_leader.inds["λ", 1]] = θ_nash[probs.nash.inds["λ", 1]]
            init[probs.A_leader.inds["s", 1]] = θ_nash[probs.nash.inds["s", 1]]
            init[probs.A_leader.inds["λ", 2]] = θ_nash[probs.nash.inds["λ", 2]]
            init[probs.A_leader.inds["s", 2]] = θ_nash[probs.nash.inds["s", 2]]

            success, θ = attempt_solve(probs.A_leader, init)
        end
    end

    if success
        τ = probs.extract_AB(θ, probs.A_leader.x_inds, x0)
    elseif nash_success
        τ = probs.extract_AB(θ, probs.nash.x_inds, x0)
    else
        τ = (; XA, UA, XB, UB)
    end

    (τ, success)
end

function solve_only_b_phantom_seq(probs, x0; dummy_init_first=false)
end

function solve_phantom_seq(probs, x0)
    @info "gnep"
    gnep_init = zeros(probs.gnep.top_level.n)
    gnep_init[probs.gnep.x_inds] = [XA; UA; XB; UB]
    gnep_init = [gnep_init; x0]
    gnep_success, θ_gnep = attempt_solve(probs.gnep, gnep_init)

    @info "phantom pain"
    phantom_init = zeros(probs.phantom.top_level.n)
    phantom_init[probs.phantom.x_inds] = [τ.XA; τ.UA; τ.XB; τ.UB; τ.Xa; τ.Ua; τ.Xb; τ.Ub]
    phantom_init = [phantom_init; x0]
end

# check initial condition feasibility
function check_feasibility(x0; r, lat_max, min_long_vel)
    x0A = x0[1:4]
    x0B = x0[5:8]
    is_x0_infeasible = false
    reason = ""

    if col(x0A, x0B; r)[1] <= 0 - 1e-4
        reason = "Collision"
        is_x0_infeasible = true
    elseif x0A[1] < -lat_max - 1e-4 || x0A[1] > lat_max + 1e-4 || x0B[1] < -lat_max - 1e-4 || x0B[1] > lat_max + 1e-4
        reason = "Out of lanes"
        is_x0_infeasible = true
    elseif x0A[4] < min_long_vel - 1e-4 || x0B[4] < min_long_vel - 1e-4
        reason = "Invalid velocity"
        is_x0_infeasible = true
    end
    (is_x0_infeasible, reason)
end

# clamp controls and get next x0
function apply_control(τ)
    xdim = 4
    udim = 2
    xA = τ.XA[1:xdim]
    xB = τ.XB[1:xdim]
    uA = τ.UA[1:udim]
    uB = τ.UB[1:udim]
    uA_maxes = u_max_ego(xA,
        xB;
        probs.params.u_max_nominal,
        probs.params.u_max_drafting,
        probs.params.box_length,
        probs.params.box_width)

    uB_maxes = u_max_ego(xB,
        xA;
        probs.params.u_max_nominal,
        probs.params.u_max_drafting,
        probs.params.box_length,
        probs.params.box_width)

    uA[1] = minimum([maximum([uA[1], -probs.params.u_max_nominal]), probs.params.u_max_nominal])
    uB[1] = minimum([maximum([uB[1], -probs.params.u_max_nominal]), probs.params.u_max_nominal])
    uA[2] = minimum([maximum([uA[2], -probs.params.u_max_braking]), uA_maxes[1][1], uA_maxes[2][1], uA_maxes[3][1], uA_maxes[4][1]])
    uB[2] = minimum([maximum([uB[2], -probs.params.u_max_braking]), uB_maxes[1][1], uB_maxes[2][1], uB_maxes[3][1], uB_maxes[4][1]])
    x0A = pointmass(xA, uA; probs.params.Δt, probs.params.cd)
    x0B = pointmass(xB, uB; probs.params.Δt, probs.params.cd)

    [x0A; x0B]
end

# Solve mode (TBD):
#						P1:						
#				NE   P1-leader  P1-follower a-phantom b-phantom     
# P2:		 NE 1   
#	  P2-Leader 2   3    
#   P2-Follower 4   5           6		   
#
function solve_simulation(probs, T; x0)
    results = Dict()

    for t = 1:T
        print("Sim timestep $t: ")
        is_x0_infeasible, reason = check_feasibility(x0; probs.params.r, probs.params.lat_max, probs.params.min_long_vel)

        if is_x0_infeasible
            print(reason)
            print("\n")
            break
        end
        τ_nash, success_nash = solve_nash_seq(probs, x0)
        if success_nash
            print("nash success")
        end

        #τ_A_leader, success_A_leader = solve_A_leader_seq(probs, x0)
        #if success_A_leader
        #    print("A leader success")
        #end
        
        print("\n")

        τ = τ_nash;

        results[t] = (; x0, τ.XA, τ.XB, τ.UA, τ.UB)

        x0 = apply_control(τ)
    end
    results
end
