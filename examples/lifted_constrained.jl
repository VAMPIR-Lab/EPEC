function dist2(a,b)
    (a-b)'*(a-b)
end

function f1(Z)
    @inbounds X1a = @view(Z[1:2])
    @inbounds X1b = @view(Z[3:4])
    @inbounds p1  = @view(Z[5:6])
    @inbounds X2a = @view(Z[7:8])
    @inbounds X2b = @view(Z[9:10])
    @inbounds p2  = @view(Z[11:12])
    @inbounds X3a = @view(Z[13:14])
    @inbounds X3b = @view(Z[15:16])
    @inbounds p3  = @view(Z[17:18])

    A = -[dist2(X1a,X2a) dist2(X1a,X2b);
          dist2(X1b,X2a) dist2(X1b,X2b)]
    p1'*A*p2
end

function f2(Z)
    -f1(Z)
end

function f3(Z)
    @inbounds X1a = @view(Z[1:2])
    @inbounds X1b = @view(Z[3:4])
    @inbounds p1  = @view(Z[5:6])
    @inbounds X2a = @view(Z[7:8])
    @inbounds X2b = @view(Z[9:10])
    @inbounds p2  = @view(Z[11:12])
    @inbounds X3a = @view(Z[13:14])
    @inbounds X3b = @view(Z[15:16])
    @inbounds p3  = @view(Z[17:18])

    A = [dist2(X3a,X2a) dist2(X3a,X2b);
         dist2(X3b,X2a) dist2(X3b,X2b)]
    p3'*A*p2
end

function sigmoid(x)
    1.0 / (1.0 + exp(-x))
end

function g1(Z)
    @inbounds X1a = @view(Z[1:2])
    @inbounds X1b = @view(Z[3:4])
    @inbounds p1  = @view(Z[5:6])
    @inbounds X2a = @view(Z[7:8])
    @inbounds X2b = @view(Z[9:10])
    @inbounds p2  = @view(Z[11:12])
    @inbounds X3a = @view(Z[13:14])
    @inbounds X3b = @view(Z[15:16])
    @inbounds p3  = @view(Z[17:18])

    [X1a; X1b; p1; sum(p1)]
end

function g2(Z, d)
    @inbounds X1a = @view(Z[1:2])
    @inbounds X1b = @view(Z[3:4])
    @inbounds p1  = @view(Z[5:6])
    @inbounds X2a = @view(Z[7:8])
    @inbounds X2b = @view(Z[9:10])
    @inbounds p2  = @view(Z[11:12])
    @inbounds X3a = @view(Z[13:14])
    @inbounds X3b = @view(Z[15:16])
    @inbounds p3  = @view(Z[17:18])

    A = d .- [dist2(X3a,X2a) dist2(X3a,X2b);
              dist2(X3b,X2a) dist2(X3b,X2b)]
    S = sigmoid.(A)
    s = p3'*S*p2

    [X2a; X2b; p2; sum(p2); s]
end

function g3(Z)
    @inbounds X1a = @view(Z[1:2])
    @inbounds X1b = @view(Z[3:4])
    @inbounds p1  = @view(Z[5:6])
    @inbounds X2a = @view(Z[7:8])
    @inbounds X2b = @view(Z[9:10])
    @inbounds p2  = @view(Z[11:12])
    @inbounds X3a = @view(Z[13:14])
    @inbounds X3b = @view(Z[15:16])
    @inbounds p3  = @view(Z[17:18])

    [X3a; X3b; p3; sum(p3)]
end

function setup(; r=1.0, 
                 x_max = 2.0,
                 pv_max = 0.25)
    
    lb = [fill(-x_max, 4); fill(0.0, 2); 1]
    ub = [fill(+x_max, 4); fill(Inf, 2); 1]

    g2_pinned = (z -> g2(z, r))

    OP1 = OptimizationProblem(18, 1:6, f1, g1, lb, ub)
    OP2 = OptimizationProblem(18, 7:12, f2, g2_pinned, [lb; -Inf], [ub; pv_max])
    OP3 = OptimizationProblem(18, 13:18, f3, g3, lb, ub)

    gnep = [OP1 OP2 OP3]

    function extract_gnep(θ)
        Z = θ[gnep.x_inds]
        @inbounds X1a = @view(Z[1:2])
        @inbounds X1b = @view(Z[3:4])
        @inbounds p1  = @view(Z[5:6])
        @inbounds X2a = @view(Z[7:8])
        @inbounds X2b = @view(Z[9:10])
        @inbounds p2  = @view(Z[11:12])
        @inbounds X3a = @view(Z[13:14])
        @inbounds X3b = @view(Z[15:16])
        @inbounds p3  = @view(Z[17:18])
        (; X1a, X1b, p1, X2a, X2b, p2, X3a, X3b, p3)
    end
    problems = (; gnep, extract_gnep, OP1, OP2, OP3,  params=(; r, x_max, pv_max))
end

function solve_lifted(probs)
    init = rand(probs.gnep.top_level.n)
    θ = solve(probs.gnep, init)
    Z = probs.extract_gnep(θ)
end

function visualize(probs, Z)

    f = Figure(resolution=(1000,1000))
    ax = Axis(f[1,1], aspect = DataAspect())
    r = sqrt(probs.params.r) / 2
    l = probs.params.x_max
    lines!(ax, [-l, -l], [-l, +l], color=:black, linewidth=3)
    lines!(ax, [-l, +l], [+l, +l], color=:black, linewidth=3)
    lines!(ax, [+l, +l], [+l, -l], color=:black, linewidth=3)
    lines!(ax, [+l, -l], [-l, -l], color=:black, linewidth=3)

    circ_x = [r*cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [r*sin(t) for t in 0:0.1:(2π+0.1)]

    lines!(ax, circ_x .+ Z.X1a[1], circ_y .+ Z.X1a[2], color=(:red, Z.p1[1]), linewidth=3)
    lines!(ax, circ_x .+ Z.X1b[1], circ_y .+ Z.X1b[2], color=(:red, Z.p1[2]), linewidth=3)
    
    lines!(ax, circ_x .+ Z.X2a[1], circ_y .+ Z.X2a[2], color=(:green, Z.p2[1]), linewidth=3)
    lines!(ax, circ_x .+ Z.X2b[1], circ_y .+ Z.X2b[2], color=(:green, Z.p2[2]), linewidth=3)
    
    lines!(ax, circ_x .+ Z.X3a[1], circ_y .+ Z.X3a[2], color=(:blue, Z.p3[1]), linewidth=3)
    lines!(ax, circ_x .+ Z.X3b[1], circ_y .+ Z.X3b[2], color=(:blue, Z.p3[2]), linewidth=3)

    x1 = scatter!(ax, Z.X1a[1], Z.X1a[2], color=(:red, Z.p1[1]), label="P1", markersize=30)
    scatter!(ax, Z.X1b[1], Z.X1b[2], color=(:red, Z.p1[2]), markersize=30)
    x2 = scatter!(ax, Z.X2a[1], Z.X2a[2], color=(:green, Z.p2[1]), label="P2", markersize=30)
    scatter!(ax, Z.X2b[1], Z.X2b[2], color=(:green, Z.p2[2]), markersize=30)
    x3 = scatter!(ax, Z.X3a[1], Z.X3a[2], color=(:blue, Z.p3[1]), label="P3", markersize=30)
    scatter!(ax, Z.X3a[1], Z.X3a[2], color=(:blue, Z.p3[2]), markersize=30)

    f[1, 2] = Legend(f, ax)
    save("lifted.png", f)
    f
end

