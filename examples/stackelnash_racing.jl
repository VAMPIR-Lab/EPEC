# Stackelnash racing
# p1, p2 ∈ ℝ² (initial positions of vehicles, parameters
# d1, d2 ∈ ℝ² (motion of vehicles)
# x := [d1; d2; p1; p2]

function f1(x)
    d1 = @view(x[1:2])
    d2 = @view(x[3:4])
    p1 = @view(x[5:6])
    p2 = @view(x[7:8])

    -(p1+d1)[1] + (p2+d2)[1] + 0.5 * d1'*d1 - 0.5 * d2'*d2
end

function f2(x)
    -f1(x)
end

function g1(x)
    d1 = @view(x[1:2])
    d2 = @view(x[3:4])
    p1 = @view(x[5:6])
    p2 = @view(x[7:8])
    [(d1+p1)[2];
     (d1+p1-d2-p2)'*(d1+p1-d2-p2)]
end

function g2(x)
    d2 = @view(x[3:4])
    p2 = @view(x[7:8])
    [(d2+p2)[2],]
end

function setup()
    OP1 = OptimizationProblem(8, [1,2], f1, g1, [-1.0, 1.0], [1.0, Inf])
    OP2 = OptimizationProblem(8, [3,4], f2, g2, [-1.0,], [1.0,])

    MCP = [OP1 OP2]
end

