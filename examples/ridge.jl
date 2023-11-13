function f1(x)
    (x[2]-x[1])^2
end

function f2(x)
    w = [-1.0; -3.5]
    d = [x[1]; x[2]] - w
    d'*d
end

function g1(x)
    [x[2],]
end

function g2(x)
    Vector{eltype(x)}()
end

function setup()
    OP1 = OptimizationProblem(2, [1,], f2, g2, [], [])
    OP2 = OptimizationProblem(2, [2,], f1, g1, [0,], [Inf, ])
    prob = [OP1; OP2]
end
