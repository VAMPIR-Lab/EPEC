# Example 4.3.1 from https://web.stanford.edu/group/SOL/dissertations/clsu-thesis.pdf

# X := [f1 f2 f3 s1 s2 s3]

function market_price(q; p̄=10)
    p̄ - q
end

function neg_forward_profit(X, player_id, c)
    total_prod = X[player_id] + X[3+player_id]
    profit = market_price(sum(X)) * total_prod - c*total_prod
    -profit
end

function neg_spot_profit(X, player_id, c)
    total_prod = X[player_id] + X[3+player_id]
    profit = market_price(sum(X)) * X[3+player_id] - c*total_prod
    -profit
end

function setup()
    c1 = 2.0
    c2 = 3.0
    c3 = 4.0
    OP1 = OptimizationProblem(6, [1,], (x->neg_forward_profit(x, 1, c1)), (x->[x[1],]), [0.0,], [Inf,])
    OP2 = OptimizationProblem(6, [2,], (x->neg_forward_profit(x, 2, c2)), (x->[x[2],]), [0.0,], [Inf,])
    OP3 = OptimizationProblem(6, [3,], (x->neg_forward_profit(x, 3, c3)), (x->[x[3],]), [0.0,], [Inf,])
    OP4 = OptimizationProblem(6, [4,], (x->neg_spot_profit(x, 1, c1)), (x->[x[4],]), [0.0,], [Inf,])
    OP5 = OptimizationProblem(6, [5,], (x->neg_spot_profit(x, 2, c2)), (x->[x[5],]), [0.0,], [Inf,])
    OP6 = OptimizationProblem(6, [6,], (x->neg_spot_profit(x, 3, c3)), (x->[x[6],]), [0.0,], [Inf,])

    epec = [OP1 OP2 OP3; OP4 OP5 OP6]
end
