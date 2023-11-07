struct OptimizationProblem
    n::Int # domain dimension
    dvars::Vector{Int}
    f::Function
    g::Function
    l::Vector{Float64}
    u::Vector{Float64}
end

#function OptimizationProblem(n, dvars, f, g, l, u)
#    (m = length(l)) == length(u) || error("length of l and u are not equal")
#
#    x_sym, λ_sym = Symbolics.@variables(x_sym[1:n], λ_sym[1:m]) .|> Symbolics.scalarize
#
#    g_sym = g(x_sym)
#    ℒ = f(x_sym) - λ_sym'*g_sym
#    ∇ℒ = Symbolics.gradient(ℒ, x_sym)
#
#    ∇g = Symbolics.sparsejacobian(g_sym, x_sym)
#    (rows_g, cols_g, sym_vals_g) = findnz(∇g)
#
#    ∇²ℒ = Symbolics.sparsejacobian(∇ℒ, x_sym)
#    (rows_ℒ, cols_ℒ, sym_vals_ℒ) = findnz(∇²ℒ)
#
#    ∇ℒ! = Symbolics.build_function(∇ℒ, x_sym, λ_sym; expression = Val(false))[2]
#
#    ∇²ℒ_vals! = Symbolics.build_function(sym_vals_ℒ, x_sym, λ_sym; expression = Val(false))[2]
#    ∇g_vals! = Symbolics.build_function(sym_vals_g, x_sym; expression = Val(false))[2]
#
#    OptimizationProblem(n, m, dvars, f, g, l, u, ∇ℒ!, ∇²ℒ_vals!, rows_ℒ, cols_ℒ, ∇g_vals!, rows_g, cols_g)
#end

function Base.hcat(OPs::OptimizationProblem...)
    @assert allequal(OP.n for OP in OPs)
    @assert isempty(intersect((OP.dvars for OP in OPs)...))
    N = length(OPs)
    n = first(OPs).n
    n_privates = [length(OP.dvars) for OP in OPs]
    m_privates = [length(OP.l) for OP in OPs] 
    m = sum(m_privates)
    all_dvars = union((OP.dvars for OP in OPs)...)
    pvars = setdiff(1:n, all_dvars)

    xs = map(1:N) do i
        sym = Symbol("x", i)
        Symbolics.@variables($sym[1:n_privates[i]])[1] |> Symbolics.scalarize
    end
    λs = map(1:N) do i
        sym = Symbol("λ", i)
        Symbolics.@variables($sym[1:m_privates[i]])[1] |> Symbolics.scalarize
    end
    ss = map(1:N) do i
        sym = Symbol("s", i)
        Symbolics.@variables($sym[1:m_privates[i]])[1] |> Symbolics.scalarize
    end
    xp = Symbolics.@variables(xp[1:length(pvars)])[1] |> Symbolics.scalarize

    x = Vector{Num}(undef, n)
    for (e, OP) in enumerate(OPs)
        x[OP.dvars] = xs[e]
    end
    x[pvars] = xp
    λ = vcat(λs...)
    s = vcat(ss...)

    grad_lags = mapreduce(vcat, enumerate(OPs)) do (i, OP)
        Lag = OP.f(x) - λs[i]'*OP.g(x)
        grad_lag = Symbolics.gradient(Lag, xs[i])
    end
    cons_s = mapreduce(vcat, enumerate(OPs)) do (i, OP)
        OP.g(x) - ss[i]
    end
    F = [grad_lags; cons_s; λ]
    F! = Symbolics.build_function(F, x, λ, s; expression = Val(false))[2]
    FF = (result, z) -> begin
        F!(result, @view(z[1:n]), @view(z[n+1:n+m]), @view(z[n+m+1:end]))
    end

    J = Symbolics.sparsejacobian(F, [x; λ; s])
    (rows, cols, vals) = findnz(J)
    J_vals! = Symbolics.build_function(vals, x, λ, s; expression = Val(false))[2]
    JV = (result, z) -> begin
        J_vals!(result, @view(z[1:n]), @view(z[n+1:n+m]), @view(z[n+m+1:end]))
    end

    l = fill(-Inf, sum(n_privates)+m)
    append!(l, (OP.l for OP in OPs)...)
    u = fill(Inf, sum(n_privates)+m)
    append!(u, (OP.u for OP in OPs)...)

    mcp_vars = setdiff(1:n+2m, pvars)
    
    MixedComplementarityProblem(n+2m, FF, JV, rows, cols, l, u, mcp_vars, pvars)
end

function Base.hvcat(blocks_per_row::Tuple{Vararg{Int}}, OPs::OptimizationProblem...)
    @assert length(blocks_per_row) == 2
    N1, N2 = blocks_per_row
    
    low_level = hcat(OPs[N1+1:end]...)
    num_ll_cons = 2*length(low_level.l)
    
end

struct MixedComplementarityProblem
    n::Cint
    F::Function
    J_vals::Function
    J_rows::Vector{Cint}
    J_cols::Vector{Cint}
    l::Vector{Cdouble}
    u::Vector{Cdouble}
    dvars::Vector{Cint}
    pvars::Vector{Cint}
end

function solve(mcp::MixedComplementarityProblem, z0=Vector{Cdouble}; silent=false)
    @assert length(z0) == mcp.n

    function z_full(z_partial)
        zf = zeros(eltype(z_partial),mcp.n)
        zf[mcp.dvars] .= z_partial
        zf[mcp.pvars] .= z0[mcp.pvars]
        zf
    end
    
    function F(n, z, result)
        zf = z_full(z)
        mcp.F(result, zf)
        Cint(0)
    end

    d_mask = map(mcp.J_cols) do i
        i ∈ mcp.dvars
    end
    p_mask = .!d_mask

    nnz_d = sum(d_mask)
    nnz_p = sum(p_mask)

    # Jacobian is with respect to all variables, including parameter variables.
    # PATH needs to reason about jacobian of F w.r.t. decision variables only.
    # The following code splits J into the portions w.r.t. dvars and pvars, for
    # use in the function below.
    J_full = sparse(mcp.J_rows, mcp.J_cols, Vector{Cdouble}(undef, length(mcp.J_rows)), length(mcp.dvars), mcp.n)
    J_dvars = J_full[:, mcp.dvars]
    J_pvars = J_full[:, mcp.pvars]
    val_buffer = zeros(Cdouble, nnz_d+nnz_p)
    J_col = J_dvars.colptr[1:end-1]
    J_len = diff(J_dvars.colptr)
    J_row = J_dvars.rowval

    function J(n, nnz, z, col, len, row, data)
        val_buffer .= 0.0
        zf = z_full(z)
        mcp.J_vals(val_buffer, zf)
        data .= val_buffer[d_mask]
        col .= J_col
        len .= J_len
        row .= J_row
        Cint(0)
    end

    status, z_out, info = PATHSolver.solve_mcp(
         F,
         J,
         mcp.l,
         mcp.u,
         z0[mcp.dvars];
         silent,
         nnz = nnz_d,
         jacobian_structure_constant = true,
         jacobian_data_contiguous = true,
     ) 
    
    (; status, z=z_full(z_out), info)
end

function get_local_solution_graph(mcp, z; tol=1e-5)
    f = zeros(mcp.dvars |> length)
    mcp.F(f, z)
    l = mcp.l  
    u = mcp.u
    z_dvars = @view z[mcp.dvars]
    z_pvars = @view z[mcp.pvars]
    n = length(z_dvars)
    J = Dict{Int, Vector{Int}}()
    for i in 1:n
        Ji = Int[]
        if f[i] ≥ tol && z_dvars[i] < l[i]+tol
            push!(Ji, 1)
        elseif -tol < f[i] < tol && z_dvars[i] < l[i]+tol
            push!(Ji, 1)
            push!(Ji, 2)
        elseif -tol < f[i] < tol && l[i]+tol ≤ z_dvars[i] ≤ u[i]-tol
            push!(Ji, 2)
        elseif -tol < f[i] < tol && z_dvars > u[i]-tol
            push!(Ji, 2)
            push!(Ji, 3)
        elseif f[i] ≤ -tol && z_dvars > u[i]-tol
            push!(Ji, 3)
        elseif isapprox(l[i], u[i]; atol=tol)
            push!(Ji, 4)
        end
        J[i] = Ji
    end
    valid_solution = !any(isempty.(Ji for Ji in values(J)))
    !valid_solution && error("Not a valid solution!")
    
    recipes = get_all_recipes(J)

end

function get_all_recipes(J)
    Ks = Vector{Dict{Int, Set{Int}}}()
    N = length(J)
    multiples = [i for i in 1:N if length(J[i]) > 1]
    singles = setdiff(1:N, multiples)
    It = Iterators.product([J[i] for i in multiples]...)
    for assignment in It
        K = Dict(j=>Set{Int}() for j in 1:4)
        for (e,ej) in enumerate(assignment)
            push!(K[ej], e)
        end
        for e in singles
            push!(K[J[e]], e)
        end
        push!(Ks, K)
    end
    Ks
end

function convert_mcp_recipe(mcp, recipe)
    n = length(mcp.l)

    lf = zeros(n)
    uf = zeros(n)
    lz = zeros(n)
    uz = zeros(n)

    for i in K[1]
        lf[i] = 0.0
        uf[i] = Inf
        lz[i] = mcp.l[i]
        uz[i] = mcp.l[i]
    end
    for i in K[2]
        lf[i] = 0.0
        uf[i] = 0.0
        lz[i] = mcp.l[i]
        uz[i] = mcp.u[i]
    end
    for i in K[3]
        lf[i] = -Inf
        uf[i] = 0.0
        lz[i] = mcp.u[i]
        uz[i] = mcp.u[i]
    end
    for i in K[4]
        lf[i] = -Inf
        uf[i] = Inf
        lz[i] = mcp.l[i]
        uz[i] = mcp.u[i]
    end
    (; lf, uf, lz, uz, mcp)
end
