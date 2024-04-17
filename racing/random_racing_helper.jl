include("road.jl")
#const xdim = 4
#const udim = 2
# generate x0s
# also generates roads
function generate_x0s(sample_size, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    # choose random P1 lateral position inside the lane limits, long pos = 0
    #c, r = get_road(0; road);
    # solve quadratic equation to find x intercepts of the road
    #lax_max = sqrt((r - road_d)^2 - c[2]^2) + c[1]
    #lax_max = sqrt((r + road_d)^2 - c[2]^2) + c[1]
    #lat_max = min()
    roads = Vector{Dict{Float64, Float64}}(undef, sample_size)

    a_lat_pos0_arr = -lat_max .+ 2 * lat_max .* rand(MersenneTwister(), sample_size)  # .5 .* ones(sample_size)
    # fix P1 longitudinal pos at 0
    a_pos0_arr = hcat(a_lat_pos0_arr, zeros(sample_size, 1))
    b_pos0_arr = zeros(size(a_pos0_arr))
    # choose random radial offset for P2
    for i in 1:sample_size
        roads[i] = gen_road()

        # shift initial position wrt to road
        road_ys = roads[i] |> keys |> collect
        sortedkeys = sortperm((road_ys .- 0) .^ 2)
        a_pos0_arr[i, 1] += roads[i][road_ys[sortedkeys[1]]] 

        r_offset = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister()))
        ϕ_offset = rand(MersenneTwister()) * 2 * π
        b_lat_pos0 = a_pos0_arr[i, 1] + r_offset * cos(ϕ_offset)
        # reroll until we b lat pos is inside the lane limits
        while b_lat_pos0 > lat_max || b_lat_pos0 < -lat_max
            r_offset = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister()))
            ϕ_offset = rand(MersenneTwister()) * 2 * π
            b_lat_pos0 = a_pos0_arr[i, 1] + r_offset * cos(ϕ_offset)
        end
        b_long_pos0 = a_pos0_arr[i, 2] + r_offset * sin(ϕ_offset)
        # offset by road
        sortedkeys = sortperm((road_ys .- b_long_pos0) .^ 2) 
        b_lat_pos0 += roads[i][road_ys[sortedkeys[1]]]
        b_pos0_arr[i, :] = [b_lat_pos0, b_long_pos0]
    end

    @assert minimum(sqrt.(sum((a_pos0_arr .- b_pos0_arr) .^ 2, dims=2))) >= 1.0 # probs.params.r
    #@assert all(-lat_max .< b_pos0_arr[:, 1] .< lat_max)
    
    #Plots.scatter(a_pos0_arr[:, 1], a_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)
    #Plots.scatter!(b_pos0_arr[:, 1], b_pos0_arr[:, 2], aspect_ratio=:equal, legend=false)

    # keep lateral velocity zero

    a_long_vel_min = b_long_vel_delta_max
    a_vel0_arr = hcat(zeros(sample_size), a_long_vel_min .+ (a_long_vel_max - a_long_vel_min) .* rand(MersenneTwister(), sample_size))
    #a_vel0_arr = hcat(zeros(sample_size), a_long_vel_max .* rand(MersenneTwister(), sample_size))

    b_vel0_arr = zeros(size(a_vel0_arr))
    # choose random velocity offset for 
    for i in 1:sample_size
        b_long_vel0_offset = -b_long_vel_delta_max + 2 * b_long_vel_delta_max .* rand(MersenneTwister())
        b_long_vel0 = a_vel0_arr[i, 2] + b_long_vel0_offset
        ## reroll until b long vel is nonnegative
        #while b_long_vel0 < 0
        #    b_long_vel0_offset = -b_long_vel_delta_max + 2 * b_long_vel_delta_max .* rand(MersenneTwister())
        #    b_long_vel0 = a_vel0_arr[i, 2] + b_long_vel0_offset
        #end
        b_vel0_arr[i, 2] = b_long_vel0
    end

    #@infiltrate
    #x0_arr = hcat(a_pos0_arr, a_vel0_arr, b_pos0_arr, b_vel0_arr)
    # really simple since lateral vel is zero
    x0_arr = hcat(a_pos0_arr, a_vel0_arr[:, 2], ones(length(a_vel0_arr[:, 1])) .* π / 2, b_pos0_arr, b_vel0_arr[:, 2], ones(length(b_vel0_arr[:, 1])) .* π / 2)
    #@infiltrate
    x0s = Dict()

    for (index, row) in enumerate(eachrow(x0_arr))
        x0s[index] = row
    end
    (x0s, roads)
end
