function gen_road()
    road_base = Dict(
        -2 => 0.01,
        0 => 0,
        2 => 0.01,
        4 => 0,
        6 => 0.01,
        8 => 0,
        10 => 0.01,
        12 => 0.0, # begin right turn
        12.5 => 0.09,
        13 => 0.25,
        13.5 => 0.5,
        14 => 0.8,
        15 => 1.6,
        16 => 2.3,
        16.5 => 2.45,
        17 => 2.5, # right peak, begin left turn
        17.5 => 2.45,
        18 => 2.3,
        19 => 1.6,
        20 => 0.6,
        21 => -0.6,
        22 => -1.6,
        23 => -2.3,
        23.5 => -2.45,
        24 => -2.5, # left peak
        24.5 => -2.45,
        25 => -2.3,
        26 => -1.6,
        27 => -0.8,
        27.5 => -0.5,
        28 => -0.25,
        28.5 => -0.09,
        29 => 0 # end of turns
    )

    # randomize road
	road_sorted = road_base |> sort
    road_len = (road_sorted |> keys |> collect |> last) - (road_sorted |> keys |> collect |> first)
    road_y_offset = (0.5 - rand()) * road_len
	#road_y_offset = road_len/3-3.5;
    road_offset = Dict{Float64,Float64}()

    for (y, x) in road_sorted
        road_offset[y+road_y_offset] = x
        road_offset[y+road_y_offset-road_len-1] = x
        road_offset[y+road_y_offset+road_len+1] = x
    end
    road_offset
end