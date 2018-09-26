function [map_horizontal, map_vertical, map_combine, refined_map] = search(parsing_map, input_edge, edge_thresh)
%%
threshold = 1;
padding = 10;
    
[row, col] = size(parsing_map);

    
    edge_map = (input_edge>=edge_thresh);
    edge_map = double(bwmorph(edge_map, 'close', inf));
    edge_map = double(bwmorph(edge_map, 'thin', inf));
    
    map_vertical = zeros([row, col]);
    map_horizontal = zeros([row, col]);
    
    num_ins = 0;
    for c = 1:col
        for r = 1:row
            if edge_map(r, c) == threshold
                if row - r < padding
                   for a_r = r+1:row
                       edge_map(a_r,c) = threshold;
                   end
                end
                continue;
            end
            if parsing_map(r,c) ~= 0
                if c-1 > 0 && sum(map_vertical(:,c-1)) == 0 && sum(map_vertical(:,c)) == 0
                    num_ins = num_ins + 1;
                    map_vertical(r,c) = num_ins;
                else
                    if r-1 > 0 && map_vertical(r-1, c) ~= 0;
                        map_vertical(r,c) = map_vertical(r-1, c);
                    end
                    if map_vertical(r,c) == 0
                        num_ins = num_ins + 1;
                        map_vertical(r,c) = num_ins;
                    end
                end
            end
        end
    end

    num_ins = 0;
    for r = 1:row
        for c = 1:col
            if edge_map(r, c) == threshold
                if col - c < padding
                   for a_c = c+1:col
                       edge_map(r,a_c) = threshold;
                   end
                end
                continue;
            end
            if parsing_map(r,c) ~= 0
                if r-1 > 0 && sum(map_horizontal(r-1,:)) == 0 && sum(map_horizontal(r,:)) == 0
                    num_ins = num_ins + 1;
                    map_horizontal(r,c) = num_ins;
                else
                    if c-1 > 0 && map_horizontal(r, c-1) ~= 0;
                        map_horizontal(r,c) = map_horizontal(r, c-1);
                    end
                    if map_horizontal(r,c) == 0
                        num_ins = num_ins + 1;
                        map_horizontal(r,c) = num_ins;
                    end
                end
            end
        end
    end

    [map_combine, num_instance] = bfs(map_horizontal, map_vertical);
    
    refined_map = region_merge(map_combine, parsing_map, num_instance, padding);
    num_refined_instance = length(unique(refined_map)) - 1;
    while num_refined_instance ~= num_instance
        num_instance = num_refined_instance;
        refined_map = region_merge(refined_map, parsing_map, num_instance, 0);
        num_refined_instance = length(unique(refined_map)) - 1;
    end
    
    

