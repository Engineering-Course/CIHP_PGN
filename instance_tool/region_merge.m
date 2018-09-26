function refined_map = region_merge(map_combine, parsing_map, num_instance, padding)

[row, col] = size(map_combine);
refined_map = zeros([row, col]);

human_regions = {};
human_labels = {};
num_human = 0;
part_regions = {};
num_part = 0;

for i = 1:num_instance
    region_i = (map_combine == i);
    if sum(region_i(:)) == 0
        continue;
    end
    parsing_i = parsing_map(region_i);
    num_label = unique(parsing_i);
    if length(num_label) > 1 && sum(region_i(:)) > 30
        num_human = num_human + 1;
        human_regions{num_human} = region_i;
        human_labels{num_human} = i;
    else
        num_part = num_part + 1;
        part_regions{num_part} = region_i;
    end
end

for i = 1:num_part
   part_i = part_regions{i}; 
   boundary_i = double(bwmorph(part_i, 'remove', inf));
   [ele_row, ele_col] = find(boundary_i > 0);
   is_merge = 0;
   for e = 1:length(ele_row)
      if (is_merge == 1)
          break;
      end
      rr = ele_row(e);
      cc = ele_col(e);
      cur_label = map_combine(rr,cc);
      if is_merge == 0 && (rr - 2 > 0 && map_combine(rr-2,cc) > 0 && map_combine(rr-2,cc) ~= cur_label)
          for p = i+1:num_part
              part_p = part_regions{p};
              if part_p(rr-2,cc) > 0
                  is_merge = 1;
                  map_combine(part_p) = cur_label;
                  part_regions{p} = (part_i + part_p) > 0;
                  break;
              end
          end
          for h = 1:num_human
              human_i = human_regions{h};
              if human_i(rr-2,cc) > 0
                  is_merge = 1;
                  map_combine(part_i) = human_labels{h};
                  break;
              end
          end
      elseif is_merge == 0 && (cc - 2 > 0 && map_combine(rr,cc-2) > 0 && map_combine(rr,cc-2) ~= cur_label)
          for p = i+1:num_part
              part_p = part_regions{p};
              if part_p(rr,cc-2) > 0
                  is_merge = 1;
                  map_combine(part_p) = cur_label;
                  part_regions{p} = (part_i + part_p) > 0;
                  break;
              end
          end
          for h = 1:num_human
              human_i = human_regions{h};
              if human_i(rr,cc-2) > 0
                  is_merge = 1;
                  map_combine(part_i) = human_labels{h};
                  break;
              end
          end
      elseif is_merge == 0 && (rr + 2 <= row && map_combine(rr+2,cc) > 0 && map_combine(rr+2,cc) ~= cur_label)
          for p = i+1:num_part
              part_p = part_regions{p};
              if part_p(rr+2,cc) > 0
                  is_merge = 1;
                  map_combine(part_p) = cur_label;
                  part_regions{p} = (part_i + part_p) > 0;
                  break;
              end
          end
          for h = 1:num_human
              human_i = human_regions{h};
              if human_i(rr+2,cc) > 0
                  is_merge = 1;
                  map_combine(part_i) = human_labels{h};
                  break;
              end
          end
      elseif is_merge == 0 && (cc + 2 <= col && map_combine(rr,cc+2) > 0 && map_combine(rr,cc+2) ~= cur_label)
          for p = i+1:num_part
              part_p = part_regions{p};
              if part_p(rr,cc+2) > 0
                  is_merge = 1;
                  map_combine(part_p) = cur_label;
                  part_regions{p} = (part_i + part_p) > 0;
                  break;
              end
          end
          for h = 1:num_human
              human_i = human_regions{h};
              if human_i(rr,cc+2) > 0
                  is_merge = 1;
                  map_combine(part_i) = human_labels{h};
                  break;
              end
          end
      end
   end
end

if padding > 0

    for r = 1:row
        if map_combine(r, col-padding) ~= 0
            for c = col-padding-1:col
                if map_combine(r,c) == 0
                    map_combine(r,c) = map_combine(r,c-1);
                end
            end
        end
        if map_combine(r, padding) ~= 0
            for c = padding-1:-1:1
                if map_combine(r,c) == 0
                    map_combine(r,c) = map_combine(r,c+1);
                end
            end
        end
    end

    for c = 1:col
        if map_combine(row-padding, c) ~= 0
            for r = row-padding-1:row
                if map_combine(r,c) == 0
                    map_combine(r,c) = map_combine(r-1,c);
                end
            end
        end
        if map_combine(padding,c) ~= 0
            for r = padding-1:-1:1
                if map_combine(r,c) == 0
                    map_combine(r,c) = map_combine(r+1,c);
                end
            end
        end
    end
end

refined_map = map_combine;

