clear;
close all;
fclose all;
%%
load('pascal_seg_colormap.mat');
parsing_folder = fullfile('../output/cihp_parsing_maps');
edge_folder = fullfile('../output/cihp_edge_maps');
filelist = textread('../datasets/CIHP/list/val_id.txt', '%s');

out_dir = fullfile('../output/cihp_human_maps');
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end

edge_thresh = 0.2;   % 
human_class_id = 1;

for i = 1:length(filelist)
    
    img_fn = filelist{i};  
    fprintf('num: %d, %s\n', i, img_fn);
    
    parsing_map = imread(fullfile(parsing_folder, [img_fn '.png']));
    edge_ave_map = imread(fullfile(edge_folder, [img_fn '.png']));
    edge_ave_map = double(edge_ave_map) / 255;

    [map_horizontal, map_vertical, map_combine, refined_map] = search(parsing_map, edge_ave_map, edge_thresh);
%%

    out_map = zeros(size(refined_map));
    
    max_ins = max(refined_map(:));
    sum_map = 0;
    count = 0;
    for ins = 1:max_ins
        indices = find(refined_map==ins);
        if sum(indices(:)) > 0
            count = count + 1;
            out_map(indices) = count;
            sum_map(count) = length(indices);
        end
    end
    fid = fopen(fullfile(out_dir, [img_fn, '.txt']), 'w');
    for c = 1:count
        fprintf(fid, '%d, %f\n', human_class_id, sum_map(c));
    end
    fclose(fid);
    imwrite(uint8(out_map), colormap, fullfile(out_dir, [img_fn '.png']));
    
end

