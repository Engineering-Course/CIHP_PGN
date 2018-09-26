clear;
close all;
fclose all;
%%
load('pascal_seg_colormap.mat');
parsing_folder = fullfile('../output/cihp_parsing_maps');
edge_folder = fullfile('../output/cihp_edge_maps');
human_folder = fullfile('../output/cihp_human_maps');
filelist = textread('../datasets/CIHP/list/val_id.txt', '%s');

instance_folder = '../output/cihp_instance_part_maps';
if ~exist(instance_folder, 'dir')
    mkdir(instance_folder);
end

class_num = 19;

for i = 1:length(filelist)
    img_fn = filelist{i};  %;
    fprintf('num: %d, %s\n', i, img_fn);
    
    human_map = imread(fullfile(human_folder, [img_fn '.png']));
    parsing_map = imread(fullfile(parsing_folder, [img_fn '.png']));
    human_edge = imread(fullfile(edge_folder, [img_fn '.png']));
    human_edge = double(human_edge) / 255;
    
    parsing_score_data = load(fullfile(parsing_folder, [img_fn '.mat']));
    parsing_score_map = parsing_score_data.data;
    
    part_edge = double(imgradient(parsing_map) > 0);
    edge_map = part_edge + human_edge;
    
    instance_map = zeros(size(human_map));
    
    fid = fopen(fullfile(instance_folder, [filelist{i} '.txt']), 'w');
    counter = 0;
    for k = 1:class_num
        scores = [];
        cur_counter = counter;
        indices = (parsing_map == k);
        part_map = uint8(double(indices) .* double(human_map));
        label_id = unique(part_map);
        for l = 1:length(label_id)
            if label_id(l) ~= 0
                counter = counter + 1;
                indices = find(part_map == label_id(l));
                instance_map(indices) = counter;
                edge_num = sum(edge_map(indices));
                edge_score = edge_num / sum(edge_map(indices)>0);
                parsing_score = mean(parsing_score_map(indices));
                scores(counter) = length(indices) * (parsing_score * edge_score);
            end
        end
        
        if cur_counter < counter
            for o = cur_counter:counter-1
                fprintf(fid, '%d %f\n', k, scores(o+1));
            end
        end
    end
    fclose(fid);

    imwrite(uint8(instance_map), colormap, fullfile(instance_folder, [img_fn '.png']));
end    

