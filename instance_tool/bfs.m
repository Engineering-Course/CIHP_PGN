function [instance_map, num_instance] = bfs(instance_map_horizontal, instance_map_vertical)

global horizontal_map vertical_map marked queue_sum row col;

horizontal_map = instance_map_horizontal;
vertical_map = instance_map_vertical;

[row, col] = size(instance_map_horizontal);
instance_map = zeros([row, col]);
marked = zeros([row*col,1]);
queue_sum = {};

for i = 1:row
    for j = 1:col
        if marked((i-1) * col + j) == 0 && horizontal_map(i,j) ~= 0 && vertical_map(i,j) ~= 0
            bfs_search((i-1) * col + j)
        end
    end
end

num_instance = length(queue_sum);

for i = 1:length(queue_sum)
    region = queue_sum{i};
    for j= 1:length(region)
        x = floor((region(j)-1)/col)+1;
        y = mod(region(j)-1, col)+1;
        instance_map(x,y) = i;
    end
end
    


function [] = bfs_search(start)

global marked queue_sum

queue = start;
region_queue = [];
while ~isempty(queue)
    node = queue(1);
    queue(1) = [];
    if marked(node) == 0
        marked(node) = 1;
        region_queue = [region_queue node];
        
        neighbours = find_neighbour(node);
        for i = 1:length(neighbours)
            queue = [queue neighbours(i)];
        end
    end
end
if length(region_queue) > 0
    point_num = length(region_queue);
    index = 1;
    for q = 1:length(queue_sum)
        point_q = length(queue_sum{q});
        if point_num < point_q 
            index = index + 1;
            continue;
        end
    end
    if ~isempty(queue_sum)
        for p = length(queue_sum)+1 : -1 : index+1
            queue_sum{p} = queue_sum{p-1};
        end
    end
    queue_sum{index} = region_queue;
    
end

function neighbours = find_neighbour(node)

global marked horizontal_map vertical_map col row;

i = floor((node-1)/col)+1;
j = mod(node-1, col)+1;
neighbours = [];

x = i-1;
y = j;
if x > 0 && vertical_map(x, y) == vertical_map(i, j) && marked((x-1)*col+y) == 0 && vertical_map(x, y) ~= 0
    neighbours = [neighbours (x-1)*col+y];
end

x = i+1;
y = j;
if x <= row && vertical_map(x,y) == vertical_map(i, j) && marked((x-1)*col+y) == 0 && vertical_map(x, y) ~= 0
    neighbours = [neighbours (x-1)*col+y];
end

x = i;
y = j-1;
if y > 0 && horizontal_map(x,y) == horizontal_map(i, j) && marked((x-1)*col+y) == 0 && horizontal_map(x,y) ~= 0
    neighbours = [neighbours (x-1)*col+y];
end

x = i;
y = j+1;
if y <= col && horizontal_map(x,y) == horizontal_map(i, j) && marked((x-1)*col+y) == 0 && horizontal_map(x,y) ~= 0
    neighbours = [neighbours (x-1)*col+y];
end
                    

        
        
        