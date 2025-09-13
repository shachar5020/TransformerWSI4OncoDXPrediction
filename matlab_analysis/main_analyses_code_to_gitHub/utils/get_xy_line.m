%{ 
    For figure display purposes - finds the xlable coordinates to add a
    short legend line
%}

function [x_line, y_line] = get_xy_line(gca, ylabel_handle, direction)

ax = gca;
% Convert ylabel position from data units to normalized figure units
yl_pos = ylabel_handle.Position;  % Y-label position in data units

yl_pos = ylabel_handle.Position;  % [x, y, z] in data units

% Get Y-label extent (bounding box around the text)
yl_extent = ylabel_handle.Extent;  % [x, y, width, height] in data units

% Compute the bottom-center position
if strcmp(direction, 'left')
    yl_center_x = yl_pos(1) - yl_extent(3) / 2;  % Move right by half the width
else
    yl_center_x = yl_pos(1) + yl_extent(3) / 2;  % Move right by half the width
end
yl_bottom_y = yl_pos(2) - yl_extent(4) / 2;  % Move down by half the height


% Get necessary values
x_label_data = yl_center_x;  % X-position of the Y-label (in data units)
x_min = ax.XLim(1);  % Min X-limit of the axis
x_max = ax.XLim(2);  % Max X-limit of the axis
x_axis_start = ax.Position(1);  % Start of the axis in normalized figure units
x_axis_width = ax.Position(3);  % Width of the axis in normalized figure units

% Compute normalized X-position of the Y-label
x_line = x_axis_start + ((x_label_data - x_min) / (x_max - x_min)) * x_axis_width;

% Get necessary values
y_label_data =yl_bottom_y;  % Y-position of the Y-label (in data units)
y_min = ax.YLim(1);  % Min Y-limit of the axis
y_max = ax.YLim(2);  % Max Y-limit of the axis
y_axis_start = ax.Position(2);  % Start of the axis in normalized figure units
y_axis_height = ax.Position(4);  % Height of the axis in normalized figure units

% Compute normalized Y-position of the Y-label
y_line = y_axis_start + ((y_label_data - y_min) / (y_max - y_min)) * y_axis_height;