function pool5_explorer()

% load pre-computed index into variable 'index'
load cachedir/convnet-selective-search/pool5_explorer_index_test_2007_fine_tuned_1_zca.mat;

figures = [1 2];
vf_sz = [8 12];
redraw = true;
position = 1;

channel = 0;
cell_y = 2;
cell_x = 2;
feature_index = get_feature_index(channel, cell_y, cell_x);

%channel_width = display_feature_grid(figures(1));

while 1
  if redraw
    feature_index = get_feature_index(channel, cell_y, cell_x);
    visualize_feature(figures(2), index, feature_index, position, vf_sz, cell_y, cell_x, channel);
    redraw = false;
  end

  % display 16x16 grid where each cell shows a 6x6 feature map

  % wait for mouse click
  %  -> map click coordinates to feature dimension

  %[feature_index_, channel_, cell_y_, cell_x_, key_code] = get_feature_selection(channel_width);
  [~, ~, ~, ~, key_code] = get_feature_selection(1);

%  if gcf == figures(1)
%    feature_index = feature_index_;
%    channel = channel_;
%    cell_y = cell_y_;
%    cell_x = cell_x_;
%    fprintf('%d (%d %d)@%d\n', feature_index, cell_y, cell_x, channel);
%  end

  switch key_code
    case 27 % ESC
      close(figures(ishandle(figures)));
      return;

    case '`'
      return;

    case 's' % take a snapshot
      filename = sprintf('./vis/pool5-explorer/shots/x%d-y%d-c%d-p%d.pdf', ...
                         cell_y, cell_x, channel, position);
      if exist(filename)
        delete(filename);
      end
      export_fig(filename);

    case 'g' % go to a specific channel
      answer = str2double(inputdlg('go to channel:'));
      if ~isempty(answer)
        answer = round(answer);
        if answer > 0
          channel = answer - 1;
          redraw = true;
        end
      end

    case 31 % up
      % decrease channel
      if channel > 0
        channel = channel - 1;
        position = 1;
        redraw = true;
      end

    case 30 % down
      % increase channel
      if channel < 255
        channel = channel + 1;
        position = 1;
        redraw = true;
      end

    case 'i' % cell up
      if cell_y > 0
        cell_y = cell_y - 1;
        position = 1;
        redraw = true;
      end

    case 'k' % cell down
      if cell_y < 5
        cell_y = cell_y + 1;
        position = 1;
        redraw = true;
      end

    case 'j' % cell left
      if cell_x > 0
        cell_x = cell_x - 1;
        position = 1;
        redraw = true;
      end

    case 'l' % cell right
      if cell_x < 5
        cell_x = cell_x + 1;
        position = 1;
        redraw = true;
      end

    case 29 % ->
      new_pos = position + prod(vf_sz);
      if new_pos < length(index.features{feature_index}.scores)
        position = new_pos;
        redraw = true;
      end

    case 28 % <-
      new_pos = position - prod(vf_sz);
      if new_pos > 0
        position = new_pos;
        redraw = true;
      end
    otherwise
      fprintf('%d\n', key_code);
    %  visualize_feature(figures(2), index, feature_index, position, vf_sz);
  end
end


% ------------------------------------------------------------------------
function f = get_feature_index(channel, cell_y, cell_x)
% ------------------------------------------------------------------------
% flip because of nasty upside down image issue
cell_y = 5 - cell_y;
f = channel*36 + cell_y*6 + cell_x + 1;


% ------------------------------------------------------------------------
function visualize_feature(fig, index, f, position, msz, cell_y, cell_x, channel)
% ------------------------------------------------------------------------
conf = voc_config();
VOCopts = conf.pascal.VOCopts;

max_val = 0;
for x_ = 0:5
  for y_ = 0:5
    f_ = get_feature_index(channel, y_, x_);
    max_val = max([max_val; index.features{f_}.scores]);
  end
end

s = 224/6;
points = round(s/2:s:224);

M = zeros(6,6,256);
M(f) = 1;
M = sum(M, 3)';

[r,c] = find(M);
r1 = max(1, points(r) - 81);
r2 = min(224, points(r) + 81);
c1 = max(1, points(c) - 81);
c2 = min(224, points(c) + 81);
h = r2-r1;
w = c2-c1;

psx = 96;
psy = 96;
h = h * psy/224;
w = w * psx/224;

% flip Y to compensate for the fact that the net runs on flipped images
r1 = 224 - r2 + 1;
r1 = (r1-1)*psy/224 + 1;
c1 = (c1-1)*psx/224 + 1;

ims = {};
start_pos = position;
end_pos = min(length(index.features{f}.scores), start_pos + prod(msz) - 1);
N = end_pos - start_pos + 1;
str = sprintf('pool5 feature: (%d,%d,%d) (top %d - %d)', cell_y+1, cell_x+1, channel+1, start_pos, end_pos);
for i = start_pos:end_pos
  val = index.features{f}.scores(i);
  image_ind = index.features{f}.image_inds(i);
  bbox = index.features{f}.boxes(i, :);

  im = imread(sprintf(VOCopts.imgpath, index.images{image_ind})); 
  im = imresize(im(bbox(2):bbox(4), bbox(1):bbox(3), :), [psy psx], 'bilinear');
  ims{end+1} = im;
end
filler = prod(msz) - N;
im = my_montage(cat(4, ims{:}, 256*ones(psy, psx, 3, filler)), msz);
figure(2);
clf;
imagesc(im);
title(str, 'Color', 'black', 'FontSize', 18, 'FontName', 'Times New Roman');
axis image;
axis off;
set(gcf, 'Color', 'white');
q = 1;
for y = 0:msz(1)-1
  for x = 0:msz(2)-1
    if q > N
      break;
    end
    x1 = c1+psx*x;
    y1 = r1+psy*y;
    rectangle('Position', [x1 y1 w h], 'EdgeColor', 'w', 'LineWidth', 3);
    text(x1, y1+7.5, sprintf('%.1f', index.features{f}.scores(start_pos+q-1)/max_val), 'BackgroundColor', 'w', 'FontSize', 10, 'Margin', 0.1, 'FontName', 'Times New Roman');
    q = q + 1;
  end
  if q > N
    break;
  end
end

% compute mean figure
%scores = index.features{f}.scores(start_pos:end_pos);
%for i = 1:length(ims)
%  ims{i} = double(ims{i})*scores(i)/sum(scores);
%end
%figure(1);
%imagesc(uint8(sum(cat(4, ims{:}), 4)));
%axis image;
%figure(2);


% ------------------------------------------------------------------------
function [feature_index, channel, cell_y, cell_x, ch] = get_feature_selection(channel_width)
% ------------------------------------------------------------------------
while 1
  [x,y,ch] = ginput(1);
  chan_y = floor(y/channel_width);
  chan_x = floor(x/channel_width);
  channel = chan_y*16 + chan_x;

  cell_y = floor(rem(y, channel_width)/7);
  cell_x = floor(rem(x, channel_width)/7);

  % flip because of nasty upside down image issue
  cell_y = 5 - cell_y;

  feature_index = channel*36 + cell_y*6 + cell_x + 1;

  if (channel < 0 || channel > 255)
    channel = nan;
  end
  if isscalar(ch)
    return;
  end
end

% ------------------------------------------------------------------------
function [channel_width] = display_feature_grid(fig)
% ------------------------------------------------------------------------
figure(fig);
clf;

cell_size = 5;
cell_pad = 1;
feature_map_glyph = 128*ones(cell_size+cell_pad*2, 'uint8');
feature_map_glyph(cell_pad+1:cell_pad+cell_size, ...
                  cell_pad+1:cell_pad+cell_size) = 0;
feature_map_glyph = repmat(feature_map_glyph, [6 6 3]);
feature_map_glyph(1,:,[1 3]) = 0;
feature_map_glyph(end,:,[1 3]) = 0;
feature_map_glyph(:,1,[1 3]) = 0;
feature_map_glyph(:,end,[1 3]) = 0;
feature_map_glyph(1,:,2) = 128;
feature_map_glyph(end,:,2) = 128;
feature_map_glyph(:,1,2) = 128;
feature_map_glyph(:,end,2) = 128;
channel_width = size(feature_map_glyph, 1);

array = repmat(feature_map_glyph, [1 1 1 256]);
im = my_montage(array, [16 16]);
imagesc(im);
axis image;


% ------------------------------------------------------------------------
function im = my_montage(ims, sz)
% ------------------------------------------------------------------------
ims_sz = [size(ims, 1) size(ims, 2)];
im = zeros(ims_sz(1)*sz(1), ims_sz(2)*sz(2), 3, class(ims));
k = 1;
for y = 0:sz(1)-1
  for x = 0:sz(2)-1
    im(y*ims_sz(1)+1:(y+1)*ims_sz(1), ...
       x*ims_sz(2)+1:(x+1)*ims_sz(2), :) = ims(:,:,:,k);
    k = k + 1;
  end
end
