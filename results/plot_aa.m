clc
clear

set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesFontSize',28)
set(0, 'DefaultLineLineWidth', 1.0);

fig_pos = [1, 600, 1100, 900];

%% Load adversarial results for chen system
depth = 1
cd('chen/aa')

g16 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma16.0.mat')));
g17 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma17.0.mat')));
g18 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma18.0.mat')));
g19 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma19.0.mat')));
g20 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma20.0.mat')));
g21 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma21.0.mat')));
g22 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma22.0.mat')));


g00 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma0.0.mat')));
rnn = cell2mat(load_file_list(dir('./aa_rnn_w10_gamma0.0.mat')));
lstm = cell2mat(load_file_list(dir('./aa_lstm_w10_gamma0.0.mat')));
cirnn = cell2mat(load_file_list(dir('./aa_cirnn_w10_gamma0.0.mat')));

cd('../..')

%% Plot adversarial examples
fig = figure('Position', fig_pos)
hold on

p1 = plot(cellfun(@(x) norm(x), g16.du), g16.SE)
p2 = plot(cellfun(@(x) norm(x), g18.du), g18.SE)
% p1 = plot(cellfun(@(x) norm(x), g19.du), g19.SE)
p3 = plot(cellfun(@(x) norm(x), g20.du), g20.SE)
% p1 = plot(cellfun(@(x) norm(x), g21.du), g21.SE)
% p4 = plot(cellfun(@(x) norm(x), g22.du), g22.SE)

p4 = plot(cellfun(@(x) norm(x), g00.du), g00.SE, 'k-s')
p5 = plot(cellfun(@(x) norm(x), cirnn.du), cirnn.SE, 'k-d')
p6 = plot(cellfun(@(x) norm(x), rnn.du), rnn.SE, 'k->')
p7 = plot(cellfun(@(x) norm(x), lstm.du), lstm.SE, 'k-*')


% calculate maximum gradient of each line

g16_grad = get_max_grad(g16);
g18_grad = get_max_grad(g18);
g20_grad = get_max_grad(g20);
g22_grad = get_max_grad(g22);
g00_grad = get_max_grad(g00);
lstm_grad = get_max_grad(lstm);
rnn_grad = get_max_grad(rnn);
cirnn_grad = get_max_grad(cirnn);

box on 
grid on
legend([p1, p2, p3, p4, p5, p6, p7],...
        {'$\gamma=16$, dJ = 13.9 ','$\gamma=18$, dJ = 15.4',...
        '$\gamma=20$, dJ = 17.8', '$\gamma=\inf$, dJ = 20.9',...
        'cirnn, dJ = 21.6', ...
        'rnn, dJ = 28.6', 'lstm, dJ=23.3'},...
        'Location', 'southeast')

% axis([0, 0.1, 0.2, 0.8])
ylabel('SE')
xlabel('$||\delta_u||$')

% print figure
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
name = strcat('test_aa_iqc_chen.pdf');
print(fig, '-dpdf', name, '-bestfit');


%% Load adversarial results for msd system
depth = 1
cd('msd/aa')

g1 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma1.0_n4.mat')));
g1p25 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma1.25_n4.mat')));
g1p5 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma1.5_n4.mat')));
g2 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma2.0_n4.mat')));
% g17 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma17.0.mat')));
% g18 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma18.0.mat')));
% g19 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma19.0.mat')));
% g20 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma20.0.mat')));
% g21 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma21.0.mat')));
% g22 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma22.0.mat')));


g00 = cell2mat(load_file_list(dir('./aa_iqc-rnn_w10_gamma0.0_n4.mat')));
rnn = cell2mat(load_file_list(dir('./aa_rnn_w10_gamma0.0_n4.mat')));
lstm = cell2mat(load_file_list(dir('./aa_lstm_w10_gamma0.0_n4.mat')));
cirnn = cell2mat(load_file_list(dir('./aa_cirnn_w10_gamma0.0_n4.mat')));

cd('../..')

%% Plot adversarial examples
fig = figure('Position', fig_pos)
hold on

% p1 = plot(cellfun(@(x) norm(x), g16.du), g16.SE)
% p2 = plot(cellfun(@(x) norm(x), g18.du), g18.SE)
% p1 = plot(cellfun(@(x) norm(x), g19.du), g19.SE)
p1 = plot(cellfun(@(x) norm(x), g1.du), g1.SE)
p2 = plot(cellfun(@(x) norm(x), g1p25.du), g1p25.SE)
% p3 = plot(cellfun(@(x) norm(x), g1p5.du), g1p5.SE)
p4 = plot(cellfun(@(x) norm(x), g2.du), g2.SE)
% p1 = plot(cellfun(@(x) norm(x), g21.du), g21.SE)
% p4 = plot(cellfun(@(x) norm(x), g22.du), g22.SE)

p5 = plot(cellfun(@(x) norm(x), g00.du), g00.SE, 'k-s')
p6 = plot(cellfun(@(x) norm(x), rnn.du), rnn.SE, 'k->')
p7 = plot(cellfun(@(x) norm(x), lstm.du), lstm.SE, 'k-*')
p8 = plot(cellfun(@(x) norm(x), cirnn.du), cirnn.SE, 'k-d')

% calculate maximum gradient of each line

g1_grad = get_max_grad(g1);
g1p25_grad = get_max_grad(g1p25);
g1p5_grad = get_max_grad(g1p5);
g2_grad = get_max_grad(g2);
% g20_grad = get_max_grad(g20);
g00_grad = get_max_grad(g00);
lstm_grad = get_max_grad(lstm);
rnn_grad = get_max_grad(rnn);
cirnn_grad = get_max_grad(cirnn);

box on 
grid on
legend([p1, p2, p4, p5, p6, p7, p8],...
        {'$\gamma=1$, dJ = 0.90 ','$\gamma=1.25$, dJ = 1.14',...
        '$\gamma=2.0, dJ = 1.84$', ...
        '$\gamma=\inf$, dJ = 1.44',...
        'rnn, dJ = 1.72', 'lstm, dJ=3.01', 'cirnn, dJ=1.42'},...
        'Location', 'southeast')

% axis([0, 0.1, 0.2, 0.8])
ylabel('SE')
xlabel('$||\delta_u||$')

% print figure
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
name = strcat('test_aa_iqc_msd.pdf');
print(fig, '-dpdf', name, '-bestfit');

%% Compare two trajectories
fig = figure('Position', fig_pos)
subplot(2, 1, 1)
hold on
plot(squeeze(cirnn{1}.test.measured{1}(1, 1, :))', 'k')
plot(squeeze(cirnn{1}.test.outputs{1}(1, 1, :))', 'Color', my_colours(2))
box on 
grid on

subplot(2, 1, 2)
hold on
plot(squeeze(cirnn{1}.test.measured{1}(1, 1, :))', 'k')
plot(squeeze(lstm{1}.test.outputs{1}(1, 1, :))', 'Color', my_colours(4))
box on 
grid on

%%
fig = figure('Position', fig_pos); hold on
gamma = [0.5, 1, 2,3, 5,10, 20];

l2gb = [g0p5{1}, g1{1}, g2{1}, g3{1} ,g5{1} , g10{1}, g20{1}];

extract_nse = @(x) mean(mean(x.test.NSE))
plot(gamma, arrayfun(extract_nse, l2gb))

plot([0, 20], extract_nse(cirnn{1})*[1, 1])
plot([0, 20], extract_nse(lstm{1})*[1, 1])
plot([0, 20], extract_nse(rnn{1})*[1, 1])
%%
fig = figure('Position', fig_pos)
hold on
p1 = plot_NSE_vs_epsilon_regions(8, g0p5_aa, my_colours(1), '-');
p2 = plot_NSE_vs_epsilon_regions(8, g1_aa, my_colours(2), '-');
p3 = plot_NSE_vs_epsilon_regions(8, g1p5_aa, my_colours(3), '-');
% p3 = plot_NSE_vs_epsilon_regions(2, g2p5_aa, my_colours(3), '-');
p4 = plot_NSE_vs_epsilon_regions(8, g5_aa, my_colours(4), '-');

p6 = plot_NSE_vs_epsilon_regions(8, lstm_aa, my_colours(7), '--');
p5 = plot_NSE_vs_epsilon_regions(8, c_aa, 'k', '-.');

box on 
grid on

xlabel('$\frac{||\Delta u||}{||u||}$')
ylabel('NSE')
legend([p1, p2, p3, p4, p5, p6], {'$\gamma = 0.5$', '$\gamma = 1.0$', '$\gamma = 1.5$', '$\gamma = 5.0$', 'C', 'LSTM'}, 'Location', 'Southeast')
% legend([p1, p2, p4, p5, p6], {'$\gamma = 0.5$', '$\gamma = 1.0$', '$\gamma = 5.0$', 'C', 'LSTM'}, 'Location', 'Southeast')
% 
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape'

ylim([0.2, 1.1])

name = strcat('aa_w64_l', int2str(depth), '.pdf')
% print(fig, '-dpdf', name, '-bestfit');


%% plot perfromance versus epsilon for each model
cmap = jet(9);
figure('Position', fig_pos);
hold on

% plot([g02.epsilon], [g02.NSE])

for ii = 1:length(g5)
    p1 = plot(g02(ii).epsilon, mean(g02(ii).SE, 2), '*', 'color', cmap(1, :));
    p2 = plot(g05(ii).epsilon, mean(g05(ii).SE, 2), '*', 'color', cmap(7, :));
    p3 = plot(g1(ii).epsilon, mean(g1(ii).SE, 2), '*', 'color', cmap(3, :));
    p4 = plot(g5(ii).epsilon, mean(g5(ii).SE, 2), '*', 'color', cmap(8, :));
%     p5 = plot(g10{ii}.epsilon, mean(g10{ii}.training.NSE, 2), '*', 'color', cmap(5, :));
%     p6 = plot(g50{ii}.epsilon, mean(g50{ii}.training.NSE, 2), '*', 'color', cmap(6, :));
%     p7 = plot(g100{ii}.epsilon, mean(g100{ii}.training.NSE, 2), '*', 'color', cmap(7, :));
%     p8 = plot(g500{ii}.epsilon, mean(g500{ii}.training.NSE, 2), '*', 'color', cmap(8, :));
%     p9 = plot(c{ii}.epsilon, mean(c{ii}.training.NSE, 2), '*', 'color', cmap(9, :));
%     p10 = plot(lstm{ii}.epsilon, mean(lstm{ii}.training.NSE, 2), 'k*');
end
% 
% legend([p1(1), p2(1), p3(1), p4(1), p5(1), p6(1), p7(1), p8(1), p9(1), p10(1)], ...
%         {'$\gamma = 0.2$', '$\gamma = 0.5$', '$\gamma = 1$', '$\gamma = 5$', ... 
%         '$\gamma = 10$', '$\gamma = 50$', '$\gamma = 100$', '$\gamma = 500$', '$C$', 'LSTM'}, 'Location', 'northwest')
    
grid on
box on
xlabel('$\epsilon$')
ylabel('NSE')


%% plot perfromance versus epsilon for each model
cmap = lines(9);
figure('Position', fig_pos);
hold on

mean_NSE =@(dat) mean(mean(dat.NSE))

g02_test = [g02.test];
g05_test = [g05.test];
g1_test = [g1.test];
g5_test = [g5.test];
g10_test = [g10.test];
c_test = [c.test];
lstm_test = [lstm.test];

p1 = plot([g02.epsilon], arrayfun(mean_NSE, g02_test))
p2 = plot([g05.epsilon], arrayfun(mean_NSE, g05_test))
p3 = plot([g1.epsilon], arrayfun(mean_NSE, g1_test))
p4 = plot([g5.epsilon], arrayfun(mean_NSE, g5_test))
p5 = plot([g10.epsilon], arrayfun(mean_NSE, g10_test))
p6 = plot([c.epsilon], arrayfun(mean_NSE, c_test))
p7 = plot([lstm.epsilon], arrayfun(mean_NSE, lstm_test))

% 
legend([p1(1), p2(1), p3(1), p4(1), p5(1), p6(1), p7(1)], ...
        {'$\gamma = 0.2$', '$\gamma = 0.5$', '$\gamma = 1$', '$\gamma = 5$', ... 
        '$\gamma = 10$', '$C$', 'LSTM'}, 'Location', 'northwest')

% legend([p1(1), p2(1), p3(1), p4(1), p5(1), p6(1), p7(1), p8(1), p9(1), p10(1)], ...
%         {'$\gamma = 0.2$', '$\gamma = 0.5$', '$\gamma = 1$', '$\gamma = 5$', ... 
%         '$\gamma = 10$', '$\gamma = 50$', '$\gamma = 100$', '$\gamma = 500$', '$C$', 'LSTM'}, 'Location', 'northwest')
grid on
box on
xlabel('$\epsilon$')
ylabel('NSE')
%%
function data = load_file_list(files)
    data = {};
    for ii = 1:length(files)
       data{ii} = load(files(ii).name);
       data{ii}.name = files(ii).name;
    end
end

function nse = nse_calc(IO_data)
    nse = zeros(length(IO_data.outputs), 1);
    for kk = 1:length(IO_data.outputs)
       y = squeeze(IO_data.outputs{kk});
       ytild = squeeze(IO_data.measured{kk});
       err = y - ytild;
       sig_mu = mean(ytild, 2)';
       nse(kk) = mean(rms(err')./rms(ytild' - sig_mu));
    end
    nse = mean(nse);
end

function [stable, unstable] = split_stable(nse, bound)
    stable = nse( nse <= bound);
    unstable = nse(nse > bound);
end

function [p1, p2] = format_boxplot(p1, p2)
    boxplot_fontsize = 16;
    height = 0.2;
    font_offset = 0.1;
    p2.Position = p2.Position + [0, font_offset, 0, -font_offset];
    
    gap = p1.Position(2) - p2.Position(2) - p2.Position(4);
    p1.Position = p1.Position + [0, +height, 0, -height];
    p2.Position = p2.Position + [0, 0, 0, +height + gap - 0.02];
    
    % turn of x ticks
    p1.XTick = [];
    p1.XLim = p2.XLim;
    % Format p1
    axes(p1); box on; grid on;
    p1.YLim = [0, 49];
    ylabel('\% Unstable', 'Interpreter', 'latex')
    
%     format p2
    axes(p2); box on; grid on;
    p2.YLim = [0.1, 0.49];
%     p2.YScale = 'log';
    p2.XTickLabelRotation = 45;
    p2.XAxis.TickLabelInterpreter = 'latex';
%     p2.XAxis.FontSize = boxplot_fontsize;
    ylabel('NSE', 'Interpreter', 'latex');
    
end

% Create a boxplot. 
function make_boxplot( unstable, NSE, Labels)
    classes = repmat(1:size(NSE, 2), size(NSE, 1), 1);
    boxplot(NSE(~unstable), classes(~unstable), 'Labels', Labels)
end

function format_bargraph(b)
   b.BarWidth = 0.5;
%    b.FaceColor = 'flat';
%    b.CData(:,1) = b.YData;
end

function C = sf(data)
    dt = diff(data, 1);
    ddt = diff(dt, 1);
    C =  norm(ddt);
end

function p = plot_NSE_vs_epsilon(aa_Data, color, LS)
    dat = cat(3, aa_Data.NSE);
    p = plot([aa_Data.epsilon], squeeze(mean(mean(dat))), 'color', color, 'linestyle', LS);
end

function p = plot_NSE_vs_epsilon_regions(n_sets, aa_Data, color, LS)
    alpha = 0.5
    
    dat = cat(3, aa_Data.NSE);
    nse = mean(mean(cat(3, aa_Data.NSE)));
    epsilon = [aa_Data.epsilon];
    
    nse = reshape(squeeze(nse), n_sets, []);
    epsilon = reshape(epsilon, n_sets, []);
    
    upper = max(nse);
    lower = min(nse);
    center = median(nse);
    
    region_x = [epsilon(1, :), epsilon(1, end:-1:1)]
    region_y = [upper, lower(end:-1:1)]
    
    patch = fill(region_x, region_y, color, 'FaceAlpha', alpha);
    p = plot(epsilon(1,:)', center, 'color', color, 'linestyle', LS);
    
end

function g = get_max_grad(data)
    max_grad = 0;
    for ii = 1:length(data.du)
        g = norm(squeeze(data.y{ii} - data.yp{ii})) / norm(data.du{ii});
        if g > max_grad
            max_grad = g;
        end
    end
end