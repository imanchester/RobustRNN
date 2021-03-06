clc
clear

set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesFontSize',28)
set(0, 'DefaultLineLineWidth', 1.5);

fig_pos = [-0 -0 993 765];
ms = 20

%% Plot training time
figure
hold on
ax = gca
ax.YScale = 'log'

% load('.\msd\dnb_w10_gamma0.0_n4.mat')
% plot(training_log.val)
% 
% load('.\msd\cirnn_w10_gamma0.0_n4.mat')
% plot(training_log.val)

% load('./msd/rnn_w10_gamma0.0_n4.mat')
% p1 = plot(training_log.val, 'Color', my_colours(2))
% 
% load('./msd/lstm_w10_gamma0.0_n4.mat')
% p2 = plot(training_log.val, 'Color', my_colours(4))
% 
% load('./msd/iqc-rnn_w10_gamma0.0_n4.mat')
% p3 = plot(training_log.val, 'Color', my_colours(1))
% 
% labels = {"RNN", "LSTM", 'Robust RNN'}
% 
% legend([p1, p2, p3], labels)
% 
% axis([0, 170, 4E-2, 1])
% 
% xlabel('Epochs')
% ylabel('NSE')
% grid on
% box on

%% Load Lipschitz constants
cd ./msd/lip
cirnn_lip = load('./lip_cirnn_w10_gamma0.0_n4.mat').gamma;
srnn_lip = load('./lip_dnb_w10_gamma0.0_n4.mat').gamma;
iqc_rnn_00_lip = load('./lip_iqc-rnn_w10_gamma0.0_n4.mat').gamma;
iqc_rnn_3_lip = load('./lip_iqc-rnn_w10_gamma3.0_n4.mat').gamma;
iqc_rnn_6_lip = load('./lip_iqc-rnn_w10_gamma6.0_n4.mat').gamma;
iqc_rnn_8_lip = load('./lip_iqc-rnn_w10_gamma8.0_n4.mat').gamma;
iqc_rnn_10_lip = load('./lip_iqc-rnn_w10_gamma10.0_n4.mat').gamma;
iqc_rnn_12_lip = load('./lip_iqc-rnn_w10_gamma12.0_n4.mat').gamma;
lstm_lip = load('./lip_lstm_w10_gamma0.0_n4.mat').gamma;
rnn_lip = load('./lip_rnn_w10_gamma0.0_n4.mat').gamma;

cd ..
% Load Lipschitz constants
cirnn = load('./cirnn_w10_gamma0.0_n4.mat');
srnn = load('./dnb_w10_gamma0.0_n4.mat');
iqc_rnn_00 = load('./iqc-rnn_w10_gamma0.0_n4.mat');
iqc_rnn_3 = load('./iqc-rnn_w10_gamma3.0_n4.mat');
iqc_rnn_6 = load('./iqc-rnn_w10_gamma6.0_n4.mat');
iqc_rnn_8 = load('./iqc-rnn_w10_gamma8.0_n4.mat');
iqc_rnn_10 = load('./iqc-rnn_w10_gamma10.0_n4.mat');
% iqc_rnn_12 = load('./iqc-rnn_w10_gamma12.0_n4.mat');
lstm = load('./lstm_w10_gamma0.0_n4.mat');
rnn = load('./rnn_w10_gamma0.0_n4.mat');

cd ..
%% Plot NSE versus Lipschitz constant
fig = figure('Position', fig_pos)
hold on
% plot bounded models
p1 = plot(iqc_rnn_3_lip, iqc_rnn_3.test.NSE(3), 'x', 'color', my_colours(1), 'MarkerSize', ms, 'LineWidth', 2);
plot([3, 3], [1E-5, 10], '--', 'color', my_colours(1))
p2 = plot(iqc_rnn_6_lip, iqc_rnn_6.test.NSE(3), 'x', 'color', my_colours(2), 'MarkerSize', ms, 'LineWidth', 2);
plot([6, 6], [1E-5, 10], '--', 'color', my_colours(2))
p3 = plot(iqc_rnn_8_lip, iqc_rnn_8.test.NSE(3), 'x', 'color', my_colours(4), 'MarkerSize', ms, 'LineWidth', 2);
plot([8, 8], [1E-5, 10], '--', 'color', my_colours(4))
% p4 = plot(iqc_rnn_10_lip, iqc_rnn_10.test.NSE(3), 'x', 'color', my_colours(4), 'MarkerSize', ms);
% plot([10, 10], [1E-5, 10], '-.', 'color', my_colours(4))
p5 = plot(iqc_rnn_00_lip, iqc_rnn_00.test.NSE(3), 'x', 'color', my_colours(5), 'MarkerSize', ms, 'LineWidth', 2);

% plot stable models
p6 = plot(cirnn_lip, cirnn.test.NSE(3), 's', 'color', my_colours(6), 'MarkerSize', ms, 'LineWidth', 2);
p7 = plot(srnn_lip, srnn.test.NSE(3), 's', 'color', my_colours(7), 'MarkerSize', ms, 'LineWidth', 2);

%lstm and rnn
p8 = plot(rnn_lip, rnn.test.NSE(3), 'b^', 'MarkerSize', ms, 'LineWidth', 2);
p9 = plot(lstm_lip, lstm.test.NSE(3), 'r^', 'MarkerSize', ms, 'LineWidth', 2);
% 
ax = gca
ax.XScale = 'log'
ax.YScale = 'log'

box on
grid on
ylabel('NSE', 'FontSize', 30)
xlabel('$\hat{\gamma}$', 'FontSize', 40)
% ax.YScale = 'log'

leg = legend([p1(1), p2(1), p3(1), p5(1), p6(1), p7(1), p8, p9],...
       {'Robust RNN (${\Theta_3}$)', 'Robust RNN ($\Theta_6$)','Robust RNN ($\Theta_8$)',...
        'Robust RNN ($\Theta_*$)','ci-RNN',...
        's-RNN', 'RNN', 'LSTM'})
leg.FontSize = 18

axis([2, 1000, 0.04, 0.3])
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
name = strcat('msd_nse_vs_gamma.pdf');
print(fig, '-dpdf', name, '-bestfit');


%% Plot generalization results for amplitude
set(0,'defaultAxesFontSize',22)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultTextInterpreter','latex');

cirnn = load('./results/msd/generalization/amp_cirnn_w10_gamma0.0_n4.mat');
srnn = load('./results/msd/generalization/amp_dnb_w10_gamma0.0_n4.mat');

lstm = load('./results/msd/generalization/amp_lstm_w10_gamma0.0_n4.mat');
rnn = load('./results/msd/generalization/amp_rnn_w10_gamma0.0_n4.mat');

g00 = load('./results/msd/generalization/amp_iqc-rnn_w10_gamma0.0_n4.mat');
g3 = load('./results/msd/generalization/amp_iqc-rnn_w10_gamma3.0_n4.mat');
g6 = load('./results/msd/generalization/amp_iqc-rnn_w10_gamma6.0_n4.mat');
g8 = load('./results/msd/generalization/amp_iqc-rnn_w10_gamma8.0_n4.mat');
g10 = load('./results/msd/generalization/amp_iqc-rnn_w10_gamma10.0_n4.mat');

%%
labels = {'', '1.0', '', '2.0', '', '3.0', '', '4.0', '', '5.0',...
          '', '6.0', '', '7.0', '', '8.0', '', '9.0', '', '10.0', ''}

fig = format_boxplots(g8, labels);
%%
title('rnn');
name = strcat('./results/msd/generalization/amp_rnn.pdf');
print(fig, '-dpdf', name, '-bestfit');


fig = format_boxplots(lstm, labels);
title('lstm');
name = strcat('./results/msd/generalization/amp_lstm.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g00, labels);
title('robust-RNN $\gamma=\infty$');
name = strcat('./results/msd/generalization/amp_ri-rnn_ginf.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g3, labels);
title('robust-RNN $\gamma=3$');
name = strcat('./results/msd/generalization/amp_ri-rnn_g3.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g6, labels);
title('robust-RNN $\gamma=6$');
name = strcat('./results/msd/generalization/amp_ri-rnn_g6.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g8, labels);
title('robust-RNN $\gamma=8$');
name = strcat('./results/msd/generalization/amp_ri-rnn_g8.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(cirnn, labels);
title('cirnn');
name = strcat('./results/msd/generalization/amp_cirnn.pdf');
print(fig, '-dpdf', name, '-bestfit');


%% One final plot showing the median of each
set(0, 'DefaultLineLineWidth', 1.5);
fig = figure('Position', [-1600 -12 993 765])
hold on

% p1 = plot(g3.amps, median(g3.NSE'))
% p2 = plot(mean(g6.NSE'))
% p3 = plot(g6.amps, median(g6.NSE'))
% p4 = plot(mean(g10.NSE'))
% p3 = plot(g8.amps, median(g8.NSE'))
% p4 = plot(mean(g10.NSE'))


p1 = plot(g3.amps, median(rnn.NSE'), '-.')
p2 = plot(g3.amps, median(lstm.NSE'), '-.')
p3 = plot(g3.amps, median(srnn.NSE'))
p4 = plot(g3.amps, median(cirnn.NSE'))
p5 = plot(g3.amps, median(g00.NSE'))

grid on
box on
axis([0.5, 10.5, 0, 0.8])

ylabel('NSE')
xlabel('$\sigma_u$')

leg = legend([p1, p2, p3, p4, p5],...
       {'RNN', 'LSTM', 's-RNN',...
        'ci-RNN','Robust RNN ($\Theta_*$)'}, 'Location', 'NorthWest')
    
    
    
name = strcat('./results/msd/generalization/median_comparison.pdf');
print(fig, '-dpdf', name, '-bestfit');
%% Plot generalization results for Period
cirnn = load('./results/msd/generalization/per_cirnn_w10_gamma0.0_n4.mat');
srnn = load('./results/msd/generalization/per_dnb_w10_gamma0.0_n4.mat');

lstm = load('./results/msd/generalization/per_lstm_w10_gamma0.0_n4.mat');
rnn = load('./results/msd/generalization/per_rnn_w10_gamma0.0_n4.mat');

g00 = load('./results/msd/generalization/per_iqc-rnn_w10_gamma0.0_n4.mat');
g3 = load('./results/msd/generalization/per_iqc-rnn_w10_gamma3.0_n4.mat');
g6 = load('./results/msd/generalization/per_iqc-rnn_w10_gamma6.0_n4.mat');
g8 = load('./results/msd/generalization/per_iqc-rnn_w10_gamma8.0_n4.mat');
g10 = load('./results/msd/generalization/per_iqc-rnn_w10_gamma10.0_n4.mat');

set(0, 'DefaultLineLineWidth', 1.5);
set(0,'defaultAxesFontSize',14)

fig = format_boxplots(rnn, rnn.period);
title('rnn');
xlabel('Period');
name = strcat('./results/msd/generalization/per_rnn.pdf');
print(fig, '-dpdf', name, '-bestfit');


fig = format_boxplots(lstm, lstm.period);
title('lstm');
xlabel('Period');
name = strcat('./results/msd/generalization/per_lstm.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g00, g00.period);
title('robust-RNN $\gamma=\infty$');
xlabel('Period');
name = strcat('./results/msd/generalization/per_ri-rnn_ginf.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g3, g3.period);
title('robust-RNN $\gamma=3$');
xlabel('Period');
name = strcat('./results/msd/generalization/per_ri-rnn_g3.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g6, g6.period);
title('robust-RNN $\gamma=6$');
xlabel('Period');
name = strcat('./results/msd/generalization/per_ri-rnn_g6.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(g8, g8.period);
title('robust-RNN $\gamma=8$');
xlabel('Period');
name = strcat('./results/msd/generalization/per_ri-rnn_g8.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_boxplots(cirnn, cirnn.period);
title('cirnn');
xlabel('Period');
name = strcat('./results/msd/generalization/per_cirnn.pdf');
print(fig, '-dpdf', name, '-bestfit');


%% Boxplots showing sensitivity for Period

cirnn = load('./results/msd/sensitivity/per_cirnn_w10_gamma0.0_n4.mat');
srnn = load('./results/msd/sensitivity/per_dnb_w10_gamma0.0_n4.mat');

lstm = load('./results/msd/sensitivity/per_lstm_w10_gamma0.0_n4.mat');
rnn = load('./results/msd/sensitivity/per_rnn_w10_gamma0.0_n4.mat');

g00 = load('./results/msd/sensitivity/per_iqc-rnn_w10_gamma0.0_n4.mat');
g3 = load('./results/msd/sensitivity/per_iqc-rnn_w10_gamma3.0_n4.mat');
g6 = load('./results/msd/sensitivity/per_iqc-rnn_w10_gamma6.0_n4.mat');
g8 = load('./results/msd/sensitivity/per_iqc-rnn_w10_gamma8.0_n4.mat');
g10 = load('./results/msd/sensitivity/per_iqc-rnn_w10_gamma10.0_n4.mat');

fig = format_sensitivity_boxplots(rnn, rnn.period);
xlabel('Period')
title('rnn');
name = strcat('./results/msd/sensitivity/per_rnn.pdf');
print(fig, '-dpdf', name, '-bestfit');


fig = format_sensitivity_boxplots(lstm, lstm.period);
xlabel('Period')
title('lstm');
name = strcat('./results/msd/sensitivity/per_lstm.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g00, g00.period);
xlabel('Period')
title('robust-RNN $\gamma=\infty$');
name = strcat('./results/msd/sensitivity/per_ri-rnn_ginf.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g3, g3.period);
xlabel('Period')
title('robust-RNN $\gamma=3$');
name = strcat('./results/msd/sensitivity/per_ri-rnn_g3.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g6, g6.period);
title('robust-RNN $\gamma=6$');
name = strcat('./results/msd/sensitivity/per_ri-rnn_g6.pdf');
xlabel('Period')
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g8, g8.period);
xlabel('Period')
title('robust-RNN $\gamma=8$');
name = strcat('./results/msd/sensitivity/per_ri-rnn_g8.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g10, g10.period);
xlabel('Period')
title('robust-RNN $\gamma=10$');
name = strcat('./results/msd/sensitivity/per_ri-rnn_g10.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(cirnn, cirnn.period);
xlabel('Period')
title('cirnn');
name = strcat('./results/msd/sensitivity/per_cirnn.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(srnn, srnn.period);
xlabel('Period')
title('srnn');
name = strcat('./results/msd/sensitivity/per_srnn.pdf');
print(fig, '-dpdf', name, '-bestfit');

%% Boxplots showing sensitivity for Amplitude

cirnn = load('./results/msd/sensitivity/amp_cirnn_w10_gamma0.0_n4.mat');
srnn = load('./results/msd/sensitivity/amp_dnb_w10_gamma0.0_n4.mat');

lstm = load('./results/msd/sensitivity/amp_lstm_w10_gamma0.0_n4.mat');
rnn = load('./results/msd/sensitivity/amp_rnn_w10_gamma0.0_n4.mat');

g00 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma0.0_n4.mat');
g3 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma3.0_n4.mat');
g6 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma6.0_n4.mat');
g8 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma8.0_n4.mat');
g10 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma10.0_n4.mat');

fig = format_sensitivity_boxplots(rnn, rnn.amps);
xlabel('Amplitude');
title('rnn');
name = strcat('./results/msd/sensitivity/amp_rnn.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(lstm, lstm.amps);
xlabel('Amplitude');
title('lstm');
name = strcat('./results/msd/sensitivity/amp_lstm.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g00, g00.amps);
xlabel('Amplitude');
title('robust-RNN $\gamma=\infty$');
name = strcat('./results/msd/sensitivity/amp_ri-rnn_ginf.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g3, g3.amps);
xlabel('Amplitude');
title('robust-RNN $\gamma=3$');
name = strcat('./results/msd/sensitivity/amp_ri-rnn_g3.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g6, g6.amps);
xlabel('Amplitude');
title('robust-RNN $\gamma=6$');
name = strcat('./results/msd/sensitivity/amp_ri-rnn_g6.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g8, g8.amps);
xlabel('Amplitude');
title('robust-RNN $\gamma=8$');
name = strcat('./results/msd/sensitivity/amp_ri-rnn_g8.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(g10, g10.amps);
xlabel('Amplitude');
title('robust-RNN $\gamma=10$');
name = strcat('./results/msd/sensitivity/amp_ri-rnn_g10.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(cirnn, cirnn.amps);
xlabel('Amplitude');
title('cirnn');
name = strcat('./results/msd/sensitivity/amp_cirnn.pdf');
print(fig, '-dpdf', name, '-bestfit');

fig = format_sensitivity_boxplots(srnn, srnn.amps);
xlabel('Amplitude');
title('srnn');
name = strcat('./results/msd/sensitivity/srnn.pdf');
print(fig, '-dpdf', name, '-bestfit');

%% Boxplots showing sensitivity for various model sets around training data
cirnn = load('./results/msd/sensitivity/amp_cirnn_w10_gamma0.0_n4.mat').Sensitivity(6,:);
srnn = load('./results/msd/sensitivity/amp_dnb_w10_gamma0.0_n4.mat').Sensitivity(6,:);

lstm = load('./results/msd/sensitivity/amp_lstm_w10_gamma0.0_n4.mat').Sensitivity(6,:);
rnn = load('./results/msd/sensitivity/amp_rnn_w10_gamma0.0_n4.mat').Sensitivity(6,:);

g00 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma0.0_n4.mat').Sensitivity(6,:);
g3 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma3.0_n4.mat').Sensitivity(6,:);
g6 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma6.0_n4.mat').Sensitivity(6,:);
g8 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma8.0_n4.mat').Sensitivity(6,:);
g10 = load('./results/msd/sensitivity/amp_iqc-rnn_w10_gamma10.0_n4.mat').Sensitivity(6,:);

labels = {'$\Theta_3$', '$\Theta_6$', '\Theta_8', '\Theta_\infty', '\Theta_\infty', 'srnn', 'lstm', 'rnn'}
data = cat(1, g3, g6, g8, g10, g00, cirnn, srnn, lstm, rnn)
boxplot(data')

%% Plots showing NSE and sensitivity
set(0, 'DefaultLineLineWidth', 1.5);
cirnn = load('./results/msd/training_stats/cirnn_w10_gamma0.0_n4.mat');
srnn = load('./results/msd/training_stats/dnb_w10_gamma0.0_n4.mat');

lstm = load('./results/msd/training_stats/lstm_w10_gamma0.0_n4.mat');
rnn = load('./results/msd/training_stats/rnn_w10_gamma0.0_n4.mat');

g00 = load('./results/msd/training_stats/iqc-rnn_w10_gamma0.0_n4.mat');
g3 = load('./results/msd/training_stats/iqc-rnn_w10_gamma3.0_n4.mat');
g6 = load('./results/msd/training_stats/iqc-rnn_w10_gamma6.0_n4.mat');
g8 = load('./results/msd/training_stats/iqc-rnn_w10_gamma8.0_n4.mat');
g10 = load('./results/msd/training_stats/iqc-rnn_w10_gamma10.0_n4.mat');

ms = 15
npoints = 10
fig_pos = [-1600 -12 993 765];
fig = figure('Position', fig_pos)
hold on
% plot bounded models
p1 = plot(g3.Sensitivity(1:npoints), g3.NSE(1:npoints), 'x', 'color', my_colours(1), 'MarkerSize', ms);
p2 = plot(g6.Sensitivity(1:npoints), g6.NSE(1:npoints), 'x', 'color', my_colours(2), 'MarkerSize', ms);
p3 = plot(g8.Sensitivity(1:npoints), g8.NSE(1:npoints), 'x', 'color', my_colours(3), 'MarkerSize', ms);
p4 = plot(g10.Sensitivity(1:npoints), g10.NSE(1:npoints), 'x', 'color', my_colours(4), 'MarkerSize', ms);
p5 = plot(g00.Sensitivity(1:npoints), g00.NSE(1:npoints), 'x', 'color', my_colours(5), 'MarkerSize', ms);

% plot stable models
p6 = plot(cirnn.Sensitivity(1:npoints), cirnn.NSE(1:npoints), 's', 'color', my_colours(6), 'MarkerSize', ms);
p7 = plot(srnn.Sensitivity(1:npoints), srnn.NSE(1:npoints), 's', 'color', my_colours(7), 'MarkerSize', ms);

%lstm and rnn
p8 = plot(rnn.Sensitivity(1:npoints), rnn.NSE(1:npoints), 'b^', 'MarkerSize', ms);
p9 = plot(lstm.Sensitivity(1:npoints), lstm.NSE(1:npoints), 'r^', 'MarkerSize', ms);
% 
ax = gca
% ax.XScale = 'log'
% ax.YScale = 'log'

box on
grid on
ylabel('NSE')
xlabel('$\hat{\gamma}$')
% ax.YScale = 'log'

leg = legend([p1(1), p2(1), p3(1), p4(1), p5(1), p6(1), p7(1), p8, p9],...
       {'Robust RNN ${\gamma=3}$', 'Robust RNN $\gamma=6$','Robust RNN $\gamma=8$',...
        'Robust RNN $\gamma=10$','Robust RNN $\gamma=\infty$','ci-RNN',...
        's-RNN', 'RNN', 'LSTM'})
leg.FontSize = 18

axis([2, 10, 0.01, 0.3])
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
name = strcat('msd_nse_vs_gamma.pdf');
print(fig, '-dpdf', name, '-bestfit');

function fig = format_boxplots(data, labels)
    fig_pos = [0 0 901 674];
    fig = figure('Position', fig_pos);
    boxplot(data.NSE', 'Labels', labels);
    xlabel('$\sigma_u$', 'FontSize', 40);
    ylabel('NSE', 'FontSize', 30);
    box on
    grid on
    ylim([0, 1]);
    fig.PaperPositionMode = 'auto';
    fig.PaperOrientation = 'landscape';
end

function fig = format_sensitivity_boxplots(data, labels)
    fig_pos = [-1600 -12 993 765];
    fig = figure('Position', fig_pos);
    boxplot(data.Sensitivity', 'Labels', labels);
    ylabel('Sensitivity')
    box on
    grid on
    ylim([0, 13]);
    fig.PaperPositionMode = 'auto';
    fig.PaperOrientation = 'landscape';
end