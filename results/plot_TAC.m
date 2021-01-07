%%
clc
clear
set(0,'DefaultLineMarkerSize',28);
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesFontSize',14)
set(0, 'DefaultLineLineWidth', 2.0);

%%
washout = 400;
path = './results/TAC_2017/';


train = {};
test = {};

N = [4, 10, 20, 50, 100];
Q = [10, 20, 50, 100, 200];

for i = 1:length(N)
    for j = 1:length(Q)
        n = N(i);
        q = Q(j);

        str = sprintf('RobustRnn_w%dq%d_gamma0.0_*.mat', n, q);
        files = dir(strcat(path, str));
        train_nse = [];
        test_nse= [];

        for ii =1:length(files)
            file_path = strcat(path, files(ii).name);
            Dii = load(file_path); 
            train_nse = [train_nse; calc_nse(Dii.training, washout)];
            test_nse = [test_nse; calc_nse(Dii.test, washout)];

        end
% 
%         train = [train, train_nse];
%         test = [test, test_nse];
        train{i, j} = train_nse;
        test{i, j} = test_nse;
    end
end
%% Meshgrid
[nmesh, qmesh] = meshgrid(N, Q)

fig_pos = [-0 -0 600 400];
fig = figure('Position', fig_pos);
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on

surf(nmesh, qmesh, cellfun(@(x) median(x), test))


fig_pos = [-0 -0 600 400];
fig = figure('Position', fig_pos);
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on
surf(nmesh, qmesh, contour(cellfun(@(x) median(x), test)))


%%
labels = {'4', '10', '20', '50', '100', '200'}


% Plot Training data
fig_pos = [-0 -0 600 400];
fig = figure('Position', fig_pos);
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on

boxplot(train, labels);
title('Train NSE');
ax = gca
ax.YScale = 'log';

ylabel('NSE');
xlabel('n');

% ylim([3E-3, 1])
print(fig, '-dpdf', 'training_axes', '-bestfit');


% Plot Testing data
fig_pos = [-0 -0 600 400];
fig = figure('Position', fig_pos);
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
grid on
box on

boxplot(test, labels)
title('Test NSE')
ax = gca
ax.YScale = 'log';
ylabel('NSE');
xlabel('n');

% ylim([1E-2, 1])
print(fig, '-dpdf', 'testing_axes', '-bestfit');

function nse = calc_nse(data, washout)
    ytilde = squeeze(data.outputs{1});
    y = squeeze(data.measured{1});
    mu = mean(ytilde);
    nse = norm(ytilde(washout:end) - y(washout:end)) ^2 / norm(ytilde(washout:end)) ^ 2;
end