clc
clear

amp3 = load('./estimates_u3.mat');
amp8 = load('./estimates_u8.mat');
amp10 = load('./estimates_u10.mat');

% amplitude 3 inpuit
range = [600, 800]
fig = plot_results(amp3, 2, range)
% axis([1, 200, -5, 5])
print(fig, '-dpdf', './example_amp3', '-bestfit');

fig = plot_error(amp3, 2, range)
axis([1, 200, -5, 5])
print(fig, '-dpdf', './error_amp3', '-bestfit');

% amplitude 10 input
range = [780, 980]
fig = plot_results(amp10, 1, range)
print(fig, '-dpdf', './example_amp10', '-bestfit');

fig = plot_error(amp10, 1, range)
axis([1, 200, -5, 5])
print(fig, '-dpdf', './error_amp10', '-bestfit');

function fig = plot_results(data, idx, range)
    fig_pos = [-1600 79 901 674];
    fig = figure('Position', fig_pos);
    hold on

    p1 = plot(squeeze(data.Y(idx, 1, range(1):range(2))), 'Color', [0, 0, 0, 0.4], 'Linewidth', 4)
    
    p2 = plot(squeeze(data.lstm(idx, 1,range(1):range(2))), '-.', 'Linewidth', 2.0, 'Color', my_colours(4))
    p3 = plot(squeeze(data.rnn(idx, 1, range(1):range(2))), ':', 'Linewidth', 2.0, 'Color', my_colours(2))
    p4 = plot(squeeze(data.g00(idx, 1, range(1):range(2))), 'Linewidth', 2.0, 'Color', my_colours(1))
%     p5 = plot(squeeze(data.g8(idx, 1, :)), 'Linewidth', 1.5)

    legend([p1, p4, p2, p3], {'Measured', 'Robust RNN ($\Theta_*$)', 'LSTM', 'RNN'}, ...
           'Location', 'NorthWest', 'FontSize', 20)

    axis([0, 200, -10, 10])
    grid on
    box on
    fig.PaperPositionMode = 'auto';
    fig.PaperOrientation = 'landscape';
    
    xlabel('Samples')
    ylabel('Position')
end

function fig = plot_error(data, idx, range)
    fig_pos = [-1600 417 901 336];
    fig = figure('Position', fig_pos);
    hold on

    true = squeeze(data.Y(idx, 1, range(1):range(2)))
    p2 = plot(true - squeeze(data.lstm(idx, 1,range(1):range(2))), '-.', 'Linewidth', 2.0, 'Color', my_colours(4))
    p3 = plot(true - squeeze(data.rnn(idx, 1, range(1):range(2))), ':', 'Linewidth', 2.0, 'Color', my_colours(2))
    p4 = plot(true - squeeze(data.g00(idx, 1, range(1):range(2))), 'Linewidth', 2.0, 'Color', my_colours(1))
%     p5 = plot(squeeze(data.g8(idx, 1, :)), 'Linewidth', 1.5)

%     legend([p4, p2, p3], {'Robust RNN ($\Theta_*$)', 'LSTM', 'RNN'}, ...
%            'Location', 'NorthWest', 'FontSize', 20)

    axis([0, 200, -10, 10])
    grid on
    box on
    fig.PaperPositionMode = 'auto';
    fig.PaperOrientation = 'landscape';
    
    xlabel('Samples')
    ylabel('Error')
end