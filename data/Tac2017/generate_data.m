
 
%% True system

% Mass - nonlinear spring - damper

% Dimensions
nx = 4;
ny = 1;
nu = 1;
nw = 1;

param.idt = 0.1; 

param.m = [0.5,0.1];        % Mass
param.c = [0.5,0.75];       % Damping
param.k = [2,1];            % Spring 
param.nl = [1,1];           % Spring nonlinearity type
param.slim = [1.25,1.25];   % Spring travel limits

nlsprng_models{1} = param;


%% Experiment parameters

% 1000 data points used in the TAC paper for both training and validation

num_datapoints_training = 1e3; % number of data points used for training
num_datapoints_validation = 1e3; % number of data points used for validation

% 30 trials used in the TAC paper for both training and validation
num_trials_training = 30; 
num_trials_validation = 30;

preamble.param = param;
preamble.num_datapoints_training = num_datapoints_training;
preamble.num_datapoints_validation = num_datapoints_validation;
preamble.num_trials_training = num_trials_training;
preamble.num_trials_validation = num_trials_validation;

results.preamble = preamble;

clear preamble

%% traiing data generation

for rep = 1:num_trials_training

    ndp = num_datapoints_training; % No. of data points
    dt = param.idt;  % Sampling time
    nseg = 5; 

    ud = [];

    for i = 1:nseg

        tt = 0:dt:(ndp*dt/nseg); % Time interval

        ts = (round(1.0*rand)*(2 + 0.75*rand))*sin(2*pi/(12+5*rand)*tt + pi/2+pi/4*rand) + ...
             (round(0.8*rand)*(1 + 0.5*rand))*sin(2*pi/(19 + 8*rand)*tt + pi/2+pi/2*rand) + ...
             (1 + 0.75*rand)*sin(2*pi/(17+10*rand)*tt + pi/2+pi/4*rand) + ...
             (round(0.95*rand)*(1 + 0.5*rand))*sin(2*pi/(8+5*rand)*tt + pi/2+pi/2*rand) + ... 
             (0.75 + 0.2*rand)*sin(2*pi/(5+2*rand)*tt + pi/2+pi/4*rand) + ...
             (round(rand)*(1 + 0.5*rand))*sin(2*pi/(4+1.5*rand)*tt + pi/2+pi/4*rand) + ...
             (1*round(rand))*sin(2*pi/(4.1+3*rand)*tt + pi/2+pi/4*rand);     

        ud = [ud,ts];
    end


    % Simulation
    % -------------------------------------------------------------------------

    dt = param.idt; 
    tspan = 0:dt:(ndp*dt);
    x0 = zeros(nx,1);

    [t_sim, x_sim] = ode45(@(t,x) msd_nonlinearSpring_gen(x,ud,t,param), tspan, x0);

    x_sim = x_sim';
    y_sim = x_sim(2,:);

    % Add some noise to the simulated quantities:
    xn = x_sim + 1*mvnrnd(zeros(size(x_sim,1),1),1e-4*eye(size(x_sim,1)),length(x_sim))';
    yn = xn(2,:); 
    
    % raw data, no scaling/state estimation    
    raw_data.u = ud(:,1:size(x_sim,2));
    raw_data.y = yn(:,1:size(x_sim,2));
    
    % Scale the quantities for better numerical properties
    ssf = diag(max(abs(xn),[],2));    % State
    osf = diag(max(abs(yn),[],2));    % Output
    xsc = ssf\xn;
    ysc = osf\yn;

    % Filter the outputs for state estimates        
    [xd,yd,ud] = processStates4nlmsd(xsc,ysc,ud,dt);

    % Structure to pass problem data to identification algorithms
    nlid_data.x = xd;
    nlid_data.u = ud;
    nlid_data.y = yd;

    % Signal to noise ratio:
    snr_emp = var(y_sim)/var(y_sim-yn);
    fprintf('SNR: %.5e (%.5edB)\n',snr_emp,10*log10(snr_emp))

    % store
    results.training{rep}.raw_data = raw_data;
    results.training{rep}.filtered_data = nlid_data;
    results.training{rep}.ssf = ssf; % state scaling factor
    results.training{rep}.osf = osf; % output scaling factor
     
end  
    

%% validation
        
for rep = 1:num_trials_validation

    % Random inputs
    % No. of segments (packed into a certain duration)
    ndp = 1e3; % No. of data points
    dt = 0.1;  % Sampling time
    nseg = 5; % No. of segments

    ud = [];

    for i = 1:nseg

        tt = 0:dt:(ndp*dt/nseg); % Time interval

        ts = (round(1.0*rand)*(2 + 0.75*rand))*sin(2*pi/(12+5*rand)*tt + pi/2+pi/4*rand) + ...
             (round(0.8*rand)*(1 + 0.5*rand))*sin(2*pi/(19 + 8*rand)*tt + pi/2+pi/2*rand) + ...
             (1 + 0.75*rand)*sin(2*pi/(17+10*rand)*tt + pi/2+pi/4*rand) + ...
             (round(0.95*rand)*(1 + 0.5*rand))*sin(2*pi/(8+5*rand)*tt + pi/2+pi/2*rand) + ... 
             (0.75 + 0.2*rand)*sin(2*pi/(5+2*rand)*tt + pi/2+pi/4*rand) + ...
             (round(rand)*(1 + 0.5*rand))*sin(2*pi/(4+1.5*rand)*tt + pi/2+pi/4*rand) + ...
             (1*round(rand))*sin(2*pi/(4.1+3*rand)*tt + pi/2+pi/4*rand);

        ud = [ud,ts];
    end


    % Simulation
    % -------------------------------------------------------------------------

    uv = ud;

    dt = param.idt; 
    tspan = 0:dt:100;
    x0 = zeros(nx,1);

    [t_sim, xv_sim] = ode45(@(t,x) msd_nonlinearSpring_gen(x,uv,t,param), tspan, x0);

    xv_sim = xv_sim';
    xv_sim = xv_sim + mvnrnd(zeros(4,1),1e-4*eye(4),length(xv_sim))';
    yv_sim = xv_sim(2,:);

    uv = uv(:,1:length(xv_sim));
    
    raw_data.u = uv;
    raw_data.y = yv_sim;
    
    results.validation{rep}.raw_data = raw_data;

end


%% save

% Generate fresh file name:
s_date = datestr(clock,'ddmmmyyyy_HHMM');
s_name = ['data_' s_date];
s_name_t = [s_name '.mat'];
save(s_name_t,'results','-v7.3')























