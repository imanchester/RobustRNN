clc
clear

% convert data from data.mat to a more convenient form for scipy to read

data = load('./data.mat').results;

% Process raw data
train_u_raw = [];
train_y_raw = [];
val_u_raw = [];
val_y_raw = [];

for ii = 1:length(data.training)

    tii = data.training{ii}.raw_data;
    vii = data.validation{ii}.raw_data;

    train_y_raw = [train_y_raw; tii.y];
    train_u_raw = [train_u_raw; tii.u];
    val_y_raw = [val_y_raw; vii.y];
    val_u_raw = [val_u_raw; vii.u];
end

processed_data.train_u = train_u_raw;
processed_data.train_y = train_y_raw;
processed_data.val_u = val_u_raw;
processed_data.val_y = val_y_raw;

save("./pydata_raw.mat",  "processed_data")

% Process filtered data
train_u_filt = [];
train_y_filt = [];
val_u_filt = [];
val_y_filt  = [];

for ii = 1:length(data.training)

    tii = data.training{ii}.filtered_data;
    vii = data.validation{ii}.filtered_data;

    train_y_filt = [train_y_filt; tii.y];
    train_u_filt = [train_u_filt; tii.u];
    val_y_filt = [val_y_filt; vii.y];
    val_u_filt = [val_u_filt; vii.u];
end

processed_data.train_u = train_u_filt;
processed_data.train_y = train_y_filt;
processed_data.val_u = val_u_filt;
processed_data.val_y = val_y_filt;

save("./pydata_filtered.mat",  "processed_data")
