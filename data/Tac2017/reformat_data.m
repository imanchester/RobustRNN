clc
clear

% convert data from data.mat to a more convenient form for scipy to read

data = load('./data.mat').results;

train_u = [];
train_y = [];

val_u = [];
val_y = [];

for ii = 1:length(data.training)

    tii = data.training{ii}.raw_data;
    vii = data.validation{ii}.raw_data;

    train_y = [train_y; tii.y];
    train_u = [train_u; tii.u];
    val_y = [val_y; vii.y];
    val_u = [val_u; vii.u];
end

processed_data.train_u = train_u;
processed_data.train_y = train_y;
processed_data.val_u = val_u;
processed_data.val_y = val_y;

save("./pydata.mat",  "processed_data")