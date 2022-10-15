%% input dataset
dataset = readtable("ENB2012_data.xlsx");
dataset = table2array(dataset);
[row_data, col_data] = size(dataset);

%% shuffle dataset
rng(16, "twister");
row_swap = randi(row_data, row_data, 2);
for n=1:row_data
    dataset([row_swap(n, 1) row_swap(n, 2)],:) = dataset([row_swap(n, 2) row_swap(n, 1)],:);
end

%% training BPNN (70%)
train_percentage = 70/100;
n_train = round(train_percentage * row_data);

input = dataset(1:n_train, 1:8);
[input_row, input_col] = size(input);
target_train = dataset(1:n_train, 9:10);
[target_row, target_col] = size(target_train);
n_data = input_row;

%% preprocessing data

% normalisasi dengan z-score
input = zscore(input);

% normalisasi data target
a = 10;
b = 1;
target_train = a + ((target_train - min(target_train)).*(b-a))./(max(target_train) - min(target_train));

%% initialization of NN parameter
n_input_layer = input_col;
n_hidden_layer = 12;
n_output_layer = target_col;

alpha = 0.5;
miu = 0.5;
error_epoch_old = 1:1000;
ibest = 0;

for Li=0:1
    %% inisialisasi bobot - metode nguyen-widrow

    % input layer -> hidden layer
    beta = 0.7 * n_hidden_layer^(1 / n_input_layer);
    v_ij = rand(n_input_layer, n_hidden_layer) - 0.5;
    for i = 1:n_hidden_layer
        norma(i) = sqrt(sum(v_ij(:,i).^2));
        v_ij(:,i) = (beta*v_ij(:,i)) / norma(i);
    end
    v_0j = (2 * beta * rand(1, n_hidden_layer) - beta);

    % hidden layer -> output layer
    w_jk = rand(n_hidden_layer, n_output_layer) - 0.5;
    w_0k = rand(1, n_output_layer) - 0.5;
    
    %% training BPNN
    max_epoch = 1000;
    target_err = 0.0001;

    stop = 0;
    epoch = 1;
    delta_wjk_old = 0;
    delta_w0k_old = 0;
    delta_vij_old = 0;
    delta_v0j_old = 0;

    while stop == 0 && epoch <= max_epoch
        for n=1:n_data
            %% Feedforward
            % Feedforward
            xi = input(n,:);
            ti = target_train(n,:);

            % komputasi input layer ke hidden layer
            z_inj = xi * v_ij + v_0j;
            for j=1:n_hidden_layer
                zj(1, j) = 1 / (1 + exp(-z_inj(1,j)));
            end

            % komputasi hidden layer ke output layer
            y_ink = zj * w_jk + w_0k;
            for k=1:n_output_layer
                yk(1,k) = 1 / (1 + exp(-y_ink(1,k)));
            end

            % store error
            error(1,n) = 0.5 * sum((yk - ti).^2);

            %% Backpropagation

            % komputasi dari output layer ke hidden layer
            dok = (yk - ti).*(yk).*(1 - yk);
            delta_wjk = alpha * zj' * dok + miu * delta_wjk_old;
            delta_w0k = alpha * dok + miu * delta_w0k_old;
            delta_wjk_old = delta_wjk;
            delta_w0k_old = delta_w0k;

            % komputasi dari hidden layer ke input layer
            doinj = dok * w_jk';
            doj = doinj.*zj.*(1-zj);
            delta_vij = alpha * xi' * doj + miu * delta_vij_old;
            delta_v0j = alpha * doj + miu * delta_v0j_old;
            delta_vij_old = delta_vij;
            delta_v0j_old = delta_v0j;

            % memperbarui bobot dan bias
            w_jk = w_jk - delta_wjk;
            w_0k = w_0k - delta_w0k;
            v_ij = v_ij - delta_vij;
            v_0j = v_0j - delta_v0j;
        end
        err_per_epoch(1, epoch) = sum(error) / n_data;

        if err_per_epoch(1, epoch) < target_err
            stop = 1;
        end

        epoch = epoch + 1;
    end
    
    if Li == 1
        error_epoch_old = err_per_epoch;
    end

    if min(err_per_epoch) < min (error_epoch_old)
        error_epoch_old = err_per_epoch;
        ibest = Li;
    end
end

err_per_epoch = error_epoch_old;
