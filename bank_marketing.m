%% Input dataset
dataset = readtable("bank.csv");

% Check missing values
missing = ismissing(dataset);
rowsWithMissing = dataset(any(missing,2),:);
% disp(rowsWithMissing)

% Remove row with unknown value

dataset.job = categorical(dataset.job);
dataset.marital = categorical(dataset.marital);
dataset.education  = categorical(dataset.education, ...
    {'primary', 'secondary', 'tertiary', 'unknown'}, 'Ordinal',true);
dataset.default = categorical(dataset.default);
dataset.housing = categorical(dataset.housing);
dataset.loan = categorical(dataset.loan);
dataset.contact = categorical(dataset.contact);
dataset.day = categorical(dataset.day);
dataset.month = categorical(dataset.month);
dataset.poutcome = categorical(dataset.poutcome);
dataset.y = categorical(dataset.y);

%% Feature encoding

oneHotTable = table();
oneHotTable = [generate_onehot_table(dataset.job, 'job') ...
    generate_onehot_table(dataset.marital, 'marital') ...
    generate_onehot_table(dataset.default, 'default') ...
    generate_onehot_table(dataset.housing, 'housing') ...
    generate_onehot_table(dataset.loan, 'loan') ...
    generate_onehot_table(dataset.contact, 'contact') ...
    generate_onehot_table(dataset.day, 'day') ...
    generate_onehot_table(dataset.month, 'month') ...
    generate_onehot_table(dataset.poutcome, 'poutcome') ...
    ];


% format compact
% disp(inputCatOneHotTable)


