function oneHotTable = generate_onehot_table(selected_column, label)
    oneHotTable = table(selected_column);
    oneHotTable = onehotencode(oneHotTable);
    for i=1:width(oneHotTable)
        oneHotTable.Properties.VariableNames{i} = [label '_' oneHotTable.Properties.VariableNames{i}];
    end
end