% Split data into train and test and label
close all; clear; clc;
%---------------------------------%
% train_percent_array = 0.5:0.1:0.9;
train_percent_array = 0.8;
%---------------------------------%
manual_label_folder = 'D:\Project\Segmentaion_revise1\02_04\manualSyllable';
energy_label_folder = 'D:\Project\Segmentaion_revise1\02_04\energySyllable';
harmar_label_folder = 'D:\Project\Segmentaion_revise1\02_04\harmarSyllable';
auto_energy_label_folder = 'D:\Project\Segmentaion_revise1\02_04\automatic_energySyllable';
renyi_label_folder = 'D:\Project\Segmentaion_revise1\02_04\renyi_entropySyllable_3';
spectral_label_folder = 'D:\Project\Segmentaion_revise1\02_04\spectral_entropySyllable';
%---------------------------------%
for i = 1:length(train_percent_array)
    train_percent = train_percent_array(i);
    
    audio_folder = 'D:\Project\Segmentaion_revise1\data\';
    audio_list = dir(audio_folder);
    audio_list = audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), audio_list));
    
    % split data into training and testing
    nSpecies = length(audio_list);
    audio_cell = cell(1, nSpecies);
    traing_len_array = zeros(1, nSpecies);
    for iSpecies = 1:nSpecies
        
        speciesName = audio_list(iSpecies).name;
        species_path = [audio_folder, speciesName];
        [audio_data, sr] = audioread(species_path);
        
        audio_length = length(audio_data);
        
        training_len = round(audio_length*train_percent);        
        training_data = audio_data(1: training_len);
        testing_data = audio_data(training_len+1: audio_length);
        
        traing_len_array(iSpecies) = training_len;
        
        % read label
        manual_label_path = [manual_label_folder, '\', speciesName(1:end-4), '.csv'];
        energy_label_path = [energy_label_folder, '\',speciesName(1:end-4), '.csv'];
        harmar_label_path = [harmar_label_folder,'\',  speciesName(1:end-4), '.csv'];
        auto_energy_label_path = [auto_energy_label_folder, '\', speciesName(1:end-4), '.csv'];
        renyi_label_path = [renyi_label_folder, '\',speciesName(1:end-4), '.csv'];
        spectral_label_path = [spectral_label_folder,'\',  speciesName(1:end-4), '.csv'];
        
        manual_data = csvread(manual_label_path);
        energy_data = csvread(energy_label_path);
        harmar_data = csvread(harmar_label_path);
        auto_energy_data = csvread(auto_energy_label_path);
        renyi_data = csvread(renyi_label_path);
        spectral_data = csvread(spectral_label_path);
        
        % figure;
        % subplot(311); plot(manual_data);
        % subplot(312); plot(energy_data);
        % subplot(313); plot(harmar_data);
        % 
        % figure;
        % plot(manual_data);
        % hold on; plot(energy_data);
        % hold on; plot(harmar_data);
        
        energy_data = energy_data(1:length(harmar_data));
        auto_energy_data= auto_energy_data(1:length(harmar_data));
        renyi_data= renyi_data(1:length(harmar_data));
        spectral_data= spectral_data(1:length(harmar_data));
        
        % combine data of different segmentation methods
        all_label = [manual_data, energy_data, harmar_data, auto_energy_data, renyi_data, spectral_data];
        
        training_label = all_label(1: training_len, :);
        testing_label = all_label(training_len+1: end, :);
        
        %================%
        saveFolder_train = ['.\02_04\allSyllable_3\training_', num2str(train_percent), '\'];
        create_folder(saveFolder_train);
        savePath = [saveFolder_train, 'fileName.csv'];
        frog = speciesName(1:(length(speciesName) - 4));
        csvPath = strrep(savePath, 'fileName', frog);
        csvwrite(csvPath,training_label);
        
        saveFolder_test = ['.\02_04\allSyllable_3\testing_', num2str(train_percent), '\'];
        create_folder(saveFolder_test);
        savePath = [saveFolder_test, 'fileName.csv'];
        frog = speciesName(1:(length(speciesName) - 4));
        csvPath = strrep(savePath, 'fileName', frog);
        csvwrite(csvPath,testing_label);
        
        training_folder =  ['.\data_split\training_', num2str(train_percent), '\'];
        testing_folder = ['.\data_split\testing_', num2str(train_percent), '\'];
        
        create_folder(training_folder);  create_folder(testing_folder);
        training_path = [training_folder, audio_list(iSpecies).name];
        testing_path = [testing_folder, audio_list(iSpecies).name];
        
        audiowrite(training_path,training_data,sr);
        audiowrite(testing_path,testing_data,sr);
        
    end
    
    save_folder_len = '.\training_len_folder\';
    create_folder(save_folder_len)
    csvwrite([save_folder_len, 'training_len_', num2str(train_percent), '.csv'], traing_len_array);
    
end
%[EOF]


