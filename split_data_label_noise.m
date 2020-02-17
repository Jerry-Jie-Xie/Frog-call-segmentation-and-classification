% Split data into train and test and label
close all; clear; clc;
%---------------------------------%
% train_percent_array = 0.5:0.1:0.9;
train_percent_array = 0.8;
%---------------------------------%
snrValue_array = [-20, -10, 0, 10, 20];

nSnr = length(snrValue_array);
for iSnr = 1:nSnr
    
    select_snr = snrValue_array(iSnr);
    
    noise_type = {'babble_noise', 'white_noise', 'pink_noise', 'rain_noise', 'wind_noise'};
    nNoise = length(noise_type);
    for iNoise = 1:nNoise
        
        select_noise_type = noise_type{iNoise};
        
        %---------------------------------%
        manual_label_folder = 'D:\Project\Segmentaion_revise1\02_04\manualSyllable';
        harmar_label_folder = ['D:\Project\Segmentaion_revise1\02_04\harmarSyllable_noise\', num2str(select_snr),'\noise_data_',  select_noise_type];
        renyi_label_folder = ['D:\Project\Segmentaion_revise1\02_04\renyi_entropySyllable_noise\', num2str(select_snr), '\noise_data_',  select_noise_type];
        %---------------------------------%
        for i = 1:length(train_percent_array)
            train_percent = train_percent_array(i);
            
            audio_folder = ['D:\Project\Segmentaion_revise1\noise_data\', num2str(select_snr), '\noise_data_',  select_noise_type];
            audio_list = dir(audio_folder);
            audio_list = audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), audio_list));
            
            % split data into training and testing
            nSpecies = length(audio_list);
            audio_cell = cell(1, nSpecies);
            traing_len_array = zeros(1, nSpecies);
            for iSpecies = 1:nSpecies
                
                speciesName = audio_list(iSpecies).name;
                species_path = [audio_folder, '\', speciesName];
                [audio_data, sr] = audioread(species_path);
                
                audio_length = length(audio_data);
                
                training_len = round(audio_length*train_percent);
                
                training_data = audio_data(1: training_len);
                testing_data = audio_data(training_len+1: audio_length);
                
                traing_len_array(iSpecies) = training_len;
                
                % read label
                manual_label_path = [manual_label_folder, '\', speciesName(1:end-4), '.csv'];
                harmar_label_path = [harmar_label_folder,'\',  speciesName(1:end-4), '.csv'];
                renyi_label_path = [renyi_label_folder, '\',speciesName(1:end-4), '.csv'];
                
                manual_data = csvread(manual_label_path);
                harmar_data = csvread(harmar_label_path);
                renyi_data = csvread(renyi_label_path);               
                renyi_data= renyi_data(1:length(harmar_data));

                all_label = [manual_data, harmar_data, renyi_data];
                
                training_label = all_label(1: training_len, :);
                testing_label = all_label(training_len+1: end, :);
                
                %================%
                saveFolder_train = ['.\02_04\allSyllable\training_', num2str(train_percent), '\', num2str(select_snr), '\noise_data_',  select_noise_type, '\'];
                create_folder(saveFolder_train);
                savePath = [saveFolder_train, 'fileName.csv'];
                frog = speciesName(1:(length(speciesName) - 4));
                csvPath = strrep(savePath, 'fileName', frog);
                csvwrite(csvPath,training_label);
                
                saveFolder_test = ['.\02_04\allSyllable\testing_', num2str(train_percent), '\', num2str(select_snr), '\noise_data_',  select_noise_type, '\'];
                create_folder(saveFolder_test);
                savePath = [saveFolder_test, 'fileName.csv'];
                frog = speciesName(1:(length(speciesName) - 4));
                csvPath = strrep(savePath, 'fileName', frog);
                csvwrite(csvPath,testing_label);
                
                training_folder =  ['.\data_split_noise\',  'training_', num2str(train_percent), '\', num2str(select_snr), '\noise_data_',  select_noise_type];
                testing_folder = ['.\data_split_noise\', 'testing_', num2str(train_percent), '\', num2str(select_snr), '\noise_data_',  select_noise_type];
                
                create_folder(training_folder);  create_folder(testing_folder);
                
                training_path = [training_folder, '\', audio_list(iSpecies).name];
                testing_path = [testing_folder, '\', audio_list(iSpecies).name];
                
                audiowrite(training_path,training_data,sr);
                audiowrite(testing_path,testing_data,sr);
                
            end
            
            save_folder_len = '.\training_len_folder\';
            create_folder(save_folder_len)
            csvwrite([save_folder_len, 'training_len_', num2str(train_percent), '_noise.csv'], traing_len_array);
            
        end
    end
end
%[EOF]


