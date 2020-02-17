% Manual segmentation
%======================%
clc; close all; clear;
%======================%
audio_folder = 'D:\Project\Segmentaion_revise1\data\';
%======================%
training_data_folder = audio_folder;
training_audio_list = dir(training_data_folder);
training_audio_list = training_audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), training_audio_list));
%=====================%
nList = length(training_audio_list);
outResult = cell(1, nList);
for iList = 1:nList
    
    speciesName = training_audio_list(iList).name;
    speciesPath = [training_data_folder, '\', speciesName];
    
    [y, fs] = audioread(speciesPath);
    
    y = y-mean(y); % remove mean value
    y = y./max(abs(y)); % normalization
        
    label_data = xlsread(['.\label\', speciesName(1:end-4), '.xlsx']);
    
    % remove NAN
    nan_index = ~isnan(label_data);
    label_data = label_data(nan_index(:,1),:);
    
    %==============%
    segments = zeros(1, length(y));
    [row, col] = size(label_data);
    for r = 1:row
        temp_start = label_data(r, 1);
        temp_stop = label_data(r, 2);        
        segments(temp_start: temp_stop) = 1;        
    end
       
    %==============%
    saveFolder = ['.\02_04\manualSyllable\'];
    create_folder(saveFolder);
    savePath = [saveFolder, 'fileName.csv'];
    frog = speciesName(1:(length(speciesName) - 4));
    csvPath = strrep(savePath, 'fileName', frog);
    csvwrite(csvPath,segments);
    
end
%[EOF]




