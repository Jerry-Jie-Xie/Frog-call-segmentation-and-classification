% Harmar method for segmentation
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
            
    [syllable,sr,S,F,T,P] = harmaSyllableSeg(y, fs, hamming(512), 128, 1024, 18); 
    numberSyllable = length(syllable);
    syllableLength = zeros(1,numberSyllable);
    for i = 1:numberSyllable
        syllableLength(i) = length(syllable(i).signal);
    end
    
    segments = zeros(1, length(y));
    start = zeros(1,numberSyllable); stop = zeros(1,numberSyllable);
    for iSyllable = 1:length(syllable)
        temp = sort(syllable(iSyllable).times);
        temp_start = round(temp(1) * sr);
        temp_stop = round(temp(end) * sr);
        
        segments(temp_start: temp_stop) = 1;
    end
    
    %====================%
    saveFolder = ['.\02_04\harmarSyllable\'];
    create_folder(saveFolder);
    frog = speciesName(1:end-4);
    savePath = [saveFolder,  frog, '.csv'];    
    csvwrite(savePath,segments);
    
end
%[EOF]




