%{
This script performs a bioacoustical signal segmentation in batch mode
using both the energy of the signal and the zero crossing rate. It is an
version of which was published in (please cite):
Expert Systems with Applications , 2015, 42, 7367 - 7374
An incremental technique for real-time bioacoustic signal segmentation.
Colonna, J. G.; Cristo, M. A. P.; Salvatierra, M. & Nakamura, E. F.

Copyright (c) 2016 Juan Gabriel Colonna, Federal University of Amazonas (UFAM - ICOMP)

%}
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
    
    %% constants
    onset = [];
    syllables = [];
    window = 1024;
    %     overlap = round(window/3);
    overlap = window;
    clear Overall_segments
    
    speciesName = training_audio_list(iList).name;
    speciesPath = [training_data_folder, '\', speciesName];
    
    [x, fs] = audioread(speciesPath);

    x = x-mean(x); % remove mean value
    x = x./max(abs(x)); % normalization
    
    %% Segmentation
    j = 1;
    for i = 1: overlap: (length(x) - window)
        start_win = i;
        stop_win = i + window - 1;
        %E(j) = mean(x(i-round(window/2):round(window/2)+i).^2);
        E(j) = mean(x(start_win:stop_win).^2);
        index(j) = i;
        j = j+1;
    end
    
    E = (E - min(E))./(max(E) - min(E));
    %T_E = abs(mean(E)-mean(1-E))/2; % Automatic threshold
    
    block_size = 1000;
    
    nBlock = ceil(length(E) / block_size);
    for iBlock = 1: nBlock
        start = (iBlock-1)*block_size +1;
        stop = start + block_size - 1;
        
        block_E = E(start:min(stop, length(E)));
        
        [idx, value] = hist(block_E);
        [~,loc] = max(idx);
        block_thresh(iBlock) = value(min(loc+1, length(value)));
    end
    
    block_thresh = [block_thresh(1:end-1), block_thresh(end-1)];   
    sample_thresh = repelem(block_thresh, block_size);
    
    segment = zeros(1,length(x));  
    for i=1:length(E)
        if E(i) > sample_thresh(i)
            segment(index(i):(index(i)+round(window))) = 1;
        end
    end

    optimal_segment = segment;
    
    %================%
    saveFolder = '.\02_04\automatic_energySyllable\';
    create_folder(saveFolder);
    savePath = [saveFolder, 'fileName.csv'];
    frog = speciesName(1:(length(speciesName) - 4));
    csvPath = strrep(savePath, 'fileName', frog);
    csvwrite(csvPath,optimal_segment');
    
    %================%
    clear E
    clear Overall_segments
    clear syllables
end
%[EOF]



