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
    %overlap = round(window/3);
    overlap = window;
    
    clear Overall_segments
    
    speciesName = training_audio_list(iList).name;
    speciesPath = [training_data_folder, '\', speciesName];
    
    [x, fs] = audioread(speciesPath);
    
    % [x, fs] = audioread('hylaedactylus_1_44khz.wav'); % load signal
    x = x-mean(x); % remove mean value
    x = x./max(abs(x)); % normalization
    
    %% Segmentation    
    j = 1;
    for i = 1: overlap: (length(x) - window)
        start_win = i;
        stop_win = i + window - 1;
        E(j) = mean(x(start_win:stop_win).^2);
        
        index(j) = i;
        j = j+1;
    end
    
    E = (E - min(E))./(max(E) - min(E));
    
    k = 1;
    for T_E = [0.005, 0.3, 0.5];
        segment = zeros(1,length(x));
        syllables = [];
        
        for i=1:length(E)
            if E(i) > T_E
                segment(index(i):(index(i)+round(window))) = 1;
            end
        end
        segment = segment(1:length(x));
        
        audio = [0,segment,0];
        j = 1;
        for i=1:length(audio)-1
            if audio(i) < audio(i+1)
                onset = i+1;
            elseif audio(i) > audio(i+1)
                syllables{j,1} = x(onset:i-1)';
                j = j + 1;
            end
        end
        Overall_segments(:,k) = segment;
        Overall_syllables{k} = syllables;
        k = k+1;
    end
    
    optimal_segment = Overall_segments(:,2);
    optimal_segment = optimal_segment(1:length(x));
    %================%
    saveFolder = '.\02_04\energySyllable\';
    create_folder(saveFolder);
    savePath = [saveFolder, 'fileName.csv'];
    frog = speciesName(1:(length(speciesName) - 4));
    csvPath = strrep(savePath, 'fileName', frog);
    csvwrite(csvPath,optimal_segment);

    %================%
    clear E
    clear Overall_segments
    clear syllables
    clear Overall_syllables
end
%[EOF]



