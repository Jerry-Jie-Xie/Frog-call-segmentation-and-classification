% Spectral entropy for segmentation
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
        temp_x = x(start_win:stop_win);
        % calculate
        %======================%
        fftLength = window;
        numOfBins = 10;
        h_step = fftLength / numOfBins;
        
        fftTemp = abs(fft(temp_x,2*fftLength));
        fftTemp = fftTemp(1:fftLength);
        S = sum(fftTemp+1e-12);
        x_haha = zeros(numOfBins, 1);
        for k=1:numOfBins
            x_haha(k) = sum(fftTemp((k-1)*h_step + 1: k*h_step)) / S;
        end
        
        x_haha = fftTemp./S;
        spectral_e(j) = -sum(x_haha.*log2(x_haha));
        
        %======================%
        index(j) = i;
        j = j+1;
    end
    E = spectral_e;
    
    E = (E - min(E))./(max(E) - min(E));
    E = 1-E;
    %======================%
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
    syllables = [];
    
    for i=1:length(E)
        if E(i) > sample_thresh(i)
            segment(index(i):(index(i)+round(window))) = 1;
            
        end
    end
    
    optimal_segment = segment;
    
    %================%
    saveFolder = '.\02_04\spectral_entropySyllable\';
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



