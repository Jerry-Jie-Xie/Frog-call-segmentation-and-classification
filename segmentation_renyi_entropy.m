% Renyi entropy for segmentation
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
    
    x = x-mean(x); % remove mean value
    x = x./max(abs(x)); % normalization
    
    %% Segmentation
    j = 1;
    for i = 1: overlap: (length(x) - window)
        start_win = i;
        stop_win = i + window - 1;
        temp_x = x(start_win:stop_win);
        %renyi_e(j) = abs(renyi_entro(temp_x, 0.6));
        renyi_e(j) = abs(renyi_entro(temp_x, 3));
                
        index(j) = i;
        j = j+1;
    end
    
    E = renyi_e;
    
    %====================%
    E = (E - min(E))./(max(E) - min(E));
    
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
            %  segment(index(i)-round(window/2):index(i)+round(window/2)) = 1;
            segment(index(i):(index(i)+round(window))) = 1;
            
        end
    end

    optimal_segment = segment;
    
    %================%
    saveFolder = '.\02_04\renyi_entropySyllable_3\';
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



