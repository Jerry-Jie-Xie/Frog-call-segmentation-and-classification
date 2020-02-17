% Renyi entropy segmentation under noise conditions
%======================%
clc; close all; clear;
%======================%
snrValue_array = [-20, -10, 0, 10];

nSnr = length(snrValue_array);
for iSnr = 1:nSnr
    
    select_snr = snrValue_array(iSnr);
    
    noise_type = {'noise_data_babble_noise', 'noise_data_white_noise', 'noise_data_pink_noise', ...
        'noise_data_rain_noise', 'noise_data_wind_noise'};
    nNoise = length(noise_type);
    for iNoise = 1:nNoise
        
        select_noise_type = noise_type{iNoise};
        
        audio_folder = ['D:\Project\Segmentaion_revise1\noise_data\', num2str(select_snr), '\', select_noise_type];
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
                renyi_e(j) = abs(renyi_entro(temp_x, 0.6));
                
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
                    segment(index(i):(index(i)+round(window))) = 1;
                end
            end
            
            optimal_segment = segment;
            
            %================%
            saveFolder = ['.\02_04\renyi_entropySyllable_noise\',  num2str(select_snr), '\', select_noise_type];
            create_folder(saveFolder);
            savePath = [saveFolder, '\fileName.csv'];
            frog = speciesName(1:(length(speciesName) - 4));
            csvPath = strrep(savePath, 'fileName', frog);
            csvwrite(csvPath,optimal_segment');
            
            %================%
            clear E
            clear Overall_segments
            clear syllables
        end
    end
end
%[EOF]



