%======================%
clc; close all; clear;
%======================%
add_zero_coeff = 1;
do_filter = 0;
%======================%
base_folder = '.\02_04\';
%======================%
snrValue_array = [-20, -10, 0,  10];
nSnr = length(snrValue_array);
for iSnr = 1:nSnr
    select_snr = snrValue_array(iSnr);
    
    noise_type = {'pink_noise', 'white_noise','rain_noise', 'wind_noise'};
    
    nNoise = length(noise_type);
    for iNoise = 1:nNoise
        select_noise_type = noise_type{iNoise};
        
        segmentation_type = {'manual', 'harmar', 'renyi'};
        nSeg = length(segmentation_type);
        for iSeg = 1:nSeg
            select_segmentation = segmentation_type{iSeg};
            %======================%
            win_time_array = [0.1]; % samples
            %win_time_array = [0.01, 0.02, 0.03, 0.04]; % samples
            win_time_array = fliplr(win_time_array);
            nTime = length(win_time_array);
            for iTime = 1:nTime
                win_time = win_time_array(iTime);
                %win_step_array = [0.2, 0.5, 0.8];
                win_step_array = [0.8];
                %======================%
                nOver = length(win_step_array);
                for iOver = 1:nOver
                    win_step = win_step_array(iOver);
                    %======================%
                    name = {'testing', 'training'};
                    nName = length(name);
                    for iName = 1:nName
                        selectName = name{iName};
                        %percent_array = 0.5:0.1:0.9;
                        percent_array = 0.8;
                        nPerc = length(percent_array);
                        for iPerc = 1:nPerc
                            selectPerc = percent_array(iPerc);
                            syllable_folder = [base_folder, 'allSyllable\', selectName, '_',  num2str(selectPerc), '_noise\', num2str(select_snr), '\noise_data_', select_noise_type];
                            %==========================%
                            speciesList = dir(syllable_folder);
                            speciesList = speciesList(arrayfun(@(x) ~strcmp(x.name(1), '.'), speciesList));
                            nSpecies = length(speciesList);
                            for iSpecies = 1:nSpecies
                                %==========================%
                                % read csv location
                                temp_species = speciesList(iSpecies).name;
                                syllable_vector = csvread([syllable_folder, '\', temp_species]);
                                %==========================%
                                % vector to syllable
                                manual_vector = syllable_vector(:,1);
                                harmar_vector = syllable_vector(:,2);
                                renyi_vector= syllable_vector(:,3);
                                %======================%
                                manual_loc = vector_to_label([0; manual_vector;0]);
                                harmar_loc = vector_to_label([0; harmar_vector; 0]);
                                renyi_loc = vector_to_label([0; renyi_vector; 0]);
                                %======================%
                                switch select_segmentation
                                    case 'manual'
                                        select_loc = manual_loc;
                                    case 'harmar'
                                        select_loc = harmar_loc;
                                    case 'renyi'
                                        select_loc = renyi_loc;
                                end
                                %==========================%
                                [nSyllable, ~] = size(select_loc);
                                % read audio data
                                speciesName = temp_species(1:end-4);
                                audio_path = ['.\data_split_noise\', selectName, '_', num2str(selectPerc), '\',  num2str(select_snr), '\noise_data_', select_noise_type,  '\', speciesName, '.wav'];
                                
                                [signal, fs] = audioread(audio_path);
                                %==========================%
                                % Feature extraction
                                comb_feat = cell(1, nSyllable);
                                for iSyllable = 1:nSyllable
                                    startLoc = select_loc(iSyllable, 1);
                                    stopLoc = select_loc(iSyllable, 2);
                                    syllabel_signal = signal(startLoc:min(stopLoc,length(signal)));
                                    if ~isempty(syllabel_signal)
                                        %==========================%
                                        % Frequency and Temporal features
                                        win_len = floor(win_time * fs); win_step_len = floor(win_step * win_len);
                                        %==========================%
                                        specCentroid = statistic_value(SpectralCentroid(syllabel_signal, win_len, win_step_len, fs));
                                        specEntropy = statistic_value(SpectralEntropy(syllabel_signal, win_len, win_step_len, fs, 5));
                                        specFlux = statistic_value(SpectralFlux(syllabel_signal, win_len, win_step_len));
                                        specRolloff = statistic_value(SpectralRollOff(syllabel_signal, win_len, win_step_len, 0.85, fs));
                                        specFlatness =  statistic_value(SpectralFlatness(syllabel_signal, win_len, win_step_len));
                                        shortEnergy = statistic_value(ShortTimeEnergy(syllabel_signal, win_len, win_step_len));
                                        zeroCrossrate = statistic_value(zcr(syllabel_signal, win_len, win_step_len));
                                        %==========================%
                                        params = MFCCs_parameter_setting(300, fs/2, win_time, win_time*win_step, 40, add_zero_coeff, do_filter);
                                        [feats_MFCC, ~, ~] = make_feat_mfcc_rastamat(syllabel_signal, fs, 1, params);
                                        [feats_LFCC, ~, ~] = make_feat_mfcc_rastamat(syllabel_signal, fs, 2, params);
                                        %==========================%
                                        mean_MFCC = mean(feats_MFCC(1:20,:),2); mean_LFCC = mean(feats_LFCC(1:20,:),2);
                                        %==========================%
                                        comb_feat{iSyllable} = [specCentroid, specEntropy, specFlux, specRolloff, specFlatness, shortEnergy, zeroCrossrate,  mean_MFCC', mean_LFCC', iSpecies];
                                        %==========================%
                                    else
                                        disp('There is empty syllable');
                                    end
                                end
                                %==========================%
                                % change cell to matrix and save the result
                                comb_feat_mat = cell2mat(comb_feat');
                                %==========================%
                                save_folder = ['.\feat_frame_noise\', select_noise_type, '_', num2str(select_snr), '\'  select_segmentation, '\',  selectName, '_', num2str(selectPerc), '\win_len_', num2str(win_len), '_win_over_', num2str(win_step)];
                                create_folder(save_folder);
                                %==========================%
                                save_path = [save_folder, '\' temp_species, '.csv'];
                                save_path_loc = [save_folder, '\' temp_species, '_loc.csv'];
                                %==========================%
                                csvwrite(save_path, comb_feat_mat);
                                csvwrite(save_path_loc, select_loc);
                                %==========================%
                            end
                        end
                    end
                end
            end
        end
    end
end
%[EOF]


