%===========================%
% use the histogram for generating the ground truth
%===========================%
clear; close all; clc;
%===========================%
fs = 44100;
baseFolder = '.\out_label_frame\';
%===========================%
% label_type_array = {'weak', 'mid', 'strong'};
label_type_array = {'weak'};
n_label_type = length(label_type_array);
for i_label_type = 1:n_label_type
    
    select_label_type = label_type_array{i_label_type};
    
    syllable_seg_type = {'manual', 'harmar', 'energy', 'auto_energy', 'renyi', 'spectral'};
    %===========================%
    n_seg_type = length(syllable_seg_type);
    for i_seg_type = 1:n_seg_type
        select_seg_type = syllable_seg_type{i_seg_type};
        %===========================%
        %percent_array = 0.5:0.1:0.9;
        percent_array = 0.8;
        percent_array = fliplr(percent_array);
        nPerc = length(percent_array);
        for iPerc = 1:nPerc
            selectPerc = percent_array(iPerc);
            %===========================%
            win_time_array = [ 0.02, 0.05, 0.1, 0.2] * fs; % ms
            %win_time_array = [ 0.005, 0.01] * fs; % ms
            nTime = length(win_time_array);
            for iTime = 1:nTime
                win_time = win_time_array(iTime);
                win_step_array = [0.2, 0.5, 0.8];
                %======================%
                nOver = length(win_step_array);
                for iOver = 1:nOver
                    win_step = win_step_array(iOver);
                    win_len = floor(win_time);
                    
                    out_label_folder = [baseFolder, select_seg_type, '\training_', num2str(selectPerc),...
                        '\win_len_', num2str(win_len), '_win_over_', num2str(win_step)];
                    out_label_path = [out_label_folder, '\outlabel.csv'];
                    
                    out_label_data = xlsread(out_label_path);
                    ground_truth = out_label_data(:, 3);
                    %======================%
                    audioFolder = ['.\data_split\testing_', num2str(selectPerc)];
                    audioList = dir(audioFolder);  audioList = audioList(arrayfun(@(x) ~strcmp(x.name(1), '.'), audioList));
                    nList = length(audioList);
                    label_vector_final = cell(nList, 1);
                    for iList = 1:nList
                        %======================%
                        [y, fs]  = audioread([audioFolder, '\', audioList(iList).name]);
                        
                        label_vector = zeros(length(y), 1);
                        %======================%
                        select_index = (ground_truth == iList);
                        select_out_label = out_label_data(select_index, :);
                        if sum(select_out_label(1, 1:2)) ~= 0
                            [row, ~] = size(select_out_label);
                            for r = 1:row
                                start = select_out_label(r,1);
                                stop = select_out_label(r,2);
                                label_vector(start: stop) = select_out_label(r,4);
                            end
                            label_vector_final{iList} = label_vector;
                        else
                            label_vector_final{iList} = label_vector;
                        end
                    end
                    
                    % obtain ground truth
                    gt_folder = ['.\02_04\allSyllable\testing_', num2str(selectPerc)];
                    gtList = dir(gt_folder);  gtList = gtList(arrayfun(@(x) ~strcmp(x.name(1), '.'), gtList));
                    kList = length(gtList);
                    gt_final = cell(kList, 1);
                    species_len = zeros(kList, 1);
                    for jList = 1:kList
                        temp_label =  csvread([gt_folder, '\', gtList(jList).name]);
                        
                        gt_temp = label_vector_final{jList};
                        gt_haha = temp_label(:,1) * jList;
                        if length(gt_temp) > length(gt_haha)
                            diff_len = length(gt_temp) - length(gt_haha);
                            gt_final{jList} = [gt_haha; ones(diff_len, 1) * jList];
                        else
                            gt_final{jList} = gt_haha(1:length(gt_temp));
                        end
                        
                        species_len(jList) = length(gt_temp);                        
                    end
                    
                    %======================%
                    label_array = cell2mat(label_vector_final);
                    gt_array = cell2mat(gt_final);
                    
                    % read label information from python output
                    % alignment
                    python_label_path = ['.\python_label_sample_2D\', select_label_type, '_win_len_', num2str(win_len),...
                        '_win_over_', num2str(win_step)];
                    label_len = csvread([python_label_path, '\len.csv']);
                    
                    final_python_label = csvread([python_label_path, '\label.csv']);
                    
                    revised_label_array = cell(length(label_vector_final), 1);
                    revised_gt_array = cell(length(label_vector_final), 1);
                    
                    for k = 1:length(label_vector_final)
                        temp_label_vector= label_vector_final{k};
                        temp_gt_vector = gt_final{k};
                        
                        temp_len = label_len(k);
                        revised_label_array{k} = temp_label_vector(1:temp_len);
                        revised_gt_array{k} = temp_gt_vector(1:temp_len);
                        
                    end
                    
                    final_syllable_label = cell2mat(revised_label_array);
                    final_gt = cell2mat(revised_gt_array);
                    
                    %======================%
                    stats = confusionmatStats(final_gt, final_syllable_label);
                    
                    %  a = hist(gt_array, 1:23);
                    %  weight_value = a/sum(a);
                    
                    % save sample/point based evaluation result
                    save_folder = ['.\evaluation_sample_frame\', select_seg_type, '\training_', num2str(selectPerc), '\win_len_', num2str(win_len), '_win_over_', num2str(win_step)];
                    create_folder(save_folder);
                    csvwrite([save_folder, '\F1_scocre.csv'], stats.Fscore);
                    csvwrite([save_folder, '\accuracy.csv'], stats.accuracy);
                    csvwrite([save_folder, '\precision.csv'], stats.precision);
                    csvwrite([save_folder, '\recall.csv'], stats.recall);
                    csvwrite([save_folder, '\confusionMat.csv'], stats.confusionMat);
                    
                end
            end
        end
    end
end

%[EOF]

