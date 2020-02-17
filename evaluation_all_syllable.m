%===========================%
% point-based evaluation and window-based evaluation
clear; close all; clc;
fs = 44100;
baseFolder = '.\out_label_frame_xgb_500\';
%===========================%
% syllable_seg_type = {'manual', 'harmar', 'energy', 'auto_energy',  'spectral', 'renyi'};
syllable_seg_type = {'renyi'};
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
        %win_time_array =[ 256, 512, 1024]; % samples
        win_time_array = fliplr(win_time_array);
        nTime = length(win_time_array);
        for iTime = 1:nTime
            win_time = win_time_array(iTime);
            win_step_array = [0.2, 0.5, 0.8];
            %======================%
            nOver = length(win_step_array);
            for iOver = 1:nOver
                win_step = win_step_array(iOver);
                win_len = win_time;
                %win_len = floor(win_time * fs); win_step_len = floor(win_step * win_len);
                
                out_label_folder = [baseFolder, select_seg_type, '\training_', num2str(selectPerc), '\win_len_', num2str(win_len), '_win_over_', num2str(win_step)];
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
                end
                %======================%
                
                label_array = cell2mat(label_vector_final);
                gt_array = cell2mat(gt_final);
                %======================%
                stats = confusionmatStats(gt_array, label_array);
                %temp_CM = confusionmat(gt_array, label_array);
                
                %  a = hist(gt_array, 1:23);
                %  weight_value = a/sum(a);
                
                % save sample/point based evaluation result
                save_folder = ['.\evaluation_sample_frame_xgb_500\', select_seg_type, '\training_', num2str(selectPerc), '\win_len_', num2str(win_len), '_win_over_', num2str(win_step)];
                create_folder(save_folder);
                csvwrite([save_folder, '\F1_scocre.csv'], stats.Fscore);
                csvwrite([save_folder, '\accuracy.csv'], stats.accuracy);
                csvwrite([save_folder, '\precision.csv'], stats.precision);
                csvwrite([save_folder, '\recall.csv'], stats.recall);
                csvwrite([save_folder, '\confusionMat.csv'], stats.confusionMat);
                
                %======================%
                % save window-based evaluation
                % define winow length only for evaluation
                % we need to do the windowing for each frog species               
                over_array = [0.2,0.5,0.8];
                nEva = 3;
                for iEva = 1:nEva
                    select_over = over_array(iEva);
                    
                    weak_win_label = cell(1, length(label_vector_final));
                    weak_win_gt = cell(1, length(label_vector_final));
                    mid_win_label = cell(1, length(label_vector_final));
                    mid_win_gt = cell(1, length(label_vector_final));
                    strong_win_label = cell(1, length(label_vector_final));
                    strong_win_gt = cell(1, length(label_vector_final));
                    for kk = 1:length(label_vector_final)
                        label_array = cell2mat(label_vector_final(kk));
                        gt_array = cell2mat(gt_final(kk));
                        
                        win_label_mat = window_move(label_array, win_len, select_over);
                        win_gt_mat  = window_move(gt_array, win_len, select_over);
                        
                        % this is bad
                        [label_weak, label_mid, label_strong] = mat_to_label(win_label_mat, win_len, kk);
                        [gt_weak, gt_mid, gt_strong] = mat_to_label(win_gt_mat, win_len, kk);
                        
                        weak_win_label{kk} = label_weak;
                        weak_win_gt{kk} = gt_weak;
                        
                        mid_win_label{kk} = label_mid;
                        mid_win_gt{kk} = gt_mid;
                        
                        strong_win_label{kk} = label_strong;
                        strong_win_gt{kk} = gt_strong;
                        
                    end
                    % combine windowed label
                    final_weak_label = cell2mat(weak_win_label);
                    final_weak_gt = cell2mat(weak_win_gt);
                    
                    final_mid_label = cell2mat(mid_win_label);
                    final_mid_gt = cell2mat(mid_win_gt);
                    
                    final_strong_label = cell2mat(strong_win_label);
                    final_strong_gt = cell2mat(strong_win_gt);
                    
                    win_stats_weak = confusionmatStats(final_weak_gt, final_weak_label);
                    win_stats_mid = confusionmatStats(final_mid_gt, final_mid_label);
                    win_stats_strong = confusionmatStats(final_strong_gt, final_strong_label);
                    
                    save_folder = ['.\evaluation_win_frame_weak_xgb_500\', select_seg_type, '\training_', num2str(selectPerc), ...
                        '\win_len_', num2str(win_len), '_win_over_', num2str(win_step), '_frame_over_', num2str(select_over)];
                    create_folder(save_folder);
                    csvwrite([save_folder, '\F1_scocre.csv'], win_stats_weak.Fscore);
                    csvwrite([save_folder, '\accuracy.csv'], win_stats_weak.accuracy);
                    csvwrite([save_folder, '\precision.csv'], win_stats_weak.precision);
                    csvwrite([save_folder, '\recall.csv'], win_stats_weak.recall);
                    csvwrite([save_folder, '\confusionMat.csv'], win_stats_weak.confusionMat);
                    
                    save_folder = ['.\evaluation_win_frame_mid_xgb_500\', select_seg_type, '\training_', num2str(selectPerc), ...
                        '\win_len_', num2str(win_len), '_win_over_', num2str(win_step), '_frame_over_', num2str(select_over)];
                    create_folder(save_folder);
                    csvwrite([save_folder, '\F1_scocre.csv'], win_stats_mid.Fscore);
                    csvwrite([save_folder, '\accuracy.csv'], win_stats_mid.accuracy);
                    csvwrite([save_folder, '\precision.csv'], win_stats_mid.precision);
                    csvwrite([save_folder, '\recall.csv'], win_stats_mid.recall);
                    csvwrite([save_folder, '\confusionMat.csv'], win_stats_mid.confusionMat);
                                        
                    save_folder = ['.\evaluation_win_frame_strong_xgb_500\', select_seg_type, '\training_', num2str(selectPerc), ...
                        '\win_len_', num2str(win_len), '_win_over_', num2str(win_step), '_frame_over_', num2str(select_over)];
                    create_folder(save_folder);
                    csvwrite([save_folder, '\F1_scocre.csv'], win_stats_strong.Fscore);
                    csvwrite([save_folder, '\accuracy.csv'], win_stats_strong.accuracy);
                    csvwrite([save_folder, '\precision.csv'], win_stats_strong.precision);
                    csvwrite([save_folder, '\recall.csv'], win_stats_strong.recall);
                    csvwrite([save_folder, '\confusionMat.csv'], win_stats_strong.confusionMat);
      
                end
            end
        end
    end
end

%====================%
function [training_final_label_weak, training_final_label_mid, training_final_label_strong] = mat_to_label(win_label_mat, win_size, label_temp)

[~, nCol] = size(win_label_mat);
training_final_label_weak = zeros(1, nCol);
training_final_label_mid = zeros(1, nCol);
training_final_label_strong = zeros(1, nCol);
for iCol = 1:nCol
    temp_label = win_label_mat(:, iCol);
    if sum(temp_label) / win_size == 1
        training_final_label_strong(iCol) = label_temp;
    end
    if sum(temp_label) / win_size > 0.75
        training_final_label_mid(iCol) = label_temp;
    end
    if sum(temp_label) / win_size > 0.5
        training_final_label_weak(iCol) = label_temp;
    end
end
end
%====================%
%[EOF]

