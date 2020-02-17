%===========================%
% use the histogram for generating the ground truth
%===========================%
clear; close all; clc;
%===========================%
t = 1;
fs = 44100;
baseFolder = '.\out_label_frame_xgb_500\';
%===========================%
label_type_array = {'strong', 'mid', 'weak'};
n_label_type = length(label_type_array);
for i_label_type = 1:n_label_type
    
    select_label_type = label_type_array{i_label_type};
    
    %syllable_seg_type = {'manual', 'harmar', 'energy', 'auto_energy', 'renyi', 'spectral'};
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
            nTime = length(win_time_array);
            for iTime = 1:nTime
                win_time = win_time_array(iTime);
                win_step_array = [0.2, 0.5, 0.8];
                %======================%
                nOver = length(win_step_array);
                for iOver = 1:nOver
                    win_step = win_step_array(iOver);
                    win_len = floor(win_time);
                    % path
                    out_label_folder = [baseFolder, select_seg_type, '\training_', num2str(selectPerc),...
                        '\win_len_', num2str(win_len), '_win_over_', num2str(win_step)];
                    out_label_path = [out_label_folder, '\outlabel.csv'];
                    % read ground truth
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
                    %label_array = cell2mat(label_vector_final);
                    %gt_array = cell2mat(gt_final);
                    
                    %disp([sum(label_array), sum(gt_array)]);
                    
                    % read label information from python output
                    % alignment
                    python_label_path = '.\python_label_sample_2D\mid_win_len_882_win_over_0.2';
                    label_len = csvread([python_label_path, '\len.csv']);
                    revised_label_array = cell(length(label_vector_final), 1);
                    revised_gt_array = cell(length(label_vector_final), 1);
                    for k = 1:length(label_vector_final)
                        temp_label_vector= label_vector_final{k};
                        temp_gt_vector = gt_final{k};
                        
                        temp_len = label_len(k);
                        revised_label_array{k} = temp_label_vector(1:temp_len);
                        revised_gt_array{k} = temp_gt_vector(1:temp_len);
                    end
                    
                    %final_syllable_label = cell2mat(revised_label_array);
                    %final_gt = cell2mat(revised_gt_array);
                    %disp([sum(cell2mat(revised_label_array)),  sum(cell2mat(revised_gt_array))]);
                    
                    %======================%
                    %stats = confusionmatStats(final_gt, final_syllable_label);
                    
                    %  a = hist(gt_array, 1:23);
                    %  weight_value = a/sum(a);
                    %======================%
                    % window-based evaluation
                    % slide-windowing for each frog species
                    win_time_array_eva = [ 0.02, 0.05, 0.1, 0.2, 0.5] * fs; % ms
                    %win_time_array_eva = [ 0.02 ] * fs; % ms
                    kTime = length(win_time_array_eva);
                    for jTime = 1:kTime
                        win_time_eva = floor(win_time_array_eva(jTime));
                        over_array = [0.2,0.5,0.8];
                        %over_array = [0.2];
                        kOver = length(over_array);
                        for jOver = 1:kOver
                            select_over = over_array(jOver);
                            
                            final_label = cell(1, length(revised_label_array));
                            final_gt = cell(1, length(revised_label_array));
                            for kk = 1:length(revised_label_array)
                                
                                label_array = cell2mat(revised_label_array(kk));
                                gt_array = cell2mat(revised_gt_array(kk));
                                
                                % windowing for generating labels
                                [win_label_array, ~] = window_move(label_array, win_time_eva, select_over);
                                [win_gt_array, ~]   = window_move(gt_array, win_time_eva, select_over);
                                
                                %======================%
                                [rr, cc] = size(win_label_array);
                                final_win_label = zeros(cc, 1);
                                final_win_gt = zeros(cc, 1);
                                for c = 1:cc
                                    
                                    % out label
                                    temp_label_array = win_label_array(:, c);
                                    real_label_len = length(temp_label_array(temp_label_array ~= 0 ));
                                    ratio = real_label_len / length(temp_label_array);
                                    if strcmp(select_label_type, 'weak')
                                        if ratio > 0.5
                                            [aa, bb] = hist(temp_label_array, [0:23]);
                                            temp_haha = bb(aa == max(aa));
                                            final_win_label(c) = temp_haha(1);
                                        end
                                    elseif strcmp(select_label_type, 'mid')
                                        if ratio > 0.75
                                            [aa, bb] = hist(temp_label_array, [0:23]);
                                            temp_haha = bb(aa == max(aa));
                                            final_win_label(c) = temp_haha(1);
                                        end
                                    else
                                        if ratio == 1
                                            [aa, bb] = hist(temp_label_array, [0:23]);
                                            temp_haha = bb(aa == max(aa));
                                            final_win_label(c) = temp_haha(1);
                                        end
                                    end
                                    
                                    % ground truth
                                    temp_gt_array = win_gt_array(:, c);
                                    real_gt_len = length(temp_gt_array(temp_gt_array ~= 0 ));
                                    ratio_gt = real_gt_len / length(temp_gt_array);
                                    if strcmp(select_label_type, 'weak')
                                        if ratio_gt > 0.5
                                            [aa, bb] = hist(temp_gt_array, [0:23]);
                                            temp_haha = bb(aa == max(aa));
                                            final_win_gt(c) = temp_haha(1);
                                        end
                                    elseif strcmp(select_label_type, 'mid')
                                        if ratio_gt > 0.75
                                            [aa, bb] = hist(temp_gt_array, [0:23]);
                                            temp_haha = bb(aa == max(aa));
                                            final_win_gt(c) = temp_haha(1);
                                        end
                                    else
                                        if ratio_gt == 1
                                            [aa, bb] = hist(temp_gt_array, [0:23]);
                                            temp_haha = bb(aa == max(aa));
                                            final_win_gt(c) = temp_haha(1);
                                        end
                                    end
                                end
                                
                                % final
                                final_label{kk} = final_win_label;
                                final_gt{kk} = final_win_gt;
                            end
                            
                            % combine windowed label
                            final_hist_label = cell2mat(final_label');
                            final_hist_gt = cell2mat(final_gt');
                            
                            win_stats_hist = confusionmatStats(final_hist_gt, final_hist_label);
                            disp(mean(win_stats_hist.Fscore));
                            %disp(win_stats_hist.confusionMat)
                            
                            save_folder = ['.\evaluation_win_frame_ratio_', select_label_type, '_xgb_500\', select_seg_type, '\training_', num2str(selectPerc), ...
                                '\win_len_', num2str(win_len), '_win_over_', num2str(win_step), ...
                                '_eva_len_', num2str(win_time_eva), '_eva_over_', num2str(select_over)];
                            create_folder(save_folder);
                            csvwrite([save_folder, '\F1_scocre.csv'], win_stats_hist.Fscore);
                            csvwrite([save_folder, '\accuracy.csv'], win_stats_hist.accuracy);
                            csvwrite([save_folder, '\precision.csv'], win_stats_hist.precision);
                            csvwrite([save_folder, '\recall.csv'], win_stats_hist.recall);
                            csvwrite([save_folder, '\confusionMat.csv'], win_stats_hist.confusionMat);
                            
                        end
                    end
                    %================%
                end
            end
        end
    end
end
%[EOF]

