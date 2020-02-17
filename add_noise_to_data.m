% add background noise
%=============%
baseFolder = '.\data';
%=============%
wavin3_pink = load('.\noise\pink.mat'); % load babble noise
wavin3_pink = wavin3_pink.pink;

[wavin3_rain, ~] = audioread('.\noise\rain.wav');
wavin3_rain = wavin3_rain(:, 1);

wavin3_babble = load('.\noise\babble.mat'); % load babble noise
wavin3_babble = wavin3_babble.babble;

[wavin3_wind, ~] = audioread('.\noise\wind.wav');
wavin3_wind = wavin3_wind(:, 1);

%=============%
snrValueArray = [-20, -10, 0, 10, 20];
nSNR = length(snrValueArray);
for iSNR = 1:nSNR
    
    snrValue = snrValueArray(iSNR);
    
    speciesList = dir(baseFolder);
    speciesList = speciesList(arrayfun(@(x) ~strcmp(x.name(1), '.'), speciesList));
    nSpecies = length(speciesList);
    for iSpecies = 1:nSpecies
        
        species_path = [baseFolder, '\', speciesList(iSpecies).name];
        
        [cleanSignal, fs] = audioread(species_path);
        
        % add white noise
        noisedSignal_white = awgn(cleanSignal, snrValue, 'measured');
        create_folder(['.\noise_data\', num2str(snrValue), '\noise_data_white_noise\', ]);
        audiowrite(['.\noise_data\', num2str(snrValue), '\noise_data_white_noise\', speciesList(iSpecies).name], noisedSignal_white, fs);
        
        % add pink noise
        [noisedSignal_pink,~] = addnoise(cleanSignal, wavin3_pink(1:length(cleanSignal)), snrValue);
        create_folder(['.\noise_data\', num2str(snrValue), '\noise_data_pink_noise\']);
        audiowrite(['.\noise_data\', num2str(snrValue), '\noise_data_pink_noise\', speciesList(iSpecies).name], noisedSignal_pink, fs);
        
         % add babble noise
        [noisedSignal_babble,~] = addnoise(cleanSignal, wavin3_babble(1:length(cleanSignal)), snrValue);
        create_folder(['.\noise_data\', num2str(snrValue), '\noise_data_babble_noise\']);
        audiowrite(['.\noise_data\', num2str(snrValue), '\noise_data_babble_noise\', speciesList(iSpecies).name], noisedSignal_babble, fs);
                
        % add rain
        [noisedSignal_rain,~] = addnoise(cleanSignal, wavin3_rain(1:length(cleanSignal)), snrValue);
        create_folder(['.\noise_data\', num2str(snrValue), '\noise_data_rain_noise\']);
        audiowrite(['.\noise_data\', num2str(snrValue), '\noise_data_rain_noise\', speciesList(iSpecies).name], noisedSignal_rain, fs);
        
        % add wind
        if length(cleanSignal) < length(wavin3_wind)
            [noisedSignal_wind,~] = addnoise(cleanSignal, wavin3_wind(1:length(cleanSignal)), snrValue);
        else
            wavin4 = [wavin3_wind; wavin3_wind];
            [noisedSignal_wind,~] = addnoise(cleanSignal, wavin4(1:length(cleanSignal)), snrValue);
        end
        
        create_folder(['.\noise_data\', num2str(snrValue), '\noise_data_wind_noise\']);
        audiowrite(['.\noise_data\', num2str(snrValue), '\noise_data_wind_noise\', speciesList(iSpecies).name], noisedSignal_wind, fs);
        
        %==============================%
    end
end

%==============================%


