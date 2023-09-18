
%% Calibration Part Loading MI
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/1_4/MI/Drive_2/_1_', '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
Matrix_Strat_1_MI = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    a = load(fullFileName);
    Matrix_Strat_1_MI = [Matrix_Strat_1_MI;a];
end
filePattern = fullfile('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/1_4/MI/Drive_2/_2_', '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
Matrix_Strat_2_MI = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    a = load(fullFileName);
    Matrix_Strat_2_MI = [Matrix_Strat_2_MI;a];
end

filePattern = fullfile('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/1_4/MI/Drive_2/_3_', '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
Matrix_Strat_3_MI = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    a = load(fullFileName);
    Matrix_Strat_3_MI = [Matrix_Strat_3_MI;a];
end

%% Calibration Loading Rest
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/1_4/Rest/Drive_2/_1_', '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
Matrix_Strat_1_Rest = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    a = load(fullFileName);
    Matrix_Strat_1_Rest = [Matrix_Strat_1_Rest;a];
end
filePattern = fullfile('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/1_4/Rest/Drive_2/_2_', '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
Matrix_Strat_2_Rest = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    a = load(fullFileName);
    Matrix_Strat_2_Rest = [Matrix_Strat_2_Rest;a];
end

filePattern = fullfile('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/1_4/Rest/Drive_2/_3_', '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
Matrix_Strat_3_Rest = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    a = load(fullFileName);
    Matrix_Strat_3_Rest = [Matrix_Strat_3_Rest;a];
end


%% Calculating The Connectivity MI & Rest
% Coherence_Strat_1_MI = zeros(15,20,64,64,251);
% ImCoherence_Strat_1_MI = zeros(15,20,64,64,251);

Coherence_Strat_2_MI = zeros(15,20,64,64,251);
ImCoherence_Strat_2_MI = zeros(15,20,64,64,251);

% 
% Coherence_Strat_3_MI = zeros(15,20,64,64,251);
% ImCoherence_Strat_3_MI = zeros(15,20,64,64,251);

% Coherence_Strat_1_Rest = zeros(15,20,64,64,251);
% ImCoherence_Strat_1_Rest = zeros(15,20,64,64,251);
% 

Coherence_Strat_2_Rest = zeros(15,20,64,64,251);
ImCoherence_Strat_2_Rest = zeros(15,20,64,64,251);
% 
% 
% Coherence_Strat_3_Rest = zeros(15,20,64,64,251);
% ImCoherence_Strat_3_Rest = zeros(15,20,64,64,251);

nCpus =  4;
delete(gcp('nocreate'));
poolObj = parpool('local' ,nCpus);
fprintf('Parrallel Pool created \n' )

parfor i = 1:15
%     b = size(Matrix_Strat_1_MI(i).mydata);
%     Time_Signal = Matrix_Strat_1_MI(i).mydata;
%     n_trials = b(1);
%     n_channels = b(2);
%     disp(n_trials);
%     
    b_2 = size(Matrix_Strat_2_MI(i).mydata);
    Time_Signal_2 = Matrix_Strat_2_MI(i).mydata;
    n_trials_2 = b_2(1);
    n_channels_2 = b_2(2);
    
%     b_3 = size(Matrix_Strat_3_MI(i).mydata);
%     Time_Signal_3 = Matrix_Strat_3_MI(i).mydata;
%     n_trials_3 = b_3(1);
%     n_channels_3 = b_3(2);
%     
%     
%     b_r = size(Matrix_Strat_1_Rest(i).mydata);
%     Time_Signal_r = Matrix_Strat_1_Rest(i).mydata;
%     n_trials_r = b_r(1);
%     n_channels_r = b_r(2);
    
    b_r_2 = size(Matrix_Strat_2_Rest(i).mydata);
    Time_Signal_r_2 = Matrix_Strat_2_Rest(i).mydata;
    n_trials_r_2 = b_r_2(1);
    n_channels_r_2 = b_r_2(2);
    
%     b_r_3 = size(Matrix_Strat_3_Rest(i).mydata);
%     Time_Signal_r_3 = Matrix_Strat_3_Rest(i).mydata;
%     n_trials_r_3 = b_r_3(1);
%     n_channels_r_3 = b_r_3(2);
    
    
    for k = 1:20
        for j = 1:64
%             Spectrum_Channel_1 = pwelch(squeeze(Time_Signal(k,j,:)),500,0.5,500);
            Spectrum_Channel_1_2 = pwelch(squeeze(Time_Signal_2(k,j,:)),500,0.5,500);
%             Spectrum_Channel_1_3 = pwelch(squeeze(Time_Signal_3(k,j,:)),500,0.5,500);
            
            
%             Spectrum_Channel_1_r = pwelch(squeeze(Time_Signal_r(k,j,:)),500,0.5,500);
            Spectrum_Channel_1_2_r = pwelch(squeeze(Time_Signal_r_2(k,j,:)),500,0.5,500);
%             Spectrum_Channel_1_3_r = pwelch(squeeze(Time_Signal_r_3(k,j,:)),500,0.5,500);
            
            for l = 1:64
%                 Spectrum_Channel_2 = pwelch(squeeze(Time_Signal(k,l,:)),500,0.5,500);
%                 Coh = abs(cpsd(squeeze(Time_Signal(k,j,:)),squeeze(Time_Signal(k,l,:)),500,0.5,500)./sqrt(Spectrum_Channel_1.*Spectrum_Channel_2));
%                 ImCoh = abs(imag(cpsd(squeeze(Time_Signal(k,j,:)),squeeze(Time_Signal(k,l,:)),500,0.5,500))./sqrt(Spectrum_Channel_1.*Spectrum_Channel_2));
%                 Coherence_Strat_1_MI(i,k,j,l,:) = Coh;
%                 ImCoherence_Strat_1_MI(i,k,j,l,:) = ImCoh;
                
                Spectrum_Channel_2_2 = pwelch(squeeze(Time_Signal_2(k,l,:)),500,0.5,500);
                Coh_2 = abs(cpsd(squeeze(Time_Signal_2(k,j,:)),squeeze(Time_Signal_2(k,l,:)),500,0.5,500)./sqrt(Spectrum_Channel_1_2.*Spectrum_Channel_2_2));
                ImCoh_2 = abs(imag(cpsd(squeeze(Time_Signal_2(k,j,:)),squeeze(Time_Signal_2(k,l,:)),500,0.5,500))./sqrt(Spectrum_Channel_1_2.*Spectrum_Channel_2_2));
                Coherence_Strat_2_MI(i,k,j,l,:) = Coh_2;
                ImCoherence_Strat_2_MI(i,k,j,l,:) = ImCoh_2;
                
                
%                 Spectrum_Channel_2_3 = pwelch(squeeze(Time_Signal_3(k,l,:)),500,0.5,500);
%                 Coh_3 = abs(cpsd(squeeze(Time_Signal_3(k,j,:)),squeeze(Time_Signal_3(k,l,:)),500,0.5,500)./sqrt(Spectrum_Channel_1_3.*Spectrum_Channel_2_3));
%                 ImCoh_3 = abs(imag(cpsd(squeeze(Time_Signal_3(k,j,:)),squeeze(Time_Signal_3(k,l,:)),500,0.5,500))./sqrt(Spectrum_Channel_1_3.*Spectrum_Channel_2_3));
%                 Coherence_Strat_3_MI(i,k,j,l,:) = Coh_3;
%                 ImCoherence_Strat_3_MI(i,k,j,l,:) = ImCoh_3;
                
                % Rest
%                 Spectrum_Channel_2_r = pwelch(squeeze(Time_Signal_r(k,l,:)),500,0.5,500);
%                 Coh_r = abs(cpsd(squeeze(Time_Signal_r(k,j,:)),squeeze(Time_Signal_r(k,l,:)),500,0.5,500)./sqrt(Spectrum_Channel_1_r.*Spectrum_Channel_2_r));
%                 ImCoh_r = abs(imag(cpsd(squeeze(Time_Signal_r(k,j,:)),squeeze(Time_Signal_r(k,l,:)),500,0.5,500))./sqrt(Spectrum_Channel_1_r.*Spectrum_Channel_2_r));
%                 Coherence_Strat_1_Rest(i,k,j,l,:) = Coh_r;
%                 ImCoherence_Strat_1_Rest(i,k,j,l,:) = ImCoh_r;
                
                Spectrum_Channel_2_2_r = pwelch(squeeze(Time_Signal_r_2(k,l,:)),500,0.5,500);
                Coh_2_r = abs(cpsd(squeeze(Time_Signal_r_2(k,j,:)),squeeze(Time_Signal_r_2(k,l,:)),500,0.5,500)./sqrt(Spectrum_Channel_1_2_r.*Spectrum_Channel_2_2_r));
                ImCoh_2_r = abs(imag(cpsd(squeeze(Time_Signal_r_2(k,j,:)),squeeze(Time_Signal_r_2(k,l,:)),500,0.5,500))./sqrt(Spectrum_Channel_1_2_r.*Spectrum_Channel_2_2_r));
                Coherence_Strat_2_Rest(i,k,j,l,:) = Coh_2_r;
                ImCoherence_Strat_2_Rest(i,k,j,l,:) = ImCoh_2_r;
                
                
%                 Spectrum_Channel_2_3_r = pwelch(squeeze(Time_Signal_r_3(k,l,:)),500,0.5,500);
%                 Coh_3_r = abs(cpsd(squeeze(Time_Signal_r_3(k,j,:)),squeeze(Time_Signal_r_3(k,l,:)),500,0.5,500)./sqrt(Spectrum_Channel_1_3_r.*Spectrum_Channel_2_3_r));
%                 ImCoh_3_r = abs(imag(cpsd(squeeze(Time_Signal_r_3(k,j,:)),squeeze(Time_Signal_r_3(k,l,:)),500,0.5,500))./sqrt(Spectrum_Channel_1_3_r.*Spectrum_Channel_2_3_r));
%                 Coherence_Strat_3_Rest(i,k,j,l,:) = Coh_3_r;
%                 ImCoherence_Strat_3_Rest(i,k,j,l,:) = ImCoh_3_r;
%                 
                
                
            end  
        end
    end
end
nCpus =  4;

fprintf('Parrallel Pool created \n' )

delete(gcp('nocreate'));


%% Statistical Analysis

%Average Across Trials

Average_Coh_Strat_1_MI = mean(Coherence_Strat_1_MI,2);
Average_Coh_Strat_2_MI = mean(Coherence_Strat_2_MI,2);
Average_Coh_Strat_3_MI = mean(Coherence_Strat_3_MI,2);

Average_Coh_Strat_1_Rest = mean(Coherence_Strat_1_Rest,2);
Average_Coh_Strat_2_Rest = mean(Coherence_Strat_2_Rest,2);
Average_Coh_Strat_3_Rest = mean(Coherence_Strat_3_Rest,2);

Average_ImCoh_Strat_1_MI = mean(ImCoherence_Strat_1_MI,2);
Average_ImCoh_Strat_2_MI = mean(ImCoherence_Strat_2_MI,2);
Average_ImCoh_Strat_3_MI = mean(ImCoherence_Strat_3_MI,2);

Average_ImCoh_Strat_1_Rest = mean(ImCoherence_Strat_1_Rest,2);
Average_ImCoh_Strat_2_Rest = mean(ImCoherence_Strat_2_Rest,2);
Average_ImCoh_Strat_3_Rest = mean(ImCoherence_Strat_3_Rest,2);

%% Node Strength
Node_Strength_Coh_Strat_1_MI = squeeze(sum(squeeze(Average_Coh_Strat_1_MI),3));
Node_Strength_Coh_Strat_2_MI = squeeze(sum(squeeze(Average_Coh_Strat_2_MI),3));
Node_Strength_Coh_Strat_3_MI = squeeze(sum(squeeze(Average_Coh_Strat_3_MI),3));

Node_Strength_Coh_Strat_1_Rest = squeeze(sum(squeeze(Average_Coh_Strat_1_Rest),3));
Node_Strength_Coh_Strat_2_Rest = squeeze(sum(squeeze(Average_Coh_Strat_2_Rest),3));
Node_Strength_Coh_Strat_3_Rest = squeeze(sum(squeeze(Average_Coh_Strat_3_Rest),3));

Node_Strength_ImCoh_Strat_1_MI = squeeze(sum(squeeze(Average_ImCoh_Strat_1_MI),3));
Node_Strength_ImCoh_Strat_2_MI = squeeze(sum(squeeze(Average_ImCoh_Strat_2_MI),3));
Node_Strength_ImCoh_Strat_3_MI = squeeze(sum(squeeze(Average_ImCoh_Strat_3_MI),3));

Node_Strength_ImCoh_Strat_1_Rest = squeeze(sum(squeeze(Average_ImCoh_Strat_1_Rest),3));
Node_Strength_ImCoh_Strat_2_Rest = squeeze(sum(squeeze(Average_ImCoh_Strat_2_Rest),3));
Node_Strength_ImCoh_Strat_3_Rest = squeeze(sum(squeeze(Average_ImCoh_Strat_3_Rest),3));

%% Test
Test_Coherence_St_1 = zeros(64,251);
Test_ImCoherence_St_1 = zeros(64,251);

Test_Coherence_St_2 = zeros(64,251);
Test_ImCoherence_St_2 = zeros(64,251);

Test_Coherence_St_3 = zeros(64,251);
Test_ImCoherence_St_3 = zeros(64,251);



for k = 1:64
    for l = 1:251
        [p,h,stats] = ranksum(squeeze(Node_Strength_Coh_Strat_1_MI(:,k,l)),squeeze(Node_Strength_Coh_Strat_1_Rest(:,k,l)),'method','approximate');
        if p < 0.05
            Test_Coherence_St_1(k,l) = stats.zval;
        end
        [p,h,stats] = ranksum(squeeze(Node_Strength_ImCoh_Strat_1_MI(:,k,l)),squeeze(Node_Strength_ImCoh_Strat_1_Rest(:,k,l)),'method','approximate');
        if p < 0.05
            Test_ImCoherence_St_1(k,l) = stats.zval;
        end
        
        [p,h,stats] = ranksum(squeeze(Node_Strength_Coh_Strat_2_MI(:,k,l)),squeeze(Node_Strength_Coh_Strat_2_Rest(:,k,l)),'method','approximate');
        if p < 0.05
            Test_Coherence_St_2(k,l) = stats.zval;
        end
        [p,h,stats] = ranksum(squeeze(Node_Strength_ImCoh_Strat_2_MI(:,k,l)),squeeze(Node_Strength_ImCoh_Strat_2_Rest(:,k,l)),'method','approximate');
        if p < 0.05
            Test_ImCoherence_St_2(k,l) = stats.zval;
        end
        
        
        [p,h,stats] = ranksum(squeeze(Node_Strength_Coh_Strat_3_MI(:,k,l)),squeeze(Node_Strength_Coh_Strat_3_Rest(:,k,l)),'method','approximate');
        if p < 0.05
            Test_Coherence_St_3(k,l) = stats.zval;
        end
        [p,h,stats] = ranksum(squeeze(Node_Strength_ImCoh_Strat_3_MI(:,k,l)),squeeze(Node_Strength_ImCoh_Strat_3_Rest(:,k,l)),'method','approximate');
        if p < 0.05
            Test_ImCoherence_St_3(k,l) = stats.zval;
        end
    end
end
%% Average



Node_Diff_1_Coh = Node_Strength_Coh_Strat_1_MI - Node_Strength_Coh_Strat_1_Rest;
Node_Diff_2_Coh = Node_Strength_Coh_Strat_2_MI - Node_Strength_Coh_Strat_2_Rest;
Node_Diff_3_Coh = Node_Strength_Coh_Strat_3_MI - Node_Strength_Coh_Strat_3_Rest;

Node_Diff_1_ImCoh = Node_Strength_ImCoh_Strat_1_MI - Node_Strength_ImCoh_Strat_1_Rest;
Node_Diff_2_ImCoh = Node_Strength_ImCoh_Strat_2_MI - Node_Strength_ImCoh_Strat_2_Rest;
Node_Diff_3_ImCoh = Node_Strength_ImCoh_Strat_3_MI - Node_Strength_ImCoh_Strat_3_Rest;

Ave_Node_Diff_1_Coh = squeeze(mean(Node_Diff_1_Coh,1));
Ave_Node_Diff_2_Coh = squeeze(mean(Node_Diff_2_Coh,1));
Ave_Node_Diff_3_Coh = squeeze(mean(Node_Diff_3_Coh,1));

Ave_Node_Diff_1_ImCoh = squeeze(mean(Node_Diff_1_ImCoh,1));
Ave_Node_Diff_2_ImCoh = squeeze(mean(Node_Diff_2_ImCoh,1));
Ave_Node_Diff_3_ImCoh = squeeze(mean(Node_Diff_3_ImCoh,1));
%% Each subject
Test_Coherence_St_1_Sub = zeros(15,64,251);
Test_ImCoherence_St_1_Sub = zeros(15,64,251);

Test_Coherence_St_2_Sub = zeros(15,64,251);
Test_ImCoherence_St_2_Sub = zeros(15,64,251);

Test_Coherence_St_3_Sub = zeros(15,64,251);
Test_ImCoherence_St_3_Sub = zeros(15,64,251);

for i = 1:15
    Diff_Node_Coh_St_1_MI = squeeze(sum(Coherence_Strat_1_MI(i,:,:,:,:),3));
%     Diff_Node_Coh_St_2_MI = squeeze(sum(Coherence_Strat_2_MI(i,:,:,:,:),3));
%     Diff_Node_Coh_St_3_MI = squeeze(sum(Coherence_Strat_3_MI(i,:,:,:,:),3));
%     
    Diff_Node_Coh_St_1_Rest = squeeze(sum(Coherence_Strat_1_Rest(i,:,:,:,:),3));
%     Diff_Node_Coh_St_2_Rest = squeeze(sum(Coherence_Strat_2_Rest(i,:,:,:,:),3));
%     Diff_Node_Coh_St_3_Rest = squeeze(sum(Coherence_Strat_3_Rest(i,:,:,:,:),3));
%     
%     Diff_Node_ImCoh_St_1_MI = squeeze(sum(ImCoherence_Strat_1_MI(i,:,:,:,:),3));
%     Diff_Node_ImCoh_St_2_MI = squeeze(sum(ImCoherence_Strat_2_MI(i,:,:,:,:),3));
%     Diff_Node_ImCoh_St_3_MI = squeeze(sum(ImCoherence_Strat_3_MI(i,:,:,:,:),3));
%     
%     Diff_Node_ImCoh_St_1_Rest = squeeze(sum(ImCoherence_Strat_1_Rest(i,:,:,:,:),3));
%     Diff_Node_ImCoh_St_2_Rest = squeeze(sum(ImCoherence_Strat_2_Rest(i,:,:,:,:),3));
%     Diff_Node_ImCoh_St_3_Rest = squeeze(sum(ImCoherence_Strat_3_Rest(i,:,:,:,:),3));
%     
%     
    for k = 1:64
        for l = 1:251
            [p,h,stats] = ranksum(Diff_Node_Coh_St_1_MI(:,k,l),Diff_Node_Coh_St_1_Rest(:,k,l),'method','approximate');
            if p < 0.05
                Test_Coherence_St_1_Sub(i,k,l) = stats.zval;
            end
        end
    end 
end

%% Test Node Strength Average Across bins

% Aver_Bins_Coherence_Strat_1_MI = squeeze(mean(Average_Coh_Strat_1_MI(:,:,:,:,27:36),5));
Aver_Bins_Coherence_Strat_2_MI = squeeze(mean(Average_Coh_Strat_2_MI(:,:,:,:,9:13),5));
% Aver_Bins_Coherence_Strat_3_MI = squeeze(mean(Average_Coh_Strat_3_MI(:,:,:,:,27:36),5));
% 
% Aver_Bins_Coherence_Strat_1_Rest = squeeze(mean(Average_Coh_Strat_1_Rest(:,:,:,:,27:36),5));
Aver_Bins_Coherence_Strat_2_Rest = squeeze(mean(Average_Coh_Strat_2_Rest(:,:,:,:,9:13),5));
% Aver_Bins_Coherence_Strat_3_Rest = squeeze(mean(Average_Coh_Strat_3_Rest(:,:,:,:,27:36),5));

% Node_Aver_Bins_Coherence_Strat_1_MI = squeeze(sum(Aver_Bins_Coherence_Strat_1_MI,2));
% Node_Aver_Bins_Coherence_Strat_2_MI = squeeze(sum(Aver_Bins_Coherence_Strat_2_MI,2));
% Node_Aver_Bins_Coherence_Strat_3_MI = squeeze(sum(Aver_Bins_Coherence_Strat_3_MI,2));
% 
% Node_Aver_Bins_Coherence_Strat_1_Rest = squeeze(sum(Aver_Bins_Coherence_Strat_1_Rest,2));
% Node_Aver_Bins_Coherence_Strat_2_Rest = squeeze(sum(Aver_Bins_Coherence_Strat_2_Rest,2));
% Node_Aver_Bins_Coherence_Strat_3_Rest = squeeze(sum(Aver_Bins_Coherence_Strat_3_Rest,2));
% 
% Test_Coherence_St_1 = mean(Node_Aver_Bins_Coherence_Strat_1_MI-Node_Aver_Bins_Coherence_Strat_1_Rest,1);
% Test_Coherence_St_2 = mean(Node_Aver_Bins_Coherence_Strat_2_MI-Node_Aver_Bins_Coherence_Strat_2_Rest,1);
% Test_Coherence_St_3 = mean(Node_Aver_Bins_Coherence_Strat_3_MI-Node_Aver_Bins_Coherence_Strat_3_Rest,1);
%%
% Specify the file name
% filename = '/Users/tristan.venot/Desktop/TravailThèse/openvibe-scripting/Cluster_FC_2_St1.csv';
% 
% % Read the CSV file into a table
% data = readtable(filename);
% 
% % Access the data in the table
% % For example, to access the first column:
% % Convert the table to a matrix if needed
% matrix_cluster = table2array(data);
% binary_cluster = matrix_cluster > 3;


% Aver_FC_Dif = (Aver_Bins_Coherence_Strat_1_MI)-(Aver_Bins_Coherence_Strat_1_Rest);
% % Specify the file name
% filename = 'All_FC_St1_Dr2.csv';
% 
% % Save the matrix as a CSV file
% writematrix(Aver_FC_Dif, filename);

% filename = '/Users/tristan.venot/Desktop/TravailThèse/openvibe-scripting/Cluster_FC_1_St3.csv';
% 
% % Read the CSV file into a table
% data = readtable(filename);
% 
% % Access the data in the table
% % For example, to access the first column:
% % Convert the table to a matrix if needed
% matrix_cluster = table2array(data);
% binary_cluster = matrix_cluster > 3;
% 
%
% Aver_FC_Dif = Aver_Bins_Coherence_Strat_1_MI-Aver_Bins_Coherence_Strat_1_Rest;
% % Specify the file name
% filename = 'All_FC_St1_Dr2_gamma.csv';
% 
% % Save the matrix as a CSV file
% writematrix(Aver_FC_Dif, filename);



Aver_FC_Dif = Aver_Bins_Coherence_Strat_2_MI-Aver_Bins_Coherence_Strat_2_Rest;
% Specify the file name
filename = 'All_FC_St2_Dr2_alpha.csv';

% Save the matrix as a CSV file
writematrix(Aver_FC_Dif, filename);




% Aver_FC_Dif = Aver_Bins_Coherence_Strat_3_MI-Aver_Bins_Coherence_Strat_3_Rest;
% % Specify the file name
% filename = 'All_FC_St3_Dr2_gamma.csv';
% 
% % Save the matrix as a CSV file
% writematrix(Aver_FC_Dif, filename);



%% Node Strength before

% Node_Aver_Bins_Coherence_Strat_1_MI = squeeze(sum(Average_Coh_Strat_1_MI,2));
% Node_Aver_Bins_Coherence_Strat_2_MI = squeeze(sum(Average_Coh_Strat_2_MI,2));
% Node_Aver_Bins_Coherence_Strat_3_MI = squeeze(sum(Average_Coh_Strat_3_MI,2));
% 
% Node_Aver_Bins_Coherence_Strat_1_Rest = squeeze(sum(Average_Coh_Strat_1_Rest,2));
% Node_Aver_Bins_Coherence_Strat_2_Rest = squeeze(sum(Average_Coh_Strat_2_Rest,2));
% Node_Aver_Bins_Coherence_Strat_3_Rest = squeeze(sum(Average_Coh_Strat_3_Rest,2));
% 
% Test_Coherence_St_1 = mean(squeeze(mean(Node_Aver_Bins_Coherence_Strat_1_MI(:,:,14:26),3))-squeeze(mean(Node_Aver_Bins_Coherence_Strat_1_Rest(:,:,14:26),3)),1);
% Test_Coherence_St_2 = mean(squeeze(mean(Node_Aver_Bins_Coherence_Strat_2_MI(:,:,14:26),3))-squeeze(mean(Node_Aver_Bins_Coherence_Strat_2_Rest(:,:,14:26),3)),1);
% Test_Coherence_St_3 = mean(squeeze(mean(Node_Aver_Bins_Coherence_Strat_3_MI(:,:,14:26),3))-squeeze(mean(Node_Aver_Bins_Coherence_Strat_3_Rest(:,:,14:26),3)),1);


%% Display 
figure();
imagesc(Test_Coherence_St_3(:,1:40));
colormap('jet')
colorbar