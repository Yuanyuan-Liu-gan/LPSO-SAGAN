%% Automatic Generation of Lacquer Painting Design (LPSO-SAGAN)
% Implemented in MATLAB R2020a

clear; 
close all;
clc; 

%% 1. Configuration
params.imageSize = [128 128 3]; 
params.batchSize = 16; 
params.maxEpochs = 1000; 
params.numParticles = 50; 
params.numSAModules = 7; 
params.LPSO_Iterations = 100; 
params.lr = 0.0002; 
params.beta1 = 0.5;
params.beta2 = 0.999;

% Load Data
imageFolder = 'dataset';
if ~exist(imageFolder, 'dir')
    error('Error: Create a folder "dataset" in this directory and add images.');
end

imds = imageDatastore(imageFolder, 'IncludeSubfolders', true);
augimds = augmentedImageDatastore(params.imageSize, imds, ...
    'ColorPreprocessing', 'gray2rgb', 'DispatchInBackground', true);

%% 2. Initialize LPSO
action_probs = 0.5 * ones(params.numSAModules, 1); 
particles.position = round(rand(params.numParticles, params.numSAModules)); 
particles.velocity = zeros(params.numParticles, params.numSAModules);
particles.pbest = particles.position;
particles.pbest_fitness = inf(params.numParticles, 1);
particles.gbest = particles.position(1,:);
particles.gbest_fitness = inf;

%% 3. LPSO Optimization Loop
disp('Starting LPSO-SAGAN Optimization...');

for iter = 1:params.LPSO_Iterations
    fprintf('LPSO Iteration (%d/%d)\n', iter, params.LPSO_Iterations);
    current_iter_fitness = zeros(params.numParticles, 1);
    
    for p = 1:params.numParticles
        current_config = particles.position(p, :);
        fprintf('  Particle %d Config: %s\n', p, mat2str(current_config));
        
        try
            [dlnetGenerator, dlnetDiscriminator] = buildLPSOSAGAN(params.imageSize, current_config);
        catch Er
            fprintf('    Build Failed: %s\n', Er.message);
            current_iter_fitness(p) = 1000;
            continue;
        end
        
        [fitness, ~, ~] = trainGAN_Subroutine(dlnetGenerator, dlnetDiscriminator, augimds, params);
        current_iter_fitness(p) = fitness;
        fprintf('    Fitness (Loss): %.4f\n', fitness);
        
        % Update Best
        if fitness < particles.pbest_fitness(p)
            particles.pbest_fitness(p) = fitness;
            particles.pbest(p,:) = current_config;
        end
        if fitness < particles.gbest_fitness
            particles.gbest_fitness = fitness;
            particles.gbest = current_config;
        end
    end
    
    % Update LPSO Logic
    avg_fitness = mean(current_iter_fitness);
    a = 0.1; 
    b = 0.1;
    for p = 1:params.numParticles
        is_good = current_iter_fitness(p) < avg_fitness;
        config = particles.position(p, :);
        for m = 1:params.numSAModules
            if config(m) == 1
                 if is_good 
                     action_probs(m) = action_probs(m) + a * (1 - action_probs(m));
                 else 
                     action_probs(m) = (1 - b) * action_probs(m); 
                 end
            end
        end
        
        % Update Velocity/Position
        w = 0.7; 
        c1 = 1.4; 
        c2 = 1.4;
        r1 = rand(1, params.numSAModules); 
        r2 = rand(1, params.numSAModules);
        particles.velocity(p,:) = w * particles.velocity(p,:) + ...
            c1 * r1 .* (particles.pbest(p,:) - particles.position(p,:)) + ...
            c2 * r2 .* (particles.gbest - particles.position(p,:));
        
        vel_sigmoid = 1 ./ (1 + exp(-particles.velocity(p,:)));
        hybrid_prob = 0.5 * vel_sigmoid + 0.5 * action_probs';
        particles.position(p,:) = rand(1, params.numSAModules) < hybrid_prob;
    end
end

fprintf('\nBest Configuration Found: %s\n', mat2str(particles.gbest));

%% 4. Final Training
disp('Starting Final Training...');
[finalGen, finalDisc] = buildLPSOSAGAN(params.imageSize, particles.gbest);
params.maxEpochs = 50; 
[~, finalGen, ~] = trainGAN_Subroutine(finalGen, finalDisc, augimds, params);

%% 5. Generate Output
Z_test = dlarray(randn(1, 1, 100, 16, 'single'), 'SSCB');
generatedImages = predict(finalGen, Z_test);
generatedImages = extractdata(generatedImages);
generatedImages = (generatedImages + 1) / 2;
figure; 
montage(generatedImages); 
title('Generated Designs');

