% Load the data from the JSON file
fid = fopen('sentiment_data.json');
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
data = jsondecode(str);

% Define the star ratings
star_ratings = [1, 2, 3, 4, 5];

% Set up the colors for each star rating
colors = {'r', 'm', 'k', 'b', 'g'};
markers = {'o', '*', '+', 'x', 's'};
alphas = {1, 0.6, 0.4, 0.3, 0.1};

% Create a 2D scatter triangle plot for each star rating

% Initialize lists to store mean sentiment scores
mean_pos_scores = [];
mean_neg_scores = [];
mean_neu_scores = [];

figure;
hold on;
for i = 1:length(star_ratings)
    current_star_rating = star_ratings(i);
    current_reviews = data([data.stars] == current_star_rating);
    
    % Extract the positive, negative, and neutral scores for each review
    positive_scores = [];
    negative_scores = [];
    neutral_scores = [];
    for j = 1:length(current_reviews)
        current_review = current_reviews(j);
        current_sentiment_scores = current_review.sentiment_scores;
        positive_scores = [positive_scores current_sentiment_scores.pos];
        negative_scores = [negative_scores current_sentiment_scores.neg];
        neutral_scores = [neutral_scores current_sentiment_scores.neu];
    end
    
    % Calculate the x and y coordinates of each point
    x = positive_scores - negative_scores;
    y = neutral_scores;% * sqrt(3);
    
    % Plot the 2D scatter plot for the current star rating
    s2 = scatter(x, y, 25, colors{i}, 'filled');
    alpha(s2, alphas{i})

    % Store the mean sentiment scores for the current star rating
    mean_pos_scores(i) = mean(positive_scores);
    mean_neg_scores(i) = mean(negative_scores);
    mean_neu_scores(i) = mean(neutral_scores);
end

% Plot the mean sentiment scores for all star ratings
for i = 1:length(star_ratings)
    % Plot the mean sentiment score for the each star rating
    mean_x = mean_pos_scores(i) - mean_neg_scores(i);
    mean_y = mean_neu_scores(i);% * sqrt(3);
    s2 = scatter(mean_x, mean_y, 100, 'k', markers{i}, 'LineWidth', 2);
    legend_strings{i} = sprintf('%d Star (%c)', star_ratings(i), markers{i});
end

xlabel('Polarity');
ylabel('Neutrality');
title('Sentiment Scores by Star Rating');
legend([legend_strings], 'Location', 'best');
grid on;
hold off;
%% 

% Prepare data for Pearson correlation
all_star_ratings = [];
all_sentiment_scores = [];
minimal_sentiment_scores = [];

for i = 1:length(star_ratings)
    current_star_rating = star_ratings(i);
    current_reviews = data([data.stars] == current_star_rating);
    
    for j = 1:length(current_reviews)
        current_review = current_reviews(j);
        current_sentiment_scores = current_review.sentiment_scores;
        all_sentiment_scores = [all_sentiment_scores; current_sentiment_scores.pos current_sentiment_scores.neg current_sentiment_scores.neu];
        all_star_ratings = [all_star_ratings; current_star_rating];
    end
end

% Calculate Pearson correlation coefficients
R = corrcoef([all_star_ratings all_sentiment_scores]);

% Display the correlation matrix
disp(R);

% Display the correlation matrix with proper formatting
fprintf('\nCorrelation Matrix:\n');
fprintf('               Star   Positive   Negative   Neutral\n');
fprintf('Star Rating    %5.3f   %5.3f   %5.3f   %5.3f\n', R(1,:));
fprintf('Positive       %5.3f   %5.3f   %5.3f   %5.3f\n', R(2,:));
fprintf('Negative       %5.3f   %5.3f   %5.3f   %5.3f\n', R(3,:));
fprintf('Neutral        %5.3f   %5.3f   %5.3f   %5.3f\n', R(4,:));
%% 

% Transform the star ratings using Box-Cox transformation
star_ratings_transformed = boxcox(star_ratings);

% Plot the histogram of the transformed star ratings
figure;
histogram(star_ratings_transformed, 'Normalization', 'pdf');
title('Histogram of Transformed Star Ratings');
xlabel('Transformed Star Ratings');
ylabel('Probability Density');

%%

% % Fit the multiple linear regression model
% X = all_sentiment_scores;
% y = all_star_ratings;
% 
% hist(y)
% 
% mdl = fitlm(X(:,1:2), y, 'VarNames', {'Positive', 'Negative', 'StarRating'});
% 
% % Display the model summary
% disp(mdl);
% 
% % Display the MLR model summary
% fprintf('\nMultiple Linear Regression Model Summary:\n');
% disp(mdl);
% 
% % Display the coefficients and p-values with proper formatting
% fprintf('\nCoefficients and P-values:\n');
% fprintf('               Estimate    P-value\n');
% fprintf('Intercept     %9.4f   %9.4f\n', mdl.Coefficients.Estimate(1), mdl.Coefficients.pValue(1));
% fprintf('Positive      %9.4f   %9.4f\n', mdl.Coefficients.Estimate(2), mdl.Coefficients.pValue(2));
% fprintf('Negative      %9.44f  %9.4f\n', mdl.Coefficients.Estimate(3), mdl.Coefficients.pValue(3));
% % fprintf('Neutral       %9.4f            %9.4f\n', mdl.Coefficients.Estimate(4), mdl.Coefficients.pValue(4));
% 
% % Generate partial dependence plots for the Positive, Negative, and Neutral variables
% figure;
% subplot(1,2, 1);
% plotPartialDependence(mdl,'Positive');
% xlabel('Positive Sentiment Score');
% ylabel('Star Rating');
% title('Partial Dependence of Star Rating on Positive Sentiment Score');
% subplot(1,2,2);
% plotPartialDependence(mdl,'Negative');
% xlabel('Negative Sentiment Score');
% ylabel('Star Rating');
% title('Partial Dependence of Star Rating on Negative Sentiment Score');
% subplot(1,3,3);
% plotPartialDependence(mdl,'Neutral');
% xlabel('Neutral Sentiment Score');
% ylabel('Star Rating');
% title('Partial Dependence of Star Rating on Neutral Sentiment Score');
%% 

% % Levene's Test for Homogeneity of Variances
% % Assuming your data is stored in a JSON object named "data"
% % and contains "stars" and a dictionary "sentiment_scores" keyed on "neg", "net", and "pos"
% 
% sentiment_types_lv = {'pos', 'neg', 'neu'};
% 
% for dep_var_lv = 1:3
%     % Extract the scores based on the sentiment type
%     scores_lv = zeros(length(data), 1);
%     for i_lv = 1:length(data)
%         current_review_lv = data(i_lv);
%         current_sentiment_scores_lv = current_review_lv.sentiment_scores;
%         scores_lv(i_lv) = current_sentiment_scores_lv.(sentiment_types_lv{dep_var_lv});
%     end
%     
%     % Get the unique star ratings
%     unique_star_ratings = unique(star_ratings);
%     
%     % Perform robust Levene's test
%     [~, p_value_lv] = robust_levene_test(scores_lv, [data.stars]');
%     fprintf('Levene''s Test P-value (Variable %d): %8.6f\n', dep_var_lv, p_value_lv);
% end
% 
% %%
% 
% % Create a matrix to store sentiment scores and corresponding star ratings
% sentiment_data = [];
% group = [];
% 
% for i = 1:length(star_ratings)
%     current_star_rating = star_ratings(i);
%     current_reviews = data([data.stars] == current_star_rating);
%     
%     for j = 1:length(current_reviews)
%         current_review = current_reviews(j);
%         current_sentiment_scores = current_review.sentiment_scores;
%         sentiment_data = [sentiment_data; current_sentiment_scores.pos, current_sentiment_scores.neg, current_sentiment_scores.neu];
%         group = [group; current_star_rating];
%     end
% end
% 
% % Perform MANOVA
% [d, p, stats] = manova1(sentiment_data, group);
% %% 
% 
% % Calculate Pillai's Trace
% pillai_trace = sum(stats.eigenval ./ (1 + stats.eigenval));
% 
% % Display the Pillai's Trace
% fprintf('\nMANOVA Results:\n');
% fprintf('Pillai''s Trace: %8.6f\n', pillai_trace);
% fprintf('d:              %8.6f\n', d);
% fprintf('P-value:        %8.6f\n', p);
% % Calculate Canonical correlations
% canonical_correlations = sqrt(stats.eigenval);
% 
% % Display the Eigenvalues
% fprintf('Eigenvalues:\n');
% for i = 1:length(stats.eigenval)
%     fprintf('Eigenvalue %d: %8.6f\n', i, stats.eigenval(i));
% end
% 
% % Display the Canonical correlations
% fprintf('\nCanonical correlations:\n');
% for i = 1:length(canonical_correlations)
%     fprintf('Canonical correlation %d: %8.6f\n', i, canonical_correlations(i));
% end
% 
% % Display the Between-group Mahalanobis distances
% fprintf('\nBetween-group Mahalanobis distances:\n');
% for i = 1:size(stats.gmdist, 1)
%     for j = i+1:size(stats.gmdist, 2)
%         fprintf('Distance between group %d and group %d: %8.6f\n', i, j, stats.gmdist(i, j));
%     end
% end

%%

% Create a matrix to store sentiment scores and corresponding star ratings
sentiment_data = [];
group = [];

for i = 1:length(star_ratings)
    current_star_rating = star_ratings(i);
    current_reviews = data([data.stars] == current_star_rating);
    
    for j = 1:length(current_reviews)
        current_review = current_reviews(j);
        current_sentiment_scores = current_review.sentiment_scores;
        sentiment_data = [sentiment_data; current_sentiment_scores.pos, current_sentiment_scores.neg, current_sentiment_scores.neu];
        group = [group; current_star_rating];
    end
end

%%
% 
% % Test the assumptions
% 
% % Homogeneity of dispersion assumption
% disp('Testing for homogeneity of dispersion:');
% [~, p_hod] = test_homogeneity_of_dispersion(sentiment_data, group);
% fprintf('P-value for homogeneity of dispersion: %8.6f\n', p_hod);
% 
% %%
% 
% install.packages("vegan")
% 
% % Convert the data to a matrix
% X = [data.sentiment_scores.pos, data.sentiment_scores.neg, data.sentiment_scores.neu];
% 
% % Convert the group labels to a factor variable
% G = factor(star_ratings);
% 
% % Write the data to a temporary file
% tmpfile = 'temp.mat';
% save(tmpfile, 'X', 'G');
% 
% % Load the data in R
% library(R.matlab)
% data = readMat(tmpfile)
% 
% % Load the vegan package
% library(vegan)
% 
% % Run PERMANOVA
% result = adonis(X ~ G, data)
% 
% % Display the PERMANOVA results
% disp(result)


%% 


% Create a histogram for each star rating
figure;
for i = 1:length(star_ratings)
% Select the sentiment scores for the current star rating
current_star_rating = star_ratings(i);
current_sentiment_scores = [data([data.stars] == current_star_rating).sentiment_scores];
current_sentiment_scores = [current_sentiment_scores.compound];

% Plot the histogram of sentiment scores
subplot(length(star_ratings), 1, i);
histogram(current_sentiment_scores, 20);
xlabel('Compound Score');
ylabel('Frequency');
title(['Star Rating ', num2str(current_star_rating)]);

% Plot the distribution curve of sentiment scores
% subplot(length(star_ratings), 1, i);
% [f, x] = ksdensity(current_sentiment_scores);
% plot(x, f);
% xlabel('Compound Score');
% ylabel('Density');
% title(['Star Rating ', num2str(current_star_rating)]);
end

% Adjust the layout of the subplots
sgtitle('Distribution of Compound Scores by Star Rating');
set(gcf,'Position',[100 100 800 800]);
set(findall(gcf,'-property','FontSize'),'FontSize',14);
%%

% % Create a 3D scatter plot for each star rating
% figure;
% hold on;
% for i = 1:length(star_ratings)
%     current_star_rating = star_ratings(i);
%     current_reviews = data([data.stars] == current_star_rating);
%     
%     % Extract the positive, negative, and neutral scores for each review
%     positive_scores = [];
%     negative_scores = [];
%     neutral_scores = [];
%     for j = 1:length(current_reviews)
%         current_review = current_reviews(j);
%         current_sentiment_scores = current_review.sentiment_scores;
%         positive_scores = [positive_scores current_sentiment_scores.pos];
%         negative_scores = [negative_scores current_sentiment_scores.neg];
%         neutral_scores = [neutral_scores current_sentiment_scores.neu];
%     end
%     
%     % Plot the 3D scatter plot for the current star rating
%     s3 = scatter3(positive_scores, negative_scores, neutral_scores, colors{i}, 'filled');
%     alpha(s3, alphas{i})
% end
% 
% xlabel('Positive Score');
% ylabel('Negative Score');
% zlabel('Neutral Score');
% title('Sentiment Scores by Star Rating');
% legend('1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars');
% grid on;
% hold off;

% Create a 2D scatter triangle plot for each star rating
% figure;
% 
% for i = 1:length(star_ratings)
%     current_star_rating = star_ratings(i);
%     current_reviews = data([data.stars] == current_star_rating);
%     
%     % Extract the positive, negative, and neutral scores for each review
%     positive_scores = [];
%     negative_scores = [];
%     neutral_scores = [];
%     for j = 1:length(current_reviews)
%         current_review = current_reviews(j);
%         current_sentiment_scores = current_review.sentiment_scores;
%         positive_scores = [positive_scores current_sentiment_scores.pos];
%         negative_scores = [negative_scores current_sentiment_scores.neg];
%         neutral_scores = [neutral_scores current_sentiment_scores.neu];
%     end
%     
%     % Calculate the x and y coordinates of each point
%     x = positive_scores - negative_scores;
%     y = neutral_scores * sqrt(3);
%     
%     % Create a subplot for the current star rating
%     subplot(2, 3, i);
%     s2 = scatter(x, y, 25, colors{i}, 'filled');
%     alpha(s2, alphas{i})
% 
%     xlabel('Polarity');
%     ylabel('Neutrality');
%     title(sprintf('Sentiment Scores for %d Star Rating', current_star_rating));
%     grid on;
% end


function [h, p_value] = robust_levene_test(data, group_labels)
    % ROBUST_LEVENE_TEST Performs a robust Levene's test for homogeneity of variances
    %
    % Inputs:
    %   data        - a column vector of data points
    %   group_labels - a column vector of group labels (same length as 'data')
    %
    % Outputs:
    %   h        - 1 if the null hypothesis is rejected, 0 otherwise
    %   p_value  - the p-value for the test

    % Extract unique group labels and the number of groups
    unique_labels = unique(group_labels);
    num_groups = length(unique_labels);

    % Calculate group medians
    group_medians = zeros(num_groups, 1);
    for i = 1:num_groups
        group_medians(i) = median(data(group_labels == unique_labels(i)));
    end

    % Calculate the absolute deviations from group medians
    deviations = zeros(size(data));
    for i = 1:num_groups
        deviations(group_labels == unique_labels(i)) = abs(data(group_labels == unique_labels(i)) - group_medians(i));
    end

    % Perform one-way ANOVA on the deviations
    [p_value, ~, stats] = anova1(deviations, group_labels, 'off');

    % Make the decision
    alpha = 0.05;
    h = p_value < alpha;
end

function [p_value, F, test_stat] = test_homogeneity_of_dispersion(data, group_labels)

    % Calculate the group centroids
    unique_groups = unique(group_labels);
    num_groups = length(unique_groups);
    centroids = zeros(num_groups, size(data, 2));
    for i = 1:num_groups
        current_group_data = data(group_labels == unique_groups(i), :);
        centroids(i, :) = mean(current_group_data);
    end

    % Calculate the distances from each observation to its group centroid
    distances = zeros(size(data, 1), 1);
    for i = 1:size(data, 1)
        group_index = find(unique_groups == group_labels(i));
        distances(i) = norm(data(i, :) - centroids(group_index, :));
    end

    % Perform PERMDISP using a permutation test
    num_permutations = 999;
    observed_F = sum((distances - mean(distances)).^2);
    permuted_F = zeros(num_permutations, 1);
    for i = 1:num_permutations
        permuted_group_labels = group_labels(randperm(length(group_labels)));
        permuted_centroids = zeros(num_groups, size(data, 2));
        for j = 1:num_groups
            current_group_data = data(permuted_group_labels == unique_groups(j), :);
            permuted_centroids(j, :) = mean(current_group_data);
        end

        permuted_distances = zeros(size(data, 1), 1);
        for j = 1:size(data, 1)
            group_index = find(unique_groups == permuted_group_labels(j));
            permuted_distances(j) = norm(data(j, :) - permuted_centroids(group_index, :));
        end

        permuted_F(i) = sum((permuted_distances - mean(permuted_distances)).^2);
    end

    % Calculate the p-value
    p_value = (sum(permuted_F >= observed_F) + 1) / (num_permutations + 1);

    % Return the observed and permuted test statistics
    F = observed_F;
    test_stat = permuted_F;
end


