clear all
close all
clc

%% Load the dataset
N_COMPONENTS = 2;
FONT_SIZE = 14
MARKER_SIZE = 14

filename = "RomeoAndJuliet.txt";
str = extractFileText(filename);

%% Tokenization and Preprocessing
textData = split(str,newline);
textData = strtrim(textData);
n_doc = textData.size(1)

documents = lower(tokenizedDocument(textData));
documents = removeWords(documents, [stopWords, "that's", 'know', 'motto']);
documents = normalizeWords(documents, 'Style', 'lemma');  
documents = erasePunctuation(documents)


documents = removeShortWords(documents, 2);
bag = bagOfWords(documents);


%% Creating document-term matrix
M = encode(bag, documents);
document_term = array2table(M, 'VariableNames', bag.Vocabulary, 'RowNames', 'd' + string(1:n_doc)) 

%% Truncated SVD decomposition
[U, S, V] = svd(full(M));

Uk = U(:, 1:N_COMPONENTS);
Sk = S(1:N_COMPONENTS, 1:N_COMPONENTS);
Vk = V(:, 1:N_COMPONENTS);


%% New representation of terms and documents
term_encoding = table(Vk*Sk, 'RowNames', bag.Vocabulary)
doc_encoding = table(Uk*Sk, 'RowNames', 'd' + string(1:n_doc))


%% K-Means clustering of documents
figure
X = doc_encoding.Variables;
y = X(:,2);
x = X(:,1);

text(x, y, '  d' + string(1:n_doc), 'FontSize', FONT_SIZE);
hold on

%opts = statset('Display','final');
[idx,C] = kmeans(X,N_COMPONENTS,'Distance','cosine','Replicates',5);

plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',MARKER_SIZE);
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',MARKER_SIZE);
plot(C(:,1),C(:,2), 'kx', 'MarkerSize',MARKER_SIZE);
legend('Cluster 1','Cluster 2','Centroids','Location','NW');
title 'Cluster Assignments and Centroids';
hold off


%% Query database

fprintf('\n\n\n')
textQuery = "die dagger"

while strcmp(textQuery, '') == 0
    
    textQuery = lower(tokenizedDocument(textQuery));
    textQuery = removeWords(textQuery, [stopWords, "that's", 'know', 'motto']);
    try
        textQuery = normalizeWords(textQuery,  'Style', 'lemma');
        % probably there is a bug in the function normalizeWords,
        % it does not allow me to specify the language. It recognize the query
        % as german and returns an error since 'lemma' is not 
    catch
    end

    textQuery = erasePunctuation(textQuery);
    textQuery = removeShortWords(textQuery, 2);
    
    query = zeros(1, N_COMPONENTS);
    for word=textQuery.Vocabulary
        if ismember(word, bag.Vocabulary)
            query = query + term_encoding(word, :).Variables;
        end
    end

    query= query/textQuery.Vocabulary.length;

    sim = 1./vecnorm(doc_encoding.Variables - query, 2, 2);
    sim = sim/norm(sim);

    [val, idx] = sort(sim, 'descend');
    
    
    for i = 1:n_doc
        ind = idx(i);
        fprintf("%.2f \t->\t (%d) d%d: %s\n", val(i),i, ind, textData(ind))
    end
    %cosine_similarities = doc_encoding.Variables*query'./(norm(doc_encoding.Variables, 2)*norm(query, 2))

    % Plot
    figure
    grid on
    x = term_encoding.Variables;
    y = x(:,2);
    x = x(:,1);
    plot(x, y, 'r.', 'MarkerSize',MARKER_SIZE)
    hold on
    text(x, y, '  ' + bag.Vocabulary,  'FontSize', FONT_SIZE)
    

    x = doc_encoding.Variables;
    y = x(:,2);
    x = x(:,1);
    plot(x, y, 'b.', 'MarkerSize',MARKER_SIZE)
    hold on
    text(x, y, '  d' + string(1:n_doc), 'FontSize', FONT_SIZE)
    hold on

    plot(query(1), query(2), 'g.', 'MarkerSize',MARKER_SIZE)
    hold on
    text(query(1), query(2), ' query',  'FontSize', FONT_SIZE)
    hold on

    legend('Terms','Documents','Query',...
           'Location','NW')
    title 'Query'
    hold off
    
    textQuery = input("\nInsert new query (empty string to end): ", 's')
    fprintf('\n\n\n')
end
