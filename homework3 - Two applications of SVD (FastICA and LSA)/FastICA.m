clear all
close all
clc

n = 3                       % Num of sources
p = n;                      % Num of independent sources

N = 10000;                  % Time slots
M = N;


%% Generating sources
files = {'gong.mat'
        'laughter.mat'
        'train.mat'
        };
    
names = replace(files, '.mat', '.wav')
S = zeros(n, N);

for i = 1:n
    test     = load(files{i});
    y        = test.y(1:N,1);
    audiowrite(strcat('original_', names{i}), y, test.Fs);
    S(i,:)   = y;
end


%% Generating observations (mixed signals)
rng(42);
A = rand(n,n);
X = A*S;


for i = 1:n
    audiowrite(strcat('mixed_', names{i}), X(i,:), test.Fs);
end


%% Prewithening
Xmean = mean(X,2)                          % Computing mean by row

for i = 1:n
    X(i,:) = X(i,:) - Xmean(i);             % Centering X
end


%ExxT    = cov(X);                           %The covariance matrix of X is computed and stored in ExxT
%[E,D]   = eig(X*X');                        %Eigenvalue decomposition is applied on the covariance matrix of X, ExxT

[U,Sig,V] = svd(X, 'econ');                  % A smarter way to do the same operations is compute the reduced svd decomposition
E = U
D = Sig^2
Z = D^(-1/2) * E'*X                        % Prewhitened matrix


%% Computation of weight matrix (independent components)
W = zeros(p,n);                             

iterations = 1000;                          %The amount of iterations used in the fastICA algorithm

for p = 1:p
    
wp = rand(n,1);
wp = wp / sqrt(wp'*wp);

    %aggiungere convergenza wp
    for i = 1:iterations
        wp = wp/norm(wp);
        %G       = tanh(wp'*Z);
        u = wp'*Z;
        G =u.* exp(-(u.^2)/2);
        %Gder    = 1-tanh(wp'*Z).^2;
        Gder = (1 - u.^2).*exp(-u.*u/2);
        
        wp = 1/M*Z*G' - 1/M*Gder*ones(M,1)*wp;
        for j = 1:p-1
            wp = wp - wp'*W(:,j)*W(:,j);
        end
        wp = wp / norm(wp);
    end
    
    W(:,p) = wp; 
end


%% Retrieving approximation of original sources
Sest = W'*Z;

Sest = Sest([1,3,2],:);

for i = 1:n
    Sest(i,:) = Sest(i,:)/norm(Sest(i,:))*norm(S(i,:));
    audiowrite(strcat('recovered_', names{i}), Sest(i,:), test.Fs)
end


%% Plot signals
figure
for i = 1:n
    subplot(3,n,i)
    ylim([-1,1])
    plot(S(i,:))
    title([names{i}])
    
    subplot(3,n,i+n)
    plot(X(i,:))
    title(['Mixed ',num2str(i)])
    
    subplot(3,n,i+2*n)
    ylim([-1,1])
    plot(Sest(i,:))
    title(['Reconstructed ',num2str(i)])
end
