%% It decides the number of items of the couting task.
% Small numbers are more likely to appear.
function number = generateNum(N, alpha)
% check parameters
if nargin == 1    
    alpha = 1;
end

% generate the unnormalized probability for 1 to N 
n = 1 : N;
prob = 1 ./ (n .^ alpha);
% choose the number
number = randomChoose(prob);
end


% take a unnormalized PDF, randomly output a value (w.r.t. that PDF)
function choice = randomChoose(strengths)
v = rand; % a number between 0 and 1

nstr = strengths/sum(strengths); % normalize strengths
cstr = cumsum(nstr);        % get top edges of bins
choice = find(cstr>v,1);    % returns the bin v falls in

if isempty(choice)
    choice = randi(length(strengths));
end

end

