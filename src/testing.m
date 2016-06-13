clear

r = 5;

for i = 1 : 1000
    v = genRandVector(5);
    nm(i) = norm(round(v));
end

mean(nm)
hist(nm)
