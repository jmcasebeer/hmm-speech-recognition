function A = symmetrize(A)
    A = triu(A) + triu(A)' - diag(diag(A));
end

