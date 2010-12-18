function features = extract_features(sound)
    Fs = 8000;
    framesize = 80;
    overlap = 20;
    D = 6; % Number of frequencies stored from each signal frame

    frames = buffer(sound, framesize, overlap);
    window = hamming(framesize);
    [~, N] = size(frames);
    features = zeros(D, N);

    i = 1;
    NFFT = 2^nextpow2(framesize);
    for frame = frames
        X = fft(frame .* window, NFFT)/framesize;
        F = Fs/2*linspace(0, 1, NFFT/2 + 1);
        
        % Finding local maxima in single-sided amplitude spectrum
        [~, peaks] = findpeaks(abs(X(1:(NFFT/2 + 1))), 'SORTSTR', 'descend');

        if isempty(peaks)
            peaks = ones(D, 1);
        elseif length(peaks) < D
            peaks = padarray(peaks, D - length(peaks), 1, 'post');
        end

        features(:, i) = F(peaks(1:D))';
        i = i + 1;
    end
end
