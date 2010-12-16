function framefrequencies = extract_features(sound)
    Fs = 8000;
    framesize = 80;
    overlap = 20;
    D = 6; % Number of frequencies stored from each signal frame

    frames = buffer(sound, framesize, overlap);
    w = hamming(framesize);
    [~, N] = size(frames);
    framefrequencies = zeros(D, N);

    i = 1;
    NFFT = 2^nextpow2(framesize);
    for frame = frames
        x = frame .* w;
        %plot(frame./max(abs(frame)))
        %hold on
        %plot(w, 'g')
        %plot(x, 'g')
        %legend('Speech signal', 'Hamming window')
        %xlabel('time')
        %pause
        X = fft(x, NFFT)/framesize;
        f = Fs/2*linspace(0, 1, NFFT/2 + 1);
        %hold off
        %plot(f, abs(X(1:(NFFT/2 + 1))))
        %xlabel('F [Hz]')
        %ylabel('|X(F)|')
        %pause
        
        % Finding local maxima in single-sided amplitude spectrum
        [~, peaklocations] = findpeaks(abs(X(1:(NFFT/2 + 1))), 'SORTSTR', 'descend');

        if isempty(peaklocations)
            peaklocations = ones(D, 1);
        elseif length(peaklocations) < D
            peaklocations = padarray(peaklocations, D - length(peaklocations), 1, 'post');
        end

        framefrequencies(:, i) = f(peaklocations(1:D))';
        i = i + 1;
    end
end
