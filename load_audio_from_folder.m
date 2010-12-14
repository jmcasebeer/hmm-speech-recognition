function [audio_signals, word_labels] = load_audio_from_folder(audio_folder)
    audio_signals = {};
    word_labels = {};

    for word_folder = struct2cell(dir(audio_folder))
        for word_file = struct2cell(dir(sprintf('%s/%s/*.wav', audio_folder, char(word_folder(1)))))
            file_path = sprintf('%s/%s/%s', audio_folder, char(word_folder(1)), char(word_file(1)));
            
            audio_signals(end + 1) = {wavread(file_path)}; %#ok<AGROW>
            word_labels(end + 1) = word_folder(1); %#ok<AGROW>
        end
    end
end