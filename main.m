clc
clear
close all

[audio_signals word_labels] = load_audio_from_folder('audio');

display(sprintf('Loaded a total of %d audio signals for the following words:', length(audio_signals)))
display(unique(word_labels))

vocabulary = Vocabulary;

crossval('mcr', audio_signals', word_labels', 'predfun', @vocabulary.train_test, 'kfold', 5)

% predicted_word_labels = vocabulary.train_test(audio_signals', word_labels', audio_signals');
% 
% misses = 0;
% for i = 1:length(word_labels)
%     fact  = word_labels(i);
%     guess = predicted_word_labels(i);
%     if ~isequal(fact, guess)
%         misses = misses + 1;
%         display(sprintf('Miss %d: Predicted %s, but was %s.', misses, char(guess), char(fact)))
%     end
% end
% 
% mcr = misses / length(word_labels)