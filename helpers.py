import numpy as np

def get_training_data(num_samples, data_path, copy_or_translate=False, verbose=True):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path).read().split('\n')
    for i, line in enumerate(lines[: min(num_samples, len(lines) - 1)]):
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        if copy_or_translate == True:
            if (i % 2) == 0:
                target_text = '\t' + input_text + '\n'
            else:
                target_text = '\t' + target_text + '\n'
        else:
            target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        if verbose == True:
            if (i % 2000) == 0:
                print(f'Input text: {input_text}\nTarget text: {target_text}')
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    return input_texts, target_texts, input_characters, target_characters

def preprocess_training_data(input_texts, target_texts, input_characters, target_characters, copy_or_translate=False, verbose=True):
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if verbose == True:
        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    
    if copy_or_translate == True:
        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens+2),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
    else:
        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
            if copy_or_translate == True:
                if (i % 2) == 0:    
                    encoder_input_data[i, t, -2] = 1
                else:
                    encoder_input_data[i, t, -1] = 1
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_target_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    
    return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, max_decoder_seq_length
