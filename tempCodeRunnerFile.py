inputText = inputText.split()
    text_sequence = []
    for word in text:
        if word not in vocabulary:
            vocabulary[word] = len(inverse_vocabulary)
            text_sequence.append(len(inverse_vocabulary))
            inverse_vocabulary.append(word)
        else:
            text_sequence.append(vocabulary[word])
    sequences.append(text_sequence)