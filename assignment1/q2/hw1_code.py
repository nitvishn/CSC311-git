from sklearn.feature_extraction import DictVectorizer


def load_data():
    
    # load clean data
    real_file = open('clean_real.txt', 'r')
    fake_file = open('clean_fake.txt', 'r')

    # build set of words, and store sentences as list of tokens
    sentences = []
    words = set()
    real_size = 0
    fake_size = 0
    for line in real_file:
        words.update(set(line.split()))
        sentences.append(line.split())
        real_size += 1
    for line in fake_file:
        words.update(set(line.split()))
        sentences.append(line.split())
        fake_size += 1

    # convert sentences from lists of tokens to mappings of words to frequencies
    for i in range(len(sentences)):
        s = sentences[i]
        s_dict = {}
        for word in words:
            s_dict[word] = s.count(word)
        sentences[i] = s_dict

    # make labels
    labels = ['real'] * real_size + ['fake'] * fake_size

    # extract features
    v = DictVectorizer()
    X = v.fit_transform(sentences)

    Y = 

    # return features and labels 
    return 

if __name__ == "__main__":
    load_data()