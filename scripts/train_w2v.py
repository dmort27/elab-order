import gensim.models
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SCHCorpus:
    """An iterator that yields sentences (lists of str)."""

    def preprocess(self, sent):
        sent = sent.replace('ELIPSISTOKEN', ' ELIPSISTOKEN ')
        tokens = []
        for w in sent.split():
            if w[0].isdigit():
                tokens.append('NUM')
            else:
                tokens.append(w.lower() if w != 'ELIPSISTOKEN' else w)
        return tokens

    def __iter__(self):
        data_folder = "../data/hmong/sch_corpus2_tok"
        data_files = os.listdir(data_folder)
        for file in data_files:
            with open(os.path.join(data_folder, file)) as f:
                for line in f:
                    # assume there's one document per line, tokens separated by whitespace
                    # yield utils.simple_preprocess(line)
                    yield self.preprocess(line)


sentences = SCHCorpus()
model = gensim.models.Word2Vec(sentences=sentences,
                               vector_size=100,
                               min_count=10,
                               workers=8,
                               sg=0,
                               epochs=5)
model.save("../data/hmong/sch.cbow.w2v")
model = gensim.models.Word2Vec(sentences=sentences,
                               vector_size=100,
                               min_count=10,
                               workers=8,
                               sg=1,
                               epochs=5)
model.save("../data/hmong/sch.sg.w2v")