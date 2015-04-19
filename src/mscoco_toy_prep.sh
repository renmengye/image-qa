TARGET=../data/cocoqa-toy
python mscoco_toy_prep.py
python word2vec_lookup.py $TARGET/question_vocabs.txt $TARGET/question_vocabs_vec.npy
python word2vec_lookup.py $TARGET/answer_vocabs.txt $TARGET/question_vocabs_vec.npy
python word_embedding.py $TARGET/question_vocabs_vec.npy $TARGET/word-emeb-q.npy 0 no
python word_embedding.py $TARGET/answer_vocabs_vec.npy $TARGET/word-embed-a.npy 0 no
