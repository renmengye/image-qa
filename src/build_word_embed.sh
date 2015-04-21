TARGET=$1
COCOWD=../../../data/mscoco/train/word2vec300.txt
VOCAB_LIST_FILE_Q=question_vocabs.txt
VOCAB_LIST_FILE_A=answer_vocabs.txt
VOCAB_NPY_FILE_Q=question_vocabs_vec.npy
VOCAB_NPY_FILE_A=answer_vocabs_vec.npy
VOCAB_NPY_FILE_CUSTOM_Q=question_vocabs_custom_vec.npy
VOCAB_NPY_FILE_CUSTOM_A=answer_vocabs_custom_vec.npy
WDEMBED_FILE_Q=word-embed-q.npy
WDEMBED_FILE_A=word-embed-a.npy
WDEMBED_FILE_CUSTOM_Q=word-embed-custom-q.npy
WDEMBED_FILE_CUSTOM_A=word-embed-custom-a.npy
python word2vec_lookup.py $TARGET/$VOCAB_LIST_FILE_Q $TARGET/$VOCAB_NPY_FILE_Q
python word2vec_lookup.py $TARGET/$VOCAB_LIST_FILE_A $TARGET/$VOCAB_NPY_FILE_A
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_Q $TARGET/$WDEMBED_FILE_Q 0 no
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_A $TARGET/$WDEMBED_FILE_A 0 no
python word2vec_lookuptxt.py $TARGET/$VOCAB_LIST_FILE_Q $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q $COCOWD
python word2vec_lookuptxt.py $TARGET/$VOCAB_LIST_FILE_A $TARGET/$VOCAB_NPY_FILE_CUSTOM_A $COCOWD
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q $TARGET/$WDEMBED_FILE_CUSTOM_Q 0 no
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q $TARGET/$WDEMBED_FILE_CUSTOM_A 0 no
