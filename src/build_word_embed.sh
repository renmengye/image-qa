TARGET=$1
COCOWD=../../../data/mscoco/train/word2vec300.txt
COCOWD500=../../../data/mscoco/train/word2vec500.txt
VOCAB_LIST_FILE_Q=question_vocabs.txt
VOCAB_LIST_FILE_A=answer_vocabs.txt
VOCAB_NPY_FILE_Q=question_vocabs_vec.npy
VOCAB_NPY_FILE_A=answer_vocabs_vec.npy
VOCAB_NPY_FILE_CUSTOM_Q=question_vocabs_custom_vec.npy
VOCAB_NPY_FILE_CUSTOM_A=answer_vocabs_custom_vec.npy
VOCAB_NPY_FILE_CUSTOM_Q_500=question_vocabs_custom_vec_500.npy
VOCAB_NPY_FILE_CUSTOM_A_500=answer_vocabs_custom_vec_500.npy
WDEMBED_FILE_Q=word-embed-q.npy
WDEMBED_FILE_A=word-embed-a.npy
WDEMBED_FILE_CUSTOM_Q=word-embed-custom-q.npy
WDEMBED_FILE_CUSTOM_A=word-embed-custom-a.npy
WDEMBED_FILE_CUSTOM_Q_500=word-embed-custom-q-500.npy
WDEMBED_FILE_CUSTOM_A_500=word-embed-custom-a-500.npy
python word2vec_lookup.py $TARGET/$VOCAB_LIST_FILE_Q $TARGET/$VOCAB_NPY_FILE_Q
python word2vec_lookup.py $TARGET/$VOCAB_LIST_FILE_A $TARGET/$VOCAB_NPY_FILE_A
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_Q $TARGET/$WDEMBED_FILE_Q 0 no
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_A $TARGET/$WDEMBED_FILE_A 0 no
python word2vec_lookuptxt.py $TARGET/$VOCAB_LIST_FILE_Q $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q $COCOWD
python word2vec_lookuptxt.py $TARGET/$VOCAB_LIST_FILE_A $TARGET/$VOCAB_NPY_FILE_CUSTOM_A $COCOWD
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q $TARGET/$WDEMBED_FILE_CUSTOM_Q 0 no
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q $TARGET/$WDEMBED_FILE_CUSTOM_A 0 no
python word2vec_lookuptxt.py $TARGET/$VOCAB_LIST_FILE_Q $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q_500 $COCOWD500
python word2vec_lookuptxt.py $TARGET/$VOCAB_LIST_FILE_A $TARGET/$VOCAB_NPY_FILE_CUSTOM_A_500 $COCOWD500
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q_500 $TARGET/$WDEMBED_FILE_CUSTOM_Q_500 0 no
python word_embedding.py $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q_500 $TARGET/$WDEMBED_FILE_CUSTOM_A_500 0 no
