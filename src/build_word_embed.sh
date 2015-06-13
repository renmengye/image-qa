TARGET=$1
COCOWD=/ais/gobi3/u/$USER/data/mscoco/word2vec300.txt
COCOWD500=/ais/gobi3/u/$USER/data/mscoco/word2vec500.txt
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
echo "Looking up Google News 300 embedding..."
python word2vec_lookup.py -w $TARGET/$VOCAB_LIST_FILE_Q -o $TARGET/$VOCAB_NPY_FILE_Q
python word2vec_lookup.py -w $TARGET/$VOCAB_LIST_FILE_A -o $TARGET/$VOCAB_NPY_FILE_A
echo "Building Google News 300 embedding..."
python word_embedding.py -v $TARGET/$VOCAB_NPY_FILE_Q $TARGET/$WDEMBED_FILE_Q -d 0
python word_embedding.py -v $TARGET/$VOCAB_NPY_FILE_A $TARGET/$WDEMBED_FILE_A -d 0
echo "Looking up COCO 300 embedding..."
python word2vec_lookuptxt.py -w $TARGET/$VOCAB_LIST_FILE_Q -o $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q -t $COCOWD
python word2vec_lookuptxt.py -w $TARGET/$VOCAB_LIST_FILE_A -o $TARGET/$VOCAB_NPY_FILE_CUSTOM_A -t $COCOWD
echo "Building COCO 300 embedding..."
python word_embedding.py -v $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q -o $TARGET/$WDEMBED_FILE_CUSTOM_Q -d 0
python word_embedding.py -v $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q -o $TARGET/$WDEMBED_FILE_CUSTOM_A -d 0
echo "Looking up COCO 500 embedding..."
python word2vec_lookuptxt.py -w $TARGET/$VOCAB_LIST_FILE_Q -o $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q_500 -t $COCOWD500
python word2vec_lookuptxt.py -w $TARGET/$VOCAB_LIST_FILE_A -o $TARGET/$VOCAB_NPY_FILE_CUSTOM_A_500 -t $COCOWD500
echo "Building COCO 500 embedding..."
python word_embedding.py -v $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q_500 -o $TARGET/$WDEMBED_FILE_CUSTOM_Q_500 -d 0
python word_embedding.py -v $TARGET/$VOCAB_NPY_FILE_CUSTOM_Q_500 -o $TARGET/$WDEMBED_FILE_CUSTOM_A_500 -d 0