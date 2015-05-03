OBJECT_FOLDER=../data/daquar-37-object
NUMBER_FOLDER=../data/daquar-37-number
COLOR_FOLDER=../data/daquar-37-color

python mscoco_prep.py -object -output $OBJECT_FOLDER -len 52
./build_word_embed.sh $OBJECT_FOLDER
python mscoco_prep.py -number -output $NUMBER_FOLDER -len 52
./build_word_embed.sh $NUMBER_FOLDER
python mscoco_prep.py -color -output $COLOR_FOLDER -len 52
./build_word_embed.sh $COLOR_FOLDER