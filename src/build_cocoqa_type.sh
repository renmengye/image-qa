OBJECT_FOLDER=../data/cocoqa-object
NUMBER_FOLDER=../data/cocoqa-number
COLOR_FOLDER=../data/cocoqa-color
LOCATION_FOLDER=../data/cocoqa-location

python mscoco_prep.py -object -output $OBJECT_FOLDER -len 52
./build_word_embed.sh $OBJECT_FOLDER
python mscoco_prep.py -number -output $NUMBER_FOLDER -len 52
./build_word_embed.sh $NUMBER_FOLDER
python mscoco_prep.py -color -output $COLOR_FOLDER -len 52
./build_word_embed.sh $COLOR_FOLDER
python mscoco_prep.py -location -output $LOCATION_FOLDER -len 52
./build_word_embed.sh $LOCATION_FOLDER