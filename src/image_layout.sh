D_CNN_LSTM_MODEL=imgwd_dq_b2i2w_rw500+-150_lt_ox_ms_cocomean-20150428-115457
D_BLIND_LSTM_MODEL=imgwd_dq_rw300+-150_zi-20150428-005041
D_BLIND_BOW_MODEL=imgwd_dq_bow_rw300+_zi-20150426-145525
C_CNN_LSTM_MODEL=imgwd_ccfull_b2i2w_cw2v500+-300_lt_ox_ms-20150423-092243
C_CNN_BOW_MODEL=imgwd_ccfull_bow_tanh_rw500+_lt_ox_ms-20150423-141534
C_BLIND_LSTM_MODEL=imgwd_ccfull_b2i2w_cw2v500+-300_zi-20150426-014426
C_BLIND_BOW_MODEL=imgwd_ccfull_bow_tanh_cw2v500+_zi-20150427-172351
python imageqa_layout.py -m $D_CNN_LSTM_MODEL,$D_BLIND_LSTM_MODEL -n 2-CNN-LSTM,BLIND-LSTM -k 1 -d ../data/daquar-37 -o ../selection -p ../daquar_img -daquar -i ../daquar_obj_list.txt -f daquar_object_sel.tex
python imageqa_layout.py -m $D_CNN_LSTM_MODEL,$D_BLIND_LSTM_MODEL -n 2-CNN-LSTM,BLIND-LSTM -k 1 -d ../data/daquar-37 -o ../selection -p ../daquar_img -daquar -i ../daquar_num_list.txt -f daquar_number_sel.tex 
python imageqa_layout.py -m $D_CNN_LSTM_MODEL,$D_BLIND_LSTM_MODEL -n 2-CNN-LSTM,BLIND-LSTM -k 1 -d ../data/daquar-37 -o ../selection -p ../daquar_img -daquar -i ../daquar_color_list.txt -f daquar_color_sel.tex 
python imageqa_layout.py -m $C_CNN_LSTM_MODEL,$C_CNN_BOW_MODEL,$C_BLIND_LSTM_MODEL,$C_BLIND_BOW_MODEL -n 2-CNN-LSTM,CNN-BOW,BLIND-LSTM,BLIND-BOW -k 1 -d ../data/cocoqa-full -o ../selection -p ../cocoqa_img -daquar -i ../cocoqa_obj_list.txt -f cocoqa_object_sel.tex
python imageqa_layout.py -m $C_CNN_LSTM_MODEL,$C_CNN_BOW_MODEL,$C_BLIND_LSTM_MODEL,$C_BLIND_BOW_MODEL -n 2-CNN-LSTM,CNN-BOW,BLIND-LSTM,BLIND-BOW -k 1 -d ../data/cocoqa-full -o ../selection -p ../cocoqa_img -daquar -i ../cocoqa_num_list.txt -f cocoqa_number_sel.tex 
python imageqa_layout.py -m $C_CNN_LSTM_MODEL,$C_CNN_BOW_MODEL,$C_BLIND_LSTM_MODEL,$C_BLIND_BOW_MODEL -n 2-CNN-LSTM,CNN-BOW,BLIND-LSTM,BLIND-BOW -k 1 -d ../data/cocoqa-full -o ../selection -p ../cocoqa_img -daquar -i ../cocoqa_color_list.txt -f cocoqa_color_sel.tex
python imageqa_layout.py -m $C_CNN_LSTM_MODEL,$C_CNN_BOW_MODEL,$C_BLIND_LSTM_MODEL,$C_BLIND_BOW_MODEL -n 2-CNN-LSTM,CNN-BOW,BLIND-LSTM,BLIND-BOW -k 1 -d ../data/cocoqa-full -o ../selection -p ../cocoqa_img -daquar -i ../cocoqa_location_list.txt -f cocoqa_location_sel.tex
