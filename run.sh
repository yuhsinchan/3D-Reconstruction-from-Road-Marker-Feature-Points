source env.sh
python segment_mp.py -t test1 -m masks_vit_h
python segment_mp.py -t test2 -m masks_vit_h

python main.py -t test1
python main.py -t test2

python utils/stiching1.py -t test1
python utils/stiching1.py -t test2

python get_pred_pose.py -t test1 --threshold 0.5 -it 100
python get_pred_pose.py -t test2 --threshold 0.5 -it 100

python utils/post_processing.py -s test1
python utils/post_processing.py -s test2
