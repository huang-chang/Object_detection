export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/model_main.py --pipeline_config_path="/data1/object/tf_record/ssd_inception_v3.config" --model_dir="/data1/object/tf_model" --num_train_steps=80000 --sample_1_of_n_eval_examples=1 --alsologtostderr

	
