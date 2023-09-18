#!/usr/bin/env bash
source conf.sh


if [[ -z $IMAGE_TAG ]]
then
	echo "No docker tag provided. Cannot run docker image."
else
	echo "Warning: Removing containers with the prefix $CONTAINER_NAME* "
	docker rm -f $CONTAINER_NAME "$CONTAINER_NAME-tensorboard"
	echo "*********************Starting train_or_test.py script.*********************"
	config_file=$1
	docker run --gpus all \
				-d \
				--env-file conf.sh \
				--shm-size 8G \
				--name $CONTAINER_NAME \
				-v /dataset:/tf/datasets \
				-v /torch_filter/latent_space:/tf \
				alpha/torch_filter:$IMAGE_TAG \
				/bin/bash -c "python train.py --config ${config_file}"
	echo "*********************Starting tensorboard at localhost:7062*********************"
	logdir="/tf/experiments/$(basename $config_file .yaml)/summaries"
	docker run --name "$CONTAINER_NAME-tensorboard" \
				--gpus=all \
				--network host \
				-d \
				--env-file conf.sh \
				-v /torch_filter/latent_space:/tf \
				alpha/torch_filter:$IMAGE_TAG \
				/bin/bash -c "tensorboard --logdir $logdir --host 0.0.0.0 --port 7062 --reload_multifile=true"
fi