# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
	"learning_rate": 0.1,
	"momentum" : 0.9,
	"network_size" : 18,
	"num_classes" : 10,
	"first_num_filters" : 16,
	# ...
}

training_configs = {
	"learning_rate": 0.01,
	"max_epochs": 10,
	"batch_size": 100,
	"save_interval": 5,


	# ...
}

### END CODE HERE