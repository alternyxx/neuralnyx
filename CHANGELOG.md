# V 0.1.1
* Now every epoch actually trains the last batch. Done by passing `batch_indices` 
to a `pipeline.compute()` call.

* Added optimizers with in TrainingOptions

* Allows shuffling of data through shuffle_data in `TrainingOptions` 

* Also added a value for stopping when meeting a threshold via `cost_threshold`

```rs
// Above three changes with the struct
TrainingOptions {
    optimizer: Optimizer::Adam(0.001),
    shuffle_data: true,
    cost_threshold: 0.01,
    ..Default::default()
}
```

* `NeuralNet` now has a gauge function to easily gauge the accuracy

```rs
let accuracy = nn.gauge(&images, &labels).unwrap();
```

* Updated the mnist example to download the datasets from github instead 
of storing them in the repository.

* Removed the Tic-Tac-Toe example as it was ridiculously simple.