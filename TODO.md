### To-DO
Things that I plan to do/change upon more time.

* Currently, the wgsl code uses nested arrays. This isn't as performant as it can be and I want 
to refactor all of them to just arrays with index math instead to minimize lookups.

* The project structure got kinda messy- but personally, I don't like the directories with just 
mod.rs either and I need more time to make a decision on this.

* Refactor the generate_wgsl function in a reasonable way without a bijillion parameters.

* ~Implement Fisher Yates shuffle for the batches and targets while keeping correspondent same.~

* Allow custom activation functions and optimizers.

* Allow softmax on layers aside from the output layers, as well as CCE with other activations.

* Allow lower precision mainly for the wgsl side.

* PLEASE BETTER ERROR HANDLING INSTEAD OF `Result<(), String>` OR PANICS

### Bugs
Bugs that I've personally experienced when using this. I have no idea if these can still occur so 
lmk if they do.

* (Only one I'm sure of) There's a UB somewhere since running this on one of my machines, result in 
ridiculously high costs (like 10000).

* I got a "thread 'main' attempted to acquire a snatch lock recursively." where 
the entire program crashes. I have experienced this once randomly and have been unable to reproduce it.

* ~Cost is always 0 when trying to train mnist on a machine of mine. Looking at the raw outputs print 
actual values but costs are somehow 0.~ Magically fixed.

* Weights and biases sometimes turn to NaNs at which the neural network is of course unable to continue 
operating. I've got no lead on where NaNs could happen...