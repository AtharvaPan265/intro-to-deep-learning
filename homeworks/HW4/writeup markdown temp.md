| Optimizer      | Description                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| SGD            | `x += - learning_rate * dx`                                                                                                                         |
| SGD + Momentum | `v = mu * v - learning_rate * dx`<br>`x += v`<br>v:  accumulates an exponentially decaying average of past gradients<br>mu: momentum typically ~0.9 |
| AdaGrad        |                                                                                                                                                     |
| RMSprop        |                                                                                                                                                     |
| Adam           |                                                                                                                                                     |

- **Conventional SGD (Stochastic Gradient Descent):**
	- Updates weights based solely on the gradient of the current batch and a fixed learning rate (`x += - learning_rate * dx`)
	- Uses the same learning rate for all parameters
	- Can be slow to converge if the loss landscape has narrow valleys or plateaus
	- May get stuck in local minima or saddle points where the gradient is zero
	- Can exhibit oscillations or jitter along steep directions in the loss landscape
	
- **SGD with Momentum:**
	- Introduces a velocity term (`v`) that accumulates an exponentially decaying average of past gradients
	- The update involves integrating velocity (`v = mu * v - learning_rate * dx`) and then updating the position (`x += v`)
	- The hyperparameter `mu` (momentum, typically ~0.9) controls the influence of past gradients
	- Helps accelerate convergence, especially along shallow directions, by building up "momentum"
	- Dampens oscillations across steep directions
	- Improves the ability to escape local minima and navigate saddle points compared to conventional SGD
- **AdaGrad (Adaptive Gradient):**
	- Adapts the learning rate for each parameter individually based on the history of squared gradients for that parameter
	- Accumulates the sum of squared gradients in a `cache` variable (`cache += dx**2`)
	- Divides the learning rate by the square root of this cache (`x += - learning_rate * dx / (np.sqrt(cache) + eps)`), effectively reducing the learning rate for parameters with large gradients and increasing it for parameters with small/infrequent gradient
	- **Limitation:** The `cache` grows monotonically, causing the effective learning rate to continually decrease throughout training, potentially stopping learning too early
- **RMSprop (Root Mean Squared Propagation):**
	- Also adapts the learning rate per parameter, similar to AdaGrad
	- Addresses AdaGrad's limitation by using an exponentially decaying _moving average_ of squared gradients instead of accumulating the full sum (`cache = decay_rate * cache + (1 - decay_rate) * dx**2`)
	- The update rule remains similar to AdaGrad's, dividing by the square root of this moving average cache (`x += - learning_rate * dx / (np.sqrt(cache) + eps)`)
	- The `decay_rate` hyperparameter (typically ~0.9 to 0.999) controls the scale of the moving average
	- Prevents the learning rate from vanishing too quickly, allowing for continued learning1.


