# APPROXIMATION HESSIAN 

The goal of this coursework was to train different network on sunspot data and compare the results. First, the Hessian Matrix as the learning rate has been computed and the the error over epochs has been compare with a simple back-propagation using batch and the R-Propagation.


The first step was to create an online neural network which compute the changes of the weights for each pattern using an error of 1 and without updating the weights. 
Then, the Jacobian matrix has been created with each columns corresponding of a weight and each lines of a pattern.
The approximation of the Hessian has then been computed using the product of the Jacobian times the transposed of the Jacobian.
The inverse of the Hessian has been obtained using the function inv in Matlab and in order to get small values for the diagonal this code has be added : 

                        (0.001*eye(size(Hessian))))/Nb_Patterns)/100

-Nb_Patterns corresponds to the number of patterns (278 here)
- 100 is an arbitrary value to obtain smaller results. It is important to notice that this value strongly change the shape of the error curve. 

The first part of the code has been copied with an error E equal to the target minus the final output in order to computed the changes of the weights delta_weights with respect of the errors. 
Finally, the weights were updated using the following equation :

                       W i+1 = W i + Inv_H(a,a) * delta_weights(a)

- Inv_H corresponds to the inverse Hessian previously computed
- delta_weights correspond to the change of weights with respect to the errors. 

## Plot of the prediction of sunspot data for the Neural Network using Approximation Hessian


![Screen Shot 2019-11-16 at 00 03 55](https://user-images.githubusercontent.com/55028120/68981728-acc98200-0804-11ea-94cc-950627821d32.png)

One can observe that the Neural Network using the approximation of the Hessian predict almost perfectly the sunspots data.

## Plot of the error over epochs (200) between a Neural Network using Approximation Hessian as the learning rate and a Neural Network using Backpropagation. 

![Screen Shot 2019-11-16 at 00 05 23](https://user-images.githubusercontent.com/55028120/68981793-f2864a80-0804-11ea-87fd-f4d2abf864de.png)

One can observe that the Neural Network using the approximation converges faster to a small error than the simple back-propagation. Besides, even after 100 epochs the error is smaller for the Neural Network using the approximation of the Hessian. The two neural networks used the same initial weights as well as the same number of hidden nodes. Those variables have been changed in the two Neural Network and the shapes of the curves have remained similar. 
