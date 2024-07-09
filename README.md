# APS1070_Project_4 - Linear Regression on a F16 Aircraft

**Key Takeaways:**
-	 F16 Aircraft dataset, make predictions with Linear Regression, Ridge, Polynomial Features, MSE
-	Split and standardize the data
-	Linear regression: scipy.linalg, analytical solution with the inverse of X^T using training data
-	Full Batch Gradient Descent, using a fixed learning rate, iterate until RMSE is within a percentage of the direct solution, choosing a random point
o	Calculate the gradient at every step, update the weight
-	Mini batch and Stochastic Gradient Descent
o	Check the effect of batch sizes, some may not converge. 
o	Stochastic gradient is when the batch size is 1, test every point
o	There is an optimal batch size with optimal computation time
-	Gradient descent learning rate:
o	Small learning rate for smaller batch sizes to converge. But more epochs so longer computation time
-	Momentum: Moving average of gradients, to determine the new gradient
-	LR is simple to implement and less complex but may diverge

**Part 1: Getting Started**

The dataset provided describes the airplane's status and wants to predict the goal column which is the command that the controller issues. We will make predictions using linear regression without regularization.

The data is converted to a pandas data frame and split into training and validation datasets. 

The data is then standardized. Then a column of 1 is added to include the bias term.

**Part 2: Linear Regression Direct Solution**

Analytical Solution: W = np.dot(inv(np.dot(X_train_std.T, X_train_std)), np.dot(X_train_std.T, y_train))
RMSE:rmse_valid = rmse(y_valid_Pred, y_valid)

**Part 3: Full Batch Gradient Descent**

Implement gradient descent for linear regression using a fixed learning rate and iterate until validation rmse converges (Set to within 0.1% of direct solution rmse of the validation set) - Convergence threshold

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/409e8d20-0784-40db-ab91-2a718cb336ec)

Record the time it takes to complete, assign random weights, and calculate the gradient every time.

**Part 4: Mini-Batch and Stochastic Gradient Descent**

Function inputs are: input data, batch size, learning rate, and convergence threshold

Function outputs: Final weights after training, training rmse at each epoch (number of passes required to converge), validation rmse, and array of time at the end of each epoch

When the batch size is 1, it is the stochastic gradient descent. When the batch size is the number of training data, it is a full batch. Tried batch sizes in powers of 2's. 

Function inputs and outputs:

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/7e00bc38-9cb6-4420-8b78-c0a70952c6c6)

The number of iterations required per epoch is the number of data/batch size

While loop is used and stops if either the rmse decreases or if it keeps looping (doesn't converge), the rmse can go down but only consecutively for a bit, thus we compare it to the 5 and 10th previous element

Need a separate condition for the last batch where there are not enough elements to fill the last batch.

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/08ee0614-4052-44be-86e0-bfaf97c3ab8a)

We call the function multiple times using different learning rates, calculate the rmse for multiple batch sizes, and the computation time

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/19be7da0-6018-4162-9cb4-523254bcd67b)
![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/16d8fae1-17ad-4398-a4b2-6552041d5c40)
![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/39025dec-d627-44da-b24f-407185bf0081)

**Takeaways:**

The curves diverge when I use a batch size of less than 64. 

From the first graph, we see that it takes more epochs to converge for larger batch sizes. 

From the second graph, we see that smaller batch sizes reach a faster computation time.

Smaller batch sizes may be more time efficient, however, the gradient decent does not always converge. 

From the third graph, we can see there's an optimal batch size that provides the fastest computation time (at 128). After which the computation time tends to increase. 

**Part 5: Gradient Descent Learning Rate**

Find the largest learning rate such that the non-converging batches in part 4 converge to a solution

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/cc3ef8b2-c993-4864-9809-86986b8aae9b)

Trial and error was used to find the maximum converging learning rate

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/36f9a6db-2220-4128-ac54-848c9d0f82e3)

Based on the batch size that had the fastest convergence time (128), we found the optimal learning rate:

lr_array = np.linspace(0.001, 0.021, 11) 

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/72593f08-29af-400a-b4a9-fed951091032)
![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/38f5805a-8fa7-4743-b474-25d77692d382)

From the first table, we can see that by choosing a smaller learning rate, we can have the gradient descent converge. We would require a smaller learning rate for smaller batch sizes to converge.

However, from the second plot we see there's a trade-off: For a constant batch size and smaller learning rate, there are more epochs which means a longer computation time. 

**Part 6: Introducing Momentum**

Helps gradient descent converge faster, and simply behaves like a moving average of gradients.

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/c6d3c855-67cb-4a38-bf88-931195c82705)

gradient = momentum* gradient + (1-momentum)*(1/BS * np.dot(X_train_std[i:j][:].T, y_train_P6[i:j][:] - y_train[i:j][:])).reshape(1,-1)

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/4f1a3ec4-94d4-4e0f-80fd-2fed596b792e)
![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/c33d3125-454f-49a5-b2d7-d9ac585655d1)

At momentum = 0, it's the same as the original gradient descent method. From the first graph, we see that the rmse is similar for different momentum rates and tend to converge at the same number of epochs. From the second graph, we can see that with very low momentum rate takes quite a while to converge compared to those with higher momentum.

**Part 7: Finalizing a Model**

![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/5b0889f4-bd42-406d-8cab-36016041d1b3)
![image](https://github.com/Chengalex96/APS1070_Project_4/assets/81919159/9c95c0f3-a07c-4ec5-b452-1f99362dfe8d)

Using a combination of a ranging learning rate with momentum: We saw from P6 that momentum in the middle worked best, a low momentum will take longer and a higher momentum rate may not converge. We also saw from P4 that a batch size of 128 allowed for the quickest computational time. We see that there's also an optimal learning rate, if the learning rate is too large, it requires more epochs since it may be overshooting. We see that a learning rate of 0.019 will provide us with the fastest computational time. 

This computational time is fairly quick for something as critical as an aircraft. Ideally would want to reduce the rmse as much as possible without regarding the computation time since the consequence of a mistake is severe. 

Pros of linear regression for this problem: Simple to implement and less complex, which makes it quick to calculate.
Cons on linear regression: Divergence is a large issue, it assumes independence and linear relationship between variables, which may over-simplify. 
