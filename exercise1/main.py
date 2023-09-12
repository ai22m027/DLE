import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30])
Y = np.array([4, 7, 16, 32, 50, 90, 115, 119, 142, 166])

def plot_trajectory(x, y, y_predicted=None):
    fig, ax = plt.subplots()

    ax.plot(x, y, label='Actual Trajectory', color='blue')

    if y_predicted is not None:
        ax.plot(x, y_predicted, label='Predicted Trajectory', color='red')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.title('Trajectory Plot')
    plt.ion()
    plt.show()
    plt.ioff()
    
def loss_plot(no_epochs, y):
    fig, ax = plt.subplots()
    x = list(range(1, no_epochs + 1))

    ax.plot(x, y, label='epoch loss', color='blue')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

    plt.title('Epoch loss')
    plt.ion()
    plt.show()
    plt.ioff()


def mm_model(theta1, theta2, x):
    return (theta1 * x) / (theta2 + x)

def squared_deviation_loss(theta1: float, theta2: float, x_data: float, y_data: float) -> float:
    N = len(x_data)
    predictions = [mm_model(theta1, theta2, x) for x in x_data]
    loss = sum([(y - y_hat) ** 2 for y, y_hat in zip(y_data, predictions)]) / (2 * N)
    return loss

def gradient(theta1: float, theta2: float, x_data: float, y_data: float):
    N = len(x_data)
    
    y_pred = (theta1 * x_data) / (theta2 + x_data)
    
    gradient_theta1 = 2 * np.mean((y_pred - y_data) * x_data / (theta2 + x_data))
    gradient_theta2 = 2 * np.mean((y_pred - y_data) * (-theta1 * x_data / (theta2 + x_data)**2))
    return gradient_theta1, gradient_theta2

def gradient_descent(x_data: float, y_data: float, theta1_init: float, theta2_init: float, learning_rate: float, num_iterations: float):
    theta1 = theta1_init
    theta2 = theta2_init
    epoch_loss_list = []

    for i in range(num_iterations):
        grad_theta1, grad_theta2 = gradient(theta1, theta2, x_data, y_data)
        theta1 -= learning_rate * grad_theta1
        theta2 -= learning_rate * grad_theta2
        epoch_loss_list.append(squared_deviation_loss(theta1, theta2, x_data, y_data))
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}: Loss = {epoch_loss_list[-1]}")

    return theta1, theta2, epoch_loss_list


def main():
    # Steps
    # 1. define the mm model
    # 2. derive the mm model
    # 3. add the squared loss fct
    # 4. compute gradients (use fct from 2.)
    # 5. impl. gradient descent
    # 6. keep track of thetas and epoch loss
    
    # Hyperparameters
    init_theta1 = 10.0
    init_theta2 = 10.0
    
    learning_rate = 0.1
    max_epochs = 30000

    # GD alog
    final_theta1, final_theta2, epoch_loss_list = gradient_descent(X, Y, init_theta1, init_theta2, learning_rate, max_epochs)

    # y_pred calculation
    y_pred_list = []
    for x,y in zip(X,Y):
        y_pred = mm_model(final_theta1, final_theta2, x)
        y_pred_list.append(y_pred)
        print(f"ACT: {y} | PRED: {y_pred}")
    
    # Calculate the final loss
    final_loss = squared_deviation_loss(final_theta1, final_theta2, X, Y)
    print(f"Final loss: {final_loss}, using theta_1 = {final_theta1} and theta_2 = {final_theta2}.")
    pass

    # compare predicted with actual
    plot_trajectory(X,Y, y_pred_list)
    # plot the epoch loss
    loss_plot(max_epochs, epoch_loss_list)
    pass

if __name__ == "__main__":
    main()