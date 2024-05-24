""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def task1():
    """ Subtask 1: Approximating Kernels

        Requirements for the plot:
        - the first row corresponds to the task in 1.1 and the second row to the task in 1.2

        for each row:
        - the first subplot should contain the Kernel matrix with each entry K_ij for k(x_i,x_j')
        - the second to fifth subplot should contain the corresponding feature approximation when using 1,10,100,1000 features
    """

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(15, 8)
    font = {'fontsize': 18}

    feat = ['Fourier', 'Gauss']
    for row in range(2):
        axes[row, 4].set_title('Exact kernel', **font)
        axes[row, 4].set_xticks([])
        axes[row, 4].set_yticks([])

        axes[row, 0].set_ylabel('%s features' % feat[row], **font)
        for col, R in enumerate([1, 10, 100, 1000]):
            axes[row, col].set_title(r'$\mathbf{Z} \mathbf{Z}^{\top}$, $R=%s$' % R, **font)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # generate random 2D data
    N = 1000
    D = 2

    X = np.ones((N, D))
    X[:, 0] = np.linspace(-3., 3., N)
    X[:, 1] = np.sort(np.random.randn(N))

    """ Start of your code 
    """
    np.random.seed(42)
    #####################################################################################
    # 1.1
    def gaussian_kernel11(x, y, sigma=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    # exact kernel matrix
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel11(X[i], X[j])

    axes[0, 4].imshow(K, cmap='viridis')

    # Random Fourier Features approximation
    Rs = [1, 10, 100, 1000]
    for col, R in enumerate(Rs):
        omega = np.random.normal(0, 1, (R, D))
        b = np.random.uniform(0, 2 * np.pi, R)
        Z_Fourier = np.sqrt(2.0 / R) * np.cos(np.dot(X, omega.T) + b)
        K_Fourier = np.dot(Z_Fourier, Z_Fourier.T)
        axes[0, col].imshow(K_Fourier, cmap='viridis')

    #####################################################################################
    # 1.2
    def gaussian_kernel12(x, y, sigma=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (4 * sigma ** 2))  # Note the division by 4*sigma^2

    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel12(X[i], X[j])

    axes[1, 4].imshow(K, cmap='viridis')

    # Gaussian Random Features approximation
    Rs = [1, 10, 100, 1000]
    for col, R in enumerate(Rs):
        t_indices = np.random.randint(0, N, size=R)
        t_samples = X[t_indices, :]

        Z_Gauss = np.array([np.sqrt(1.0 / R) * np.exp(-np.linalg.norm(X - t, axis=1) ** 2 / 2) for t in t_samples]).T
        K_Gauss = np.dot(Z_Gauss, Z_Gauss.T)
        axes[1, col].imshow(K_Gauss, cmap='viridis')

    """ End of your code 
    """

    return fig


def task2():
    """ Subtask 2: Linear Regression with Feature Transforms

        Requirements for the plot:
        - the left and right subplots should cover the cases with random Fourier and Gauss features, respectively

        for each subplot:
        - plot the averaged (over 5 runs) mean and standard deviation of training and test errors over the number of features
        - include labels for the curves in a legend
    """

    def gen_data(n, d):
        sig = 1.

        v_star = np.random.randn(d)
        v_star = v_star / np.sqrt((v_star ** 2).sum())

        # create input data on unit sphere
        x = np.random.randn(n, d)
        x = x / np.sqrt((x ** 2).sum(1, keepdims=True))

        # create targets y
        y = np.zeros((n))
        for n_idx in np.arange(n):
            y[n_idx] = 1 / (0.25 + (x[n_idx]).sum() ** 2) + sig * np.random.randn(1)

        return x, y

    n = 200
    n_test = 100
    D = 5

    x_, y_ = gen_data(n + n_test, D)
    idx = np.random.permutation(np.arange(n + n_test))
    x, y, x_test, y_test = x_[idx][:n], y_[idx][:n], x_[idx][n::], y_[idx][n::]

    # features
    R = np.arange(1, 100)

    # plot
    fig2, ax = plt.subplots(1, 2)
    ax[0].set_title('Random Fourier Features')
    ax[0].set_xlabel('features R')

    ax[1].set_title('Random Gauss Features')
    ax[1].set_xlabel('features R')

    """ Start of your code 
    """
    alpha = 3.0 # Regularization parameter

    def LSR_regularized_fit(Phi, y_labels, alpha):
        Phi_T = Phi.T
        n_features = Phi.shape[1]
        I = np.eye(n_features)
        theta = np.linalg.pinv(Phi_T @ Phi + alpha * I) @ Phi_T @ y_labels
        return theta

    def LSR_regularized_predict(Phi, theta):
        return Phi @ theta

    def compute_errors(Phi_train, Phi_test, y_train, y_test, alpha):
        theta = LSR_regularized_fit(Phi_train, y_train, alpha)
        train_error = 1 / y_train.shape[0] * np.sum((LSR_regularized_predict(Phi_train, theta) - y_train) ** 2)
        test_error = 1 / y_test.shape[0] * np.sum((LSR_regularized_predict(Phi_test, theta) - y_test) ** 2)
        return train_error, test_error

    def random_fourier_features(X, r):
        omega = np.random.normal(0, 1, (r, D))
        b = np.random.uniform(0, 2 * np.pi, r)
        Phi = np.sqrt(2.0 / r) * np.cos(np.dot(X, omega.T) + b)
        return Phi

    def random_gauss_features(X, r):
        t_indices = np.random.randint(0, X.shape[0], size=r)
        t_samples = X[t_indices, :]
        Phi = np.array([np.sqrt(1.0 / r) * np.exp(-np.linalg.norm(X - t, axis=1) ** 2 / 2) for t in t_samples]).T
        return Phi

    def task26(feature_function):
        train_errors = []
        test_errors = []
        for r in R:
            train_errors_r = []
            test_errors_r = []
            for _ in range(5):  # Averaging over 5 runs
                Phi_train = feature_function(x, r)
                Phi_test = feature_function(x_test, r)
                train_error, test_error = compute_errors(Phi_train, Phi_test, y, y_test, alpha)
                train_errors_r.append(train_error)
                test_errors_r.append(test_error)
            train_errors.append((np.mean(train_errors_r), np.std(train_errors_r)))
            test_errors.append((np.mean(test_errors_r), np.std(test_errors_r)))
        return train_errors, test_errors

    def plot_errors(ax, R, train_errors, test_errors, title):
        ax.set_title(title)
        ax.set_xlabel('Number of features (R)')
        ax.set_ylabel('Error')
        train_mean = [e[0] for e in train_errors]
        train_std = [e[1] for e in train_errors]
        test_mean = [e[0] for e in test_errors]
        test_std = [e[1] for e in test_errors]

        ax.plot(R, train_mean, label='Train Error')
        ax.fill_between(R, np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        alpha=0.2)
        ax.plot(R, test_mean, label='Test Error')
        ax.fill_between(R, np.array(test_mean) - np.array(test_std),
                        np.array(test_mean) + np.array(test_std),
                        alpha=0.2)
        ax.legend()

    train_errors_fourier, test_errors_fourier = task26(random_fourier_features)
    train_errors_gauss, test_errors_gauss = task26(random_gauss_features)

    plot_errors(ax[0], R, train_errors_fourier, test_errors_fourier, 'Random Fourier Features')
    plot_errors(ax[1], R, train_errors_gauss, test_errors_gauss, 'Random Gauss Features')

    """ End of your code 
    """

    ax[0].legend()
    ax[1].legend()

    return fig2


if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    fig1 = task1()
    fig2 = task2()
    pdf.savefig(fig1)
    pdf.savefig(fig2)

    pdf.close()
