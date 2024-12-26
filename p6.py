import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, bernoulli

def get_user_data():
    
    data_input = input("Enter the data values separated by commas (e.g., 10, 20, 30): ")
    frequencies_input = input("Enter the corresponding frequencies separated by commas (e.g., 2, 3, 4): ")
    
    data = list(map(int, data_input.split(',')))
    freq = list(map(int, frequencies_input.split(',')))
    
    return data, freq

def plot_normal_distribution(data, freq):
    
    mean = np.mean(data)
    std_dev = np.std(data)
    
    x = np.linspace(min(data), max(data), 100)
    pdf = norm.pdf(x, mean, std_dev)

    plt.plot(x, pdf, 'r-', lw=2, label='Normal Distribution')
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()

def plot_binomial_distribution(data, freq):
    
    n = max(data)
    p = np.mean(data) / n
    
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)

    plt.bar(x, pmf, alpha=0.7, color='b', label='Binomial Distribution')
    plt.title('Binomial Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.show()

def plot_poisson_distribution(data, freq):
    
    lam = np.mean(data)
    
    x = np.arange(0, max(data)+1)
    pmf = poisson.pmf(x, lam)

    plt.bar(x, pmf, alpha=0.7, color='g', label='Poisson Distribution')
    plt.title('Poisson Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.show()

def plot_bernoulli_distribution(data, freq):
    
    success_prob = np.mean(data) / max(data)
    
    x = [0, 1]
    pmf = bernoulli.pmf(x, success_prob)

    plt.bar(x, pmf, alpha=0.7, color='purple', label='Bernoulli Distribution')
    plt.title('Bernoulli Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.show()

def analyze_distributions(data, freq):
    
    print("Analyzing Normal Distribution:")
    plot_normal_distribution(data, freq)
    
    print("Analyzing Binomial Distribution:")
    plot_binomial_distribution(data, freq)
    
    print("Analyzing Poisson Distribution:")
    plot_poisson_distribution(data, freq)
    
    print("Analyzing Bernoulli Distribution:")
    plot_bernoulli_distribution(data, freq)

data, freq = get_user_data()
analyze_distributions(data, freq)
