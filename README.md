# Long Memory Process

This repo contais a Julia script to generate realizations of a class of stationary Gaussian stochastic processes with extremely long memory.
In detail the script can be used to generate realizations $X(t)$ of a real stationary Gaussian process with correlation function

$$ \mathrm{cf}(t)=\mathcal{E}(X(t_0+t)X(t_0))=(\Gamma(y+1, 0)-\Gamma(y+1, \theta_\mathrm{max}|t|)) (\theta_\mathrm{max}|t|)^{-y-1}$$

with $\theta_\mathrm{max}>0$ and $y>-1$. $\Gamma$ is the incomplete gamma function. 
The correlation function decays algebraically for large $t$ (heavy tailed)

$$ \mathrm{cf}(t)\rightarrow \Gamma(y+1) (\theta_\mathrm{max}|t|)^{-y-1} .$$

## Theory
Standard tools to generate stationary processes can become slow when the memory time is very large. The scheme that I use here approximates the true process as a finite sum of Ornstein-Uhlenbeck processes. This may be much more efficient than convolution in the time domain (as in ARFIMA(0,d,0)) or using a Fourier method.
The advantage of the specific form for the correlation function is that it has the following integral representation

$$ \mathrm{cf}(t)= (\theta_\mathrm{max})^{-y-1} \int_0^{\theta_\mathrm{max}}\theta^y \mathrm{e}^{-\theta |t|} \mathrm{d}\theta . $$

$\theta_\mathrm{max}$ determines the fastest decay rate that we allow for and can be used to modify the short time behavior. 

When we apply a numerical quadrature scheme to the integral it reduces to a finite sum over exponentials (in $t$)

$$ \mathrm{cf}(t)\approx (\theta_\mathrm{max})^{-y-1}\sum_k w_k (\theta_k)^y \mathrm{e}^{-\theta_k |t|} $$

with weights $w_k$ and abscissas $\theta_k$.
Exponentially decaying correlation functions are realized by a stationary Ornstein-Uhlenbeck process for which the transition probability is known analytically. Thus, the numerical quadrature scheme delivers an approximation of the process as a sum over a finite number of stationary Ornstein-Uhlenbeck processes. An efficient scheme for the diverging integrand is tanh-sinh quadrature (this is used in the code).

This strategy allows for an efficient generation of realizations of stationary Gaussian processes with very long memory (i.e. $y$ close to $-1$).

## Usage

### Generation of realizations
First create a `LongMemoryProcess` struct.

```julia
lmp = LongMemoryProcess(y, θ_max)
```

To generate a single realization along a vector of times t one can then simply call

```julia
Xₜ = lmp(t)
```

### Accuracy

You can check the accuracy by comparing the true correlation function with the fitted function (that is the correlation function of the approximated process).

```julia
y = -0.9; θ_max = 1.0; t_f = 1e8;
lmp = LongMemoryProcess(y, θ_max);
relative_error = abs(cf(y, θ_max, t_f) - cf_fit(lmp, t_f)) / cf(y, θ_max, t_f)   # 9.013557696260914e-7
```

Calling `cf` requires the incomplete gamma function via SpecialFunctions.jl. If you need more speed and less accuracy you can decrease the number of Ornstein-Uhlenbeck processes with the optional argument `n` (the default value is `n = 401`).
```julia
number_of_OU_processes = 61;
lmp = LongMemoryProcess(y, θ_max, n = number_of_OU_processes);
relative_error = abs(cf(y, θ_max, t_f) - cf_fit(lmp, t_f)) / cf(y, θ_max, t_f)   # 0.0035874732157112664
```

### Advanced stuff

If you want to plug in the process as noise in a differential equation you can use an SDE solver and integrate the Ornstein-Uhlenbeck processes along. The individual Ornstein-Uhlenbeck processes obey the SDE
$$ \mathrm{d}x_k(t)=-\theta_k x_k(t)\mathrm{d}t+\sigma_k\mathrm{d}W_k(t), $$
and the full process is just their sum
$$X(t)=\sum_k x_k(t).$$
The parameters $\sigma_k$, $\theta_k$ are fields of the `LongMemoryProcess` struct.
```julia
θ = lmp.θ; σ = lmp.σ;
```
To obtain a valid initial condition for $x_k(t)$ simply generate a realization for a single time step.
```julia
X₀, x₀ = lmp(Vector([0.]); return_full=true);
X₀[end] == sum(x₀)   # true
```

## Citing

If you found this project useful for your work please directly cite this GitHub repository.
