import SpecialFunctions.gamma


"""
    stationary_ornstein_uhlenbeck_realization(θ::Vector{Float64}, σ::Vector{Float64}, t::Vector{<:Number})
Generate a realization of a stationary Gaussian stochastic which is the sum of `N = length(θ)` real independent Ornstein-Uhlenbeck processes (with parameters `θ`, `σ`) evaluated at times `t`.


# Output
Returns a realization of the process evaluated at times `t`

It is also possible to additionally return the values of the individual processes at the final time by setting the optional argument `return_full=true`
"""
function stationary_ornstein_uhlenbeck_realization(θ::Vector{Float64}, σ::Vector{Float64}, t::Vector{<:Number}; return_full::Bool=false)
    @assert length(θ) == length(σ) "σ and θ do not have the same length"
    @assert all(θ .>= 0) "θ has negative entries"
    @assert all(σ .>= 0) "σ has negative entries"
    xₜ = (@. σ / sqrt(2 * θ)) .* randn(length(θ))
    Xₜ = Vector(zeros(length(t)))
    Xₜ[1] = sum(xₜ)
    for i in 2:length(t)
        dt = (t[i] - t[i-1])
        xₜ = (@. sqrt(σ^2 / (2θ) * (1 - exp(-2 * θ * dt)))) .* randn(length(σ)) .+ (@. xₜ * exp(-θ * dt))
        Xₜ[i] = sum(xₜ)
    end
    return return_full ? (Xₜ, xₜ) : Xₜ
end


"""
    TanhSinhQuad(h::Float64, k::Array{Int64})
Constructor for `TanhSinhQuad` struct given stepsize `h` and number of integration steps `k`. 


# Example
To compute an integral from -1 to 1 of a function f(x) using tanh-sinh scheme one can do the following.
```
    thq = TanhSinhQuad(h, k)
    integral = sum(@. thq.w * f(thq.x))
```
"""
mutable struct TanhSinhQuad
    h::Float64
    k::Array{Int64}
    x::Array{BigFloat}
    w::Array{BigFloat}
end
function TanhSinhQuad(h::Float64, k::Array{Int64})
    x = @. tanh(0.5 * π * sinh(k * h * BigFloat(1.0)))
    w = @. 0.5 * h * π * cosh(k * h) / (cosh(0.5 * π * sinh(k * h * BigFloat(1.0))))^2
    return TanhSinhQuad(h, k, x, w)
end


"""
    LongMemoryProcess(y::AbstractFloat, θ_max::Number; n::Int=401)
`LongMemoryProcess` can generate a class of stationary Gaussian processes with long memory algebraically decaying correlations. `y` must be larger than -1.
The correlation function of the process cf(t) = E(X(t)X(0)) is given as

    cf(y, θ_max, t) = (Γ(y + 1, 0) - Γ(y + 1, θ_max t)) / (θ_max t)^(y + 1)

where Γ is the incomplete gamma function. The algorithm approximates the process with a finite number of Ornstein-Uhlenbeck processes by discretizing an integral representation of cf. `n` is the number of Ornstein-Uhlenbeck processes used for the simulation (larger `n` yields better accuracy). 

# Usage example
First construct the struct.
```
    lmp = LongMemoryProcess(y, θ_max)
```
To generate a realization along a vector of times t one can then simply call:
```
    Xₜ = lmp(t)
```
You can check the fitting accuracy by comparing the true correlation function with the fitted function (that is the correlation function of the approximated process).
```
    cfₜ = [cf(y, θ_max, t_) for t_ in t]
    cf_fitₜ = [cf_fit(lmp, t_) for t_ in t]
```
"""
mutable struct LongMemoryProcess
    th::TanhSinhQuad
    y::AbstractFloat
    θ::Vector{Float64}
    σ::Vector{Float64}
end
function LongMemoryProcess(th::TanhSinhQuad, y::AbstractFloat, θ_max::Number)
    @assert y > -1 "y must be larger than -1"
    f(x) = x^y
    g = Float64.(@. th.w / θ_max^y * 0.5 * f((th.x + 1) / 2 * θ_max))
    θ = @. Float64((th.x + 1) / 2 * θ_max)
    σ = sqrt.(2 * θ .* g)
    return LongMemoryProcess(th, y, θ, σ)
end
function LongMemoryProcess(y::AbstractFloat, θ_max::Number; n::Int=401)
    @assert y > -1 "y must be larger than -1."
    h = 4.6 / floor(n / 2)
    # 4.6 this is the upper limit for using standard BigFloat accuracy. This is fine for Y > -0.95.
    k = collect(-Int(floor(n / 2)):Int(floor(n / 2 - 1 / 2)))
    th = TanhSinhQuad(Float64(h), k)
    return LongMemoryProcess(th, y, θ_max)
end
function (proc::LongMemoryProcess)(t::Vector{<:Number}; return_full=false)
    return stationary_ornstein_uhlenbeck_realization(proc.θ, proc.σ, t; return_full=return_full)
end


"""
    cf_fit(proc::LongMemoryProcess, t::Number)
The fitted correlation function of the process 'proc' evaluated at time 't'.
"""
function cf_fit(σ::Vector{Float64}, θ::Vector{Float64}, t::Number)
    return sum(@. σ^2 / (2 * θ) * exp(-θ * abs(t)))
end
function cf_fit(proc::LongMemoryProcess, t::Number)
    return cf_fit(proc.σ, proc.θ, t)
end


"""
    cf(y::AbstractFloat, θ_max::Number, t::Number)
The function 
    cf(y, θ_max, t) = (Γ(y + 1, 0) - Γ(y + 1, θ_max t)) / t^(y + 1)
"""
function cf(y::AbstractFloat, θ_max::Number, t::Number)
    return (gamma(y + 1, 0.0) - gamma(y + 1, θ_max * t)) / (θ_max * t)^(y + 1)
end