# Hodgkin-Huxley Neuron Model Simulator

A Python implementation of the **Hodgkin-Huxley model** — the seminal biophysical model of action potential generation in neurons, awarded the Nobel Prize in Physiology or Medicine (1963).

## Overview

This simulation models the electrical activity of a neuron membrane by numerically integrating four coupled differential equations that describe:

- **V** — membrane potential (mV)
- **n** — K⁺ channel activation gate
- **m** — Na⁺ channel activation gate
- **h** — Na⁺ channel inactivation gate

The numerical integration uses a **4th-order Runge-Kutta (RK4)** method for accuracy and stability.

## Physics of the Model

The membrane potential evolves according to:

```
Cm · dV/dt = I_ext - g_Na·m³·h·(V - E_Na) - g_K·n⁴·(V - E_K) - g_L·(V - E_L)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Cm` | 1.0 µF/cm² | Membrane capacitance |
| `g_Na` | 120.0 mS/cm² | Max Na⁺ conductance |
| `g_K` | 36.0 mS/cm² | Max K⁺ conductance |
| `g_L` | 0.3 mS/cm² | Leak conductance |
| `E_Na` | +50 mV | Na⁺ reversal potential |
| `E_K` | −77 mV | K⁺ reversal potential |
| `E_L` | −54.387 mV | Leak reversal potential |

## Features

- Full RK4 numerical integration of the HH equations
- Live animated plot showing membrane potential and gating variables
- Sweep of external current `I_ext` from 0 to 50 µA/cm² to visualize the **firing threshold**

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

## Usage

```bash
python NN/hodgkin_huxley_neuro.py
```

The simulation will open an animated window sweeping through 100 values of external current. You will observe:
- **Sub-threshold** regime: membrane potential returns to rest
- **Action potential** generation: stereotypical spike when `I_ext` crosses the threshold (~6.3 µA/cm²)
- **Repetitive firing** at higher stimulation intensities

## Output

The live plot displays:
- `V × 0.05` (purple) — scaled membrane potential
- `n` (green) — K⁺ activation
- `m` (red) — Na⁺ activation
- `h` (blue) — Na⁺ inactivation

## References

- Hodgkin, A.L. & Huxley, A.F. (1952). *A quantitative description of membrane current and its application to conduction and excitation in nerve.* Journal of Physiology, 117(4), 500–544.
