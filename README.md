# FlagAlgebra-Graph

Flag Algebra computation in Python for graphs.

## Introduction

Flag Algebra is a framework designed by Alexander Razborov, which allows one to reason about extremal combinatorics problems together with simple arithmetics.
For instance, it allows one to prove the minimal triangle density on the graph whose edge density if $p$.

Most of the proofs in flag algebra are based on Cauchy-Schwarz inequality. 
In other words, the inequalities are result of weighted sum of $\langle v, v\rangle \ge 0$ for some vector $v$.
Such inequalities can be reformulated with SDP (Semidefinite Programming) problems, and allows fast solving.

This repository supports basic coefficient computations and problem building under flag algebra framework.

## Installation

Use `env.yaml` file specified to create conda environment.

## Usage

See notebooks in examples to see applications.
Current examples include
- Homomorphism density minimisation under edge density constraint.
- Induced homomorphism density maximisation.
- Ramsey multiplicity minimisation.