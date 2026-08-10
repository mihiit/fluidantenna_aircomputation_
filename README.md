# FAA-AirComp

**Energy-Efficient Fluid Antenna Array for Over-the-Air Computation: Joint Port Selection and Power Control**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Academic--Reproducibility-lightgrey)](#license)
[![Status](https://img.shields.io/badge/IEEE%20WCL-Submitted%202026-orange)](#citation)
[![Tests](https://img.shields.io/badge/tests-65%2B%20passing-brightgreen)](#running-the-test-suite)

**Mihit Nanda** — IILM University, Greater Noida, India
**Hannah Nagpall** — Texas A&M University–Kingsville, USA

*Submitted to IEEE Wireless Communications Letters (2026)*

---

## What Is This?

When many wireless devices (think: thousands of IoT sensors or edge devices in a federated-learning system) all need to send data to a central receiver at the same time, a technique called **over-the-air computation (AirComp)** lets them all transmit simultaneously and lets the receiver directly compute the aggregate (e.g., an average) from the combined signal — instead of receiving each device's data separately. This is much faster than traditional communication, but it comes with an energy cost: every device today transmits at **full power on every round**, even when a lower power level would have been perfectly sufficient to hit the required accuracy. Over thousands of rounds, this wastes a lot of battery life.

Separately, a newer antenna technology called a **fluid antenna array (FAA)** lets the receiving base station physically reposition its antenna ports to find better channel conditions — but prior work using this only optimized *where* the antenna sits, never *how much power* each device should use. Power was always left fixed at maximum.

**This project combines both ideas for the first time**: it jointly decides (1) where to position the antenna ports, and (2) how much power each device should transmit, to meet a required accuracy target using as little total energy as possible. We built an optimization algorithm (block coordinate descent, with a proven convergence guarantee) that solves this jointly, and showed — both mathematically and through simulation — that this combination saves substantially more energy (up to ~54%) than either idea alone.

This repository is the full, reproducible codebase behind that result: every number, figure, and claim in the paper can be regenerated from this code.

---

## Overview

This repository is the complete, self-contained reproducibility package for a Fluid Antenna Array (FAA)-enhanced Over-the-Air Computation (AirComp) system. It jointly optimizes **port selection** and **power control** to minimize total transmit energy across devices, using a four-step **Block Coordinate Descent (BCD)** algorithm with a provable monotone-convergence guarantee.

The code in this repository will let you:

- **Reproduce all 6 figures** from the paper exactly as submitted
- **Run the full BCD simulation** with verified monotone convergence
- **Verify the key numerical claims** of the paper (energy savings ratio, scaling-law exponents, R²)
- **Run the full test suite** (65+ unit and integration tests)
- **Recompile the paper PDF** from LaTeX source

All results are deterministic — fixed random seeds produce identical outputs across runs and machines.

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [System Parameters](#system-parameters-table-ii)
- [Algorithm Summary](#algorithm-summary)
- [Key Results](#key-results)
- [Recompiling the Paper](#recompiling-the-paper)
- [Module API Reference](#module-api-reference)
- [Running the Test Suite](#running-the-test-suite)
- [Citation](#citation)
- [License](#license)

---

## Repository Structure
