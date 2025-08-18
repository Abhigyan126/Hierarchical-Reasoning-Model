# Hierarchical Reasoning Model (HRM) — Sudoku

This repository implements the **Hierarchical Reasoning Model (HRM)** from scratch in Python using PyTorch, applied to Sudoku as a reasoning task.
> Note: The main branch contains implmementation that consumes a lots of memory, to run in envirnment with less than 22GB of ram use the branch with optimised implementation
## Overview

The Hierarchical Reasoning Model is a new approach to structured reasoning in AI. Instead of handling problems in a flat, step-by-step way, HRM operates on multiple levels of abstraction. Higher levels capture strategic planning, while lower levels work out the fine details.  

By applying HRM to Sudoku, this project highlights how the model can learn to reason over constraints and dependencies in a logical and interpretable way. The emphasis is on exploring hierarchical thinking in artificial intelligence rather than rote memorization.

## Key Features

- **HRM built from scratch** in PyTorch.
- **Hierarchical reasoning structure** that separates high-level planning from low-level execution.
- **Applied to Sudoku**, a task well-suited for showcasing structured reasoning.
- **Readable, minimal implementation** for those interested in understanding how HRM works.

## Why Sudoku?

Sudoku puzzles require balancing local moves with global consistency — exactly the kind of challenge HRM is designed to tackle. This project demonstrates how hierarchical reasoning enables an AI system to solve structured tasks with logical coherence.

## Acknowledgments

This work draws inspiration from the original research introducing HRM:  

> **“Hierarchical Reasoning Model” (2025)** — Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori. *arXiv:2506.21734*
> https://arxiv.org/abs/2506.21734
