# Algorithmic Trading Competition Experience

## Introduction
From April 6-22, 2025, I participated in the Algorithmic Trading Competition as a personal challenge to develop my trading knowledge and enhance my Python coding and problem-solving skills.

## The Competition Structure
The competition consisted of multiple rounds featuring different challenges. Participants traded virtual seashells across various rounds, including both manual trading and algorithmic trading components. This blog focuses primarily on the algorithmic aspects of the competition, as they formed the core challenge and learning experience.

## My Background
When I began this challenge, I was midway through my FE459 Programming for Investments class, where I had gained experience coding the Black-Scholes Model to calculate implied volatility and learned other financial concepts like portfolio returns and descriptive statistics.

## Initial Challenges
As I started the competition, I quickly realized that while my academic knowledge was useful for data analysis, algorithmic trading was significantly more complex, requiring both a deeper understanding of market mechanics and more sophisticated Python skills than simply importing CSV files from Yahoo Finance.

## Getting Started
During the pre-competition tutorial, I invested considerable time studying how to write effective trading algorithms. After reviewing a Wikipedia guide that provided a basic template for accessing market information and executing trades, I created my first algorithms through trial and error. To improve my approach, I needed a way to backtest my strategies. Fortunately, Jmerle, a long-time competition participant, had developed an open-source backtesting tool that I was able to use. This allowed me to visualize my trades and better understand my strategy's performance.

## Round 1: Building on Previous Success
For the first round, my teammate discovered a GitHub repository containing the previous year's second-place submission. We noted that two of the three assets were identical to the previous competition, with only "Squid Ink" being new and more volatile. Studying this successful code proved crucial to our approach. We adopted their well-structured format with clearly defined parameter, helper methods, and execution sections, making our strategies more modular and easier to test.

Our Round 1 strategies included:

- **Rainforest Resin**: Market making at ±2 seashells from the 10,000 price point
- **Kelp**: Market making based on moving average calculations
- **Squid Ink**: Bollinger Band mean reversion strategy

## Subsequent Rounds - Still Drafting

- **Round 2**: We traded Gift Baskets with prices determined by underlying assets. Our strategy involved calculating implied price differences and trading based on synthetic basket values.
- **Round 3**: Due to exam commitments, we were unable to participate actively.
- **Round 4**: We implemented the Black-Scholes model to optimize our trading decisions.
- **Round 5**: We focused on finalizing our strategy for the last asset.

## Final Outcome
We finished the competition ranked 383 out of 12,600 participants, a respectable showing for our first attempt at algorithmic trading.
