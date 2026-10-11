# Regime Detection in Danish Equities with a Hidden Markov Model

A Gaussian HMM fitted to the OMX Copenhagen General Index from 2013 to 2024, with a working implementation of the distinction that decides whether regime models are usable at all: **filtered against smoothed state probabilities**.

---

## The problem this repository is actually about

Most regime-detection work shows a chart of regime probabilities over the sample and stops there. That chart is almost always produced by the forward-backward algorithm, which conditions on the *entire* sample, including everything that happened after the date it is labelling.

In `hmmlearn`, `predict_proba()` returns that smoothed posterior and `predict()` returns the Viterbi path. Both use the full sample. Neither function name says so.

The consequence is systematic and hard to spot. The smoothed path identifies the onset of the 2020 and 2022 drawdowns almost exactly, because it has seen what followed. A backtest that allocates on that signal exits before every crash, in every crash. The equity curve looks like skill. It is unachievable in principle, and because a smoothed chart looks exactly like a filtered one, nothing in the output signals the problem.

What a decision at time *t* can use is the **filtered** probability, conditioned only on data through *t*:

```
alpha_1(i) = pi_i b_i(x_1)
alpha_t(j) = [ sum_i alpha_{t-1}(i) a_ij ] b_j(x_t)
```

normalised at each step to prevent underflow. The normalised alpha is the filtered posterior, and the sum of log normalisers recovers the log-likelihood, which gives a free correctness check: the hand-rolled recursion must agree with `model.score()` to machine precision or the implementation is wrong.

This repository implements that recursion directly and reports the gap between the two as a measured quantity rather than a caveat.

## What is in it

**Sections 1 to 2** build the feature matrix (daily log return plus 21-day realised volatility) and document the stylized facts that motivate the model: leptokurtosis, volatility clustering in the ACF of squared returns, and rolling parameter instability.

**Section 3** specifies the model and derives forward-backward and Baum-Welch, with the three modelling assumptions stated as restrictions rather than boilerplate. The Markov assumption forces geometric sojourn times, so the model cannot express a regime whose hazard of ending depends on how long it has already lasted.

**Section 4** fits it. State count examined by BIC with **25 random restarts at each K**, then fixed at three after BIC ran to the edge of its range and the log-likelihood spread across restarts reported, because Baum-Welch is EM and converges to a local maximum of a multimodal surface. Label switching is resolved by sorting states on mean return, which matters the moment the model is refitted on a different window. Transition matrix, stationary distribution and implied expected durations follow.

**Section 5** is the core. Filtered against smoothed, plotted on the same axis with their difference shaded, plus **detection lag**: for each crisis episode the smoothed path identifies, how many trading days until the filtered probability crossed one half, and how much of the drawdown had already happened by then. That number decides whether the model is a tool or a historian.

**Section 6** removes the remaining advantage. Filtering fixes the timing of the data but not of the parameters; the full-sample fit still used knowledge of the holdout. The model is refitted on 2013 to 2018 data only, the parameters are frozen, and the filter runs forward through 2024, so it must recognise the COVID crash using states estimated on milder stress.

**Section 7** builds an allocation on filtered signals only, with training-period parameters, a one-day lag and 10 bps of transaction cost, then runs the identical rule on smoothed probabilities as a counterfactual on the same chart. That second curve cannot be implemented by anyone. It is included because it is the result that gets published by accident.

**Section 8** lists what is wrong with the model.

## Results

Three states, fixed a priori (BIC ran to the boundary of its search range, which is evidence of misspecification rather than a selection; see section 4.1).

| State | Annualised return | Realised volatility | Expected duration |
|---|---|---|---|
| Calm | +19% | 11% | 51 days |
| Normal | +21% | 16% | 26 days |
| Stress | -7% | 25% | 33 days |

Out of sample, 2019 to 2024, with parameters frozen at the end of 2018:

| | Buy and hold | Filtered, lagged, net of costs | Smoothed (not implementable) |
|---|---|---|---|
| Sharpe ratio | 0.74 | 0.84 | 1.39 |
| Maximum drawdown | -27.4% | -28.0% | -15.2% |

The honest strategy is a de-risking overlay: lower volatility, slightly lower return, no reduction in maximum drawdown, and a Sharpe improvement well inside sampling noise. The smoothed column is the same rule with access to the future. The gap between the last two columns is the point of the repository.

The state classification itself holds up out of sample as a volatility signal: next-day volatility after a stress-state reading was 22.6%, against 13.6% after a calm one.

### A labelling bug, found and disclosed

The first run sorted states by mean return and treated the lowest-return state as the crisis state. With six states that was not the high-volatility state, the crisis diagnostics measured the wrong object, and the strategy returned a Sharpe ratio of 0.21. States are now ordered by realised volatility. The change to three states was made after that result was seen; the notebook says so, and the justification is the boundary solution and the labelling error rather than performance.

## Limitations

Gaussian emissions still understate the tail inside a crisis state; Student-*t* emissions are the natural extension. The 21-day realised volatility observable is backward looking, so part of the measured detection lag is mechanical rather than inferential. *K* is chosen from a candidate set, not discovered. The sample starts in 2013 because that is where Yahoo's history for the index begins, so there is no 2008 in it and the crisis state is estimated mostly from the COVID crash; I chose that over a four-stock composite whose single-name scandals would read as market regimes. One index, one country, and Danish equities are concentrated enough that the benchmark is closer to a single-name proxy than a diversified index over much of the later sample.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook HMM_Regime_Change.ipynb
```

Fitting with 25 restarts across five values of K takes roughly a minute.

## References

Rabiner (1989), *A tutorial on hidden Markov models and selected applications in speech recognition*.
Hamilton (1989), *A new approach to the economic analysis of nonstationary time series and the business cycle*.
Ang and Bekaert (2002), *International asset allocation with regime shifts*.

## Author

High_Magus_of_Risk.
