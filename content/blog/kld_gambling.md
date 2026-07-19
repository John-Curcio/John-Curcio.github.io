+++
title = "KL Divergence and the Kelly Criterion"
date = "2026-07-18T02:09:23-04:00"
draft = true
math = true
#
# description is optional
#
# description = "Kullback-Leibler-Divergence, KL Divergence, Kelly Criterion"


tags = []
+++

# Portfolio Splitting and the Kelly Criterion

Suppose the Knicks play the Spurs tomorrow. 

A bookmaker believes $P(\text{Knicks win}) = p$. He sells two contracts, each priced at \$1:
* One that pays $\frac{1}{p}$ if the Knicks win
* One that pays $\frac{1}{1-p}$ if the Spurs win

*(Note: Zero vig $\implies$ fair value = \$1)*

A gambler believes $P(\text{Knicks win}) = q > p$. He’s broke but bold, aiming to [maximize his rate of return](https://en.wikipedia.org/wiki/Kelly_criterion). So with his last \$1 to his name, he bets:
* \$q on the Knicks
* $\$(1-q)$ on the Spurs

This is equivalent to wagering $a = \frac{q-p}{1-p}$ on the Knicks and leaving the rest in cash (a much easier formula to remember).

As the game is about to start, the gambler is feeling good. His expected log-wealth is:

$$E_q[\log(W)] = q \log\left(\frac{q}{p}\right) + (1-q) \log\left(\frac{1-q}{1-p}\right)$$

---

### Does this formula look familiar?
If you’ve taken a course in statistics or machine learning, you will recognize this as the [Kullback-Leibler (KL) Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), denoted as $D_{\text{KL}}(q \parallel p)$.

### Interpretation
* **The Kelly Criterion:** The $q, 1-q$ split is algebraically the same as staking $f^* = \frac{q-p}{1-p}$ on the Knicks per the Kelly criterion.
* **Expected Log-Return:** The bet’s expected log‑return is the Kullback-Leibler divergence between the gambler’s beliefs and the market’s beliefs.
* **Actual Payoff:** After the game, his actual wealth-multiplier equals the likelihood ratio ($\frac{q}{p}$ or $\frac{1-q}{1-p}$).

---

### Finger-Exercises

1. **Portfolio Replication:** Convince yourself that wagering $a = \frac{q-p}{1-p}$ on the Knicks and keeping the rest in cash perfectly replicates the gambler’s $q, 1-q$ split portfolio.
2. **The Bookie's Perspective:** Under the bookie’s beliefs, show that the gambler’s portfolio has fair expected value: $$E_p[W] = 1.$$

     But also show that the gambler’s expected log-wealth is:
     $$E_p[\log W]
     = p \log\left(\frac{q}{p}\right)
     + (1-p)\log\left(\frac{1-q}{1-p}\right)
     = -D_{\text{KL}}(p \parallel q) \leq 0.$$

     How can both statements be true? What, exactly, is the bookie happy about?