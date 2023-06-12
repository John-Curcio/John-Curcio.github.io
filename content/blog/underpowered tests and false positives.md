---
title: "Underpowered Tests and False Positives"
date: 2023-02-03T10:00:00-00:00
draft: true
math: true
---

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ttest_ind
```

<!-- TODO: this isn't organized very well. I think i should state it first as a question. -->

Review of some terms:
1. p-value: $P(\text{get a test statistic this extreme}|H_0\text{ true})$
2. significance level $\alpha$: $P(\text{rejected} H_0|H_0\text{ true})$
2. Let's consider the case of $H_0: \theta = 0; H_1: \theta = \theta_1$
3. **power**: probability of discovering an effect: $P(\text{reject }H_0|H_1\text{ true})$
4. **% of discoveries that are true**: $P(H_1\text{ true}|\text{rejected} H_0)$

# Frequentist anxiety

Note that for the last two, the conditional order is flipped!

I should point out that, as frequentists, we start to freak out when we start to talk about $P(H_1\text{ true})$ and $P(H_1\text{ true}|\text{evidence})$. **It gets down to the very interpretation of probability!**

From the first line of the [wikipedia page on frequentist probability](https://en.wikipedia.org/wiki/Frequentist_probability):

> Frequentist probability or frequentism is an interpretation of probability; it defines an event's probability as the limit of its relative frequency in many trials (the long-run probability)

But if you were to somehow repeat this experiment many times, $\theta$ wouldn't randomly change between $0$ and $\theta_1$. It's either one or the other.

*TODO: I should really just cite the original source*

# Introducing a contrived situation: war on moles

* You're playing a bizarre game of whack-a-mole, with 100 holes. Deep in an underground lab, a nefarious molemaster plots to rain terror up on your garden. For every hole, the molemaster randomly flips a (possibly biased) coin - if heads, he puts a mole in the hole.
* You're onto the molemaster's scheme. You walk around your garden, and you put your ear up against each hole, and depending on how much noise you hear, you stuff a stick of dynamite in the hole. You do this for every hole independently. 
* Finally, you push down on the handle of your blasting box, and then you walk around and count how many dead moles you've found.

Each hole is like running a separate hypothesis test, testing for an effect of the same size, with the same $P(H_1\text{ true})$. And the % of dynamite craters containing a dead mole is the proportion of true positives.

The insight here is that in this situation, $P(H_1\text{ true})$ is well-defined!


# Relating false positive rate and power

Okay, suppose you're willing to say that $P(H_1\text{ true}) \in (0, 1)$ is a number that exists. We don't know what it is, but it has a value.

This is going to help us relate the true/false positive rate and power. Recall:

* **power**: probability of discovering an effect: $P(\text{reject }H_0|H_1\text{ true})$
* **% of discoveries that are true**: $P(H_1\text{ true}|\text{rejected} H_0)$

You can relate them by Bayes' Rule:

$$
P(H_1\text{ true}|\text{rejected} H_0) = \frac{
    P(\text{reject }H_0|H_1\text{ true}) \cdot P(H_1\text{ true})
}{
    P(\text{rejected }H_0)
}$$ 
expanding out the denominator:

$$
= \frac{
    P(\text{reject }H_0|H_1\text{ true}) \cdot P(H_1\text{ true})
}{
    P(\text{rejected} H_0|H_1\text{ true}) \cdot P(H_1\text{ true}) +
    P(\text{rejected} H_0|H_1\text{ false}) \cdot P(H_1\text{ false})
}$$

Recall that $\alpha = P(\text{rejected} H_0|H_0\text{ true})$ and $\text{power} = P(\text{reject }H_0|H_1\text{ true})$. Plugging these in:

$$
= \frac{
    \text{power} \cdot P(H_1\text{ true})
}{
    \text{power} \cdot P(H_1\text{ true}) +
    \alpha \cdot P(H_1\text{ false})
} \\
= \frac{
    \text{power} \cdot P(H_1\text{ true})
}{
    \text{power} \cdot P(H_1\text{ true}) +
    \alpha \cdot (1 - P(H_1\text{ true}))
}
$$


```python
power = 0.8
alpha = 0.05
p_h_1 = np.arange(0.01, 1, 0.01)

def get_true_positive_rate(alpha, power, p_h_1):
    return power * p_h_1 / ((power * p_h_1) + (alpha * (1 - p_h_1)))

true_positive_rate = get_true_positive_rate(
    power=power, 
    alpha=alpha, 
    p_h_1=p_h_1,
)

plt.plot(p_h_1, true_positive_rate)
plt.ylim([-0.02, 1.02])
plt.xlabel("$P(H_1\ true)$")
plt.ylabel("true pos rate = $P(H_1\ true|rejected\ H_0)$")
plt.title("true positive rate, as a fn of true positive rate")
plt.show()
```


    
![png](/output_2_0.png)
    



```python
power = np.arange(0.01, 1, 0.01)
alpha = 0.05
p_h_1 = 0.5 #np.arange(0.01, 1, 0.01)

true_positive_rate = get_true_positive_rate(
    power=power, 
    alpha=alpha, 
    p_h_1=p_h_1,
)

plt.plot(power, true_positive_rate)
plt.ylim([-0.02, 1.02])
plt.xlabel("power = $P(rejected\ H_0|H_1\ true)$")
plt.ylabel("true pos rate = $P(H_1\ true|rejected\ H_0)$")
plt.title("true positive rate, as a fn of power, given $P(H_1\ true) = %.2f$"%p_h_1)
plt.show()
```


    
![png](/output_3_0.png)
    



```python
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

power = np.arange(0.0, 1, 0.01)
alpha = 0.05
p_h_1 = np.arange(0.0, 1, 0.01)

power, p_h_1 = np.meshgrid(power, p_h_1)
true_positive_rate = get_true_positive_rate(
    power=power, 
    alpha=alpha, 
    p_h_1=p_h_1,
)

X = power
Y = p_h_1
Z = true_positive_rate


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15,15))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel("power = $P(rejected\ H_0|H_1\ true)$")
ax.set_ylabel("$P(H_1\ true)$")
ax.set_zlabel("true pos rate = $P(H_1\ true|rejected\ H_0)$")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```


    
![png](/output_4_0.png)
    

