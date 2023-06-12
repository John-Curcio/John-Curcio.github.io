+++
title = "Statistical Intuition for Elo Scores"
date = "2023-06-11T21:04:16-04:00"
math = true
draft = false

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

# Elo Scoring

Two players have elo scores $A$ and $B$.

With no info other than that, it's generally accepted that

$P(A\ beats\ B) = 1 / (1 + 10^{(B - A) / 400})$

If A indeed does beat B, here are the new elo scores:

* $A' = A + K \cdot (1 - P(A\ beats\ B))$

* $B' = B - K \cdot (1 - P(A\ beats\ B))$

Where $K$ is a hyperparameter, [usually something between 10 and 40](https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details:~:text=New%20players%20have%20a%20K%C2%A0%3D%C2%A040%2C%20which%20drops%20to%20K%C2%A0%3D%C2%A020%20after%2030%20played%20games%2C%20and%20to%20K%C2%A0%3D%C2%A010%20when%20the%20player%20reaches%202400.%5B31%5D).

If A and B go to a draw, this is the update rule:

* $A' = A + K \cdot (0.5 - P(A\ beats\ B))$

* $B' = B - K \cdot (0.5 - P(A\ beats\ B))$

So $A$ and $B$ are pulled towards one another.

[Chess.com gives new players an elo of 400 to start](https://support.chess.com/article/671-how-do-i-create-an-accoun#:~:text=Choose%20your%20skill,Advanced%3A%201600).

# Statistical Intuition

The first time I saw this, I thought it was gobbledygook. Then I saw this graph:

![png](/opp_elo_score.png)

Turns out, with some algebraic manipulations, one can show that the following is an equivalent system:

* Let $X_A$ be a one-hot encoding of the chess player $A$
* Let $Y$ be $1$ if A beat B, $0.5$ if they draw, and $0$ if B beat A

$P(A\ beats\ B) = 1 / (1 + e^{-\beta (X_A - X_B)})$

That looks familiar, right? It's logistic regression without an intercept term. To show these are equivalent, set $A = \frac{400}{\ln 10} \beta X_A$.

And then, for the update rule:

$ \beta' = \beta + \alpha (Y - P(A\ beats\ B))$

Ignoring draws, this is the gradient of the binary cross-entropy loss, with $\alpha$ for the learning rate.

**So the Elo rating system is stochastic gradient descent on a really, really wide logistic regression model, with a batch size of 1**. The K-factor is a scaled learning rate.

# Is this really more intuitive?

It's easy to criticise the Elo rating system. For instance, suppose you compete in a tournament, and you lose your first match against a similarly-ranked opponent. For your own ego's sake, you should hope that guy does well in the rest of the tournament - if he also beats everyone else, you can tell yourself that his original elo was an underestimate.

I believe the function of Elo scores in chess culture is similar to that of Black-Scholes in finance. Nobody takes the model as gospel; it's useful as a shorthand and social convention. Traders can compute its Greeks in their heads and can expect other traders to understand it as well. The model was so useful that its adoption led to a boom in options trading. **Anything more sophisticated would be a worse coordinating mechanism.** 

If you want a large number of people to adopt a system, and to mutually employ its inferences to interact with each other, it must be as simple as possible. Had the logistic regression model not been burned into my brain, I would find it far less intuitive than the original Elo formulation, and at any rate impossible to calculate.

It adds a new dimension to my understanding of the otherwise almost trite saying: all models are wrong, some are useful.