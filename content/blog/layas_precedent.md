+++
title = "Laya's Precedent is Embarrassingly Bad, Calling This Jev Prior Art is Absurd"
date = "2026-09-21T01:51:44-04:00"
draft = false
+++

# Laya’s Precedent is Bad

Brief context:

- [Jev](https://typesafe.ai/) came out recently, offering API access to a foundation model that is 1) strong and 2) extremely convenient to integrate into applications
- An author published his own Laya model, claiming that his prior art (SalesRLAgent) went [unjustly uncredited](https://laya.convaiinnovations.com/). His post has since [circulated](https://news.ycombinator.com/item?id=49765348) [broadly](https://x.com/JFPuget/status/2101667766692384980?s=20)

I dug into SalesRLAgent over the weekend. I found serious errors and virtually zero commonality with Jev.

# Egregious Data Leakage

SalesRLAgent’s `train.py` has the eventual conversion `outcome` as a model input. You can trace it [here](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/main/train.py#L226):

```python
        # Parse metrics
        metrics = {
            ...
            'outcome': float(row.get('outcome', 0.5)),
        }
...
						messages, metrics, probability_trajectory = self._parse_conversation(self.current_conversation_idx)
...
            metrics = self.conversation_state.conversation_metrics.copy()
...
            self.conversation_state = ConversationState(
                conversation_history=history,
                embedding=embedding,
                conversation_metrics=metrics,
                turn_number=self.current_turn,
                conversion_probabilities=conv_probs
            )
...
        return np.concatenate([
            self.embedding,
            metric_values,
            turn_info,
            padded_probs
        ])
```

The information flow is: `outcome` → `metrics` → `ConversationState.conversation_metrics` → `state_vector` → policy input.

# Bizarre Application of PPO

The author [described](https://www.reddit.com/r/LocalLLaMA/comments/1kl0uvv) it as:

> a chess game kinda system for predicting sales conversion probabilities from sales conversations… Then I just trained an RL with PPO, by reducing the dimension using a linear layer and using that to do the final prediction with PPO.
> 

From a distance, I thought the role of PPO here was to have a reward model estimate conversion likelihood given state, and train the agent to recommend actions/salesspeak which maximize estimated conversion likelihood (this has a litany of problems and would probably break, but I can appreciate the concept).

It seems to be:

- An MLP over OpenAI text embeddings (plus conversation_metrics, which included the target)
- A synthetic dataset of sales conversations (I can’t say which generator revision produced it)
- RL via PPO (seems unnecessary, boils down to supervised probability regression)

The model is [rewarded](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L250-L254) based on:

$$
r_t = 1 - |\hat{p}_t - q_t|
$$

where $q_t$ is a stored annotation from the synthetic dataset, plus a [penalty](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L256-L263) for being on the wrong side of $0.5$:

```python
        # Apply higher reward/penalty at final step based on outcome
        if self.current_turn == self.max_turns - 1:
            outcome = self.conversation_state.conversation_metrics['outcome']
            # Stronger penalty for confident wrong predictions
            if outcome == 1 and predicted_prob < 0.5:
                reward -= 1.0 * (0.5 - predicted_prob)
            elif outcome == 0 and predicted_prob > 0.5:
                reward -= 1.0 * (predicted_prob - 0.5)
```

I want to believe that these probabilities were used to generate the synthetic dataset in the first place, but I can’t say for sure. And from [peeking at commit history](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/commit/36fa6dcf75a438d4727ca370157474211e818743), that seems to be false, though I can’t take this as authoritative (`generate_dataset.py` was deleted and never put back).

I don’t see any evidence that the agent learns a policy to causally affect sales conversion in the environment, under partial information (recall the leakage). In the training environment, the policy outputs a prediction, but doesn’t sample a sales intervention whose consequences are then simulated or observed.

The author claims that "the guiding brain in my system was always reinforcement learning," but it's unclear how PPO is actually helpful here.

# Jev

Jev is a generally-capable model with a strict, yet generic interface. I can’t verify much of this because it’s not open-source, but there aren’t that many ways to skin a cat, so I’ll throw in my speculation anyway:

- Souped-up bidirectional encoder with (I assume) commensurate pre-training
- Post-trained with an RL objective that rewards calibration (unknown precisely what)
- Inference engine guarantees type-safety by restricting the support of the output distribution

It may not be calibrated w.r.t. ***your*** data, but that’s another discussion.

# Shaky Comparison to Jev

From [his post](https://laya.convaiinnovations.com/), emphasis mine:

> 
> 
> 
> They proposed the ***exact same non-autoregressive decision concept*** as if it was a brand-new scientific breakthrough.
> 
> …
> 
> My earlier model used ***PPO over sequence representations to output turn-by-turn conversion trajectories*** (probabilities from 0.0 to 1.0) in vertical sales conversations. Jev generalized parallel sampling using what they called RLCD (Reinforcement Learning for Calibrated Decisions) to output confidence distributions and schema choices horizontally, charging $0.042 per million input tokens with typical response times around 150 ms.
> 

The main point of comparison here appears to be the concept of making decisions based on a non-autoregressive model, using RL.

| Point of comparison | SalesRLAgent | Jev |
| --- | --- | --- |
| Support for variable developer schemas | None whatsoever; SalesRLAgent supported predictions for a single binary outcome (poorly) | Arguably its primary selling point |
| RL for calibration | SalesRLAgent’s application of PPO seemed confused and unnecessary, let alone novel | Unknown what RLCD is precisely |
| Quality | Given the leakage, on a synthetic dataset no less, almost certainly not very good | Untested for sales conversion |
| Model architecture | Linear layer over `text-embedding-3-large` | Not public |
| Inference cost | No hosted API, reliant on OpenAI text embeddings | $0.042 per million input tokens with typical response times around 150 ms |

Open releases make scrutiny possible, which is one reason they are valuable. But the released SalesRLAgent implementation has serious flaws and, unless the author has nonpublic information about Jev’s training and architecture, little demonstrably in common.

The longer it takes for us to identify unsupported claims, the less credence we can give to legitimate open-source work, and the more we must defer to shallow reputation signals and closed-source solutions.
