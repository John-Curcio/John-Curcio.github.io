+++
title = "Laya's Prior Art Claim is Absurd"
date = "2026-09-22T11:11:44-04:00"
draft = false
+++

Brief context:

- [Jev](https://typesafe.ai/) came out recently, offering API access to a closed, API-only model that answers typed decision questions (developer-specified schemas) in a single pass, with probabilities post-trained for calibration.
- SalesRLAgent ([arXiv](https://arxiv.org/abs/2503.23303), [HF repo](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning)) is an earlier model by Nandakishor Mukkunnoth that predicts sales-conversion probability from sales conversations.
- Laya is an open-weights, Jev-compatible alternative, released 3 days after Jev. In its launch post, Mukkunnoth claims SalesRLAgent was prior art for Jev and went [unjustly uncredited](https://laya.convaiinnovations.com/). His claim has since [circulated](https://news.ycombinator.com/item?id=49765348) [broadly](https://x.com/JFPuget/status/2101667766692384980?s=20), and many actually seem [confused](https://news.ycombinator.com/item?id=49803157) over [precedence](https://x.com/nedwize/status/2102020796516647050?s=20).

I dug into SalesRLAgent over the weekend. I found serious errors and nothing uniquely in common with Jev.

# Multiple Instances of Data Leakage

SalesRLAgent’s `train.py` has the eventual conversion `outcome` as a model input. The information flow is: [`row['outcome']`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L170) → [`metrics`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L166-L172) → [`ConversationState.conversation_metrics`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L235-L241) → [`metric_values`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L63) → [`state_vector`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L71-L76) → [observation returned by `reset`](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L243). Each `step` [copies `self.conversation_state.conversation_metrics` forward](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L272) into the [next state](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L282-L288), so `outcome` is in [every observation](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/blob/fa341daeefc0fe2843e2df839ab52dd78cab1dc0/train.py#L290) of the episode.

A few more things I caught:
* At turn 0, the agent sees an embedding of the entire conversation (including its ending).
* `conversion_probabilities` is initialized with `true_probabilities[0]`, so the ground-truth annotation $q_0$ (see next section) leaks into the state.

I've been guilty of data leakage before and I surely will be again, but it is and forever will be a serious mistake.

# Unnecessary Invocation of PPO

The author [described](https://www.reddit.com/r/LocalLLaMA/comments/1kl0uvv) it as:

> a chess game kinda system for predicting sales conversion probabilities from sales conversations… Then I just trained an RL with PPO, by reducing the dimension using a linear layer and using that to do the final prediction with PPO.
> 

SalesRLAgent seems to be:

- An MLP over OpenAI text embeddings (plus conversation_metrics, which included the target)
- Trained on a synthetic dataset of sales conversations (I can’t say which generator revision produced it)
- RL via PPO?

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

If $q_t$ is a latent probability used to generate the synthetic data, then regressing on it is at least a coherent supervised target. From [peeking at commit history](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/commit/36fa6dcf75a438d4727ca370157474211e818743), that seems not the case here, though I can’t take this as authoritative (`generate_dataset.py` was deleted and never put back, best I've got).

Does the agent learn by intervening in its environment? No, it's **just doing regression**: In the training environment, the policy outputs a prediction, which doesn't affect the progression of the sales conversation.

The author [claims](https://laya.convaiinnovations.com/#:~:text=The%20guiding%20brain%20in%20my%20system%20was%20always%20reinforcement%20learning%2C%20not%20just%20an%20embedding%20model%20or%20an%20autoregressive%20LLM.) that "the guiding brain in my system was always reinforcement learning," but **it's unclear why PPO is here at all**.

# Jev

TypeSafe describes Jev as a generally-capable model with a strict, yet generic interface. It guarantees type-safety by restricting the support of the output distribution, and was apparently post-trained with an RL objective that rewards calibration. It may not be calibrated w.r.t. **your** data but that's another discussion.

We can't verify the specific objective or architecture, as it's not open-source. I assume the type-safety trick works by constrained decoding, which has been around since at least [2021](https://arxiv.org/abs/2109.05093) but has admittedly gone under-exploited.

# Comparison to Jev

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

The main point of comparison here appears to be the concept of making decisions based on a non-autoregressive model, using RL. SalesRLAgent's application of PPO cannot be credibly described as RL; I can trivially wrap any scalar regression problem in an RL environment by treating the prediction as an action and the instance loss as a reward.

| Point of comparison | SalesRLAgent | Jev |
| --- | --- | --- |
| Support for variable developer schemas | None; SalesRLAgent supported predictions for a single binary outcome | Arguably its primary selling point |
| RL for calibration | Reward not a proper scoring rule, PPO unnecessary | Unknown what RLCD is precisely |
| Emphasis on sales | Sales conversations only | None |
| Model architecture | MLP over `text-embedding-3-large` | Not public |
| Inference cost | No hosted API, reliant on OpenAI text embeddings | $0.042 per million input tokens with typical response times around 150 ms |

Open releases make scrutiny possible, which is one reason they are valuable. But the released SalesRLAgent implementation has disappointing flaws which are immediately obvious upon inspection; I find it hard to believe that this could have inspired Jev. Its flaws aside, unless the author has nonpublic information about Jev's training and architecture, they have nothing uniquely in common.