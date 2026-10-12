# Knowing When to Think Fast or Slow at Test Time

*Tinevimbo Musingadi · October 2026 · [interactive version with all figures](https://tinevimbomusingadi.github.io/kwtfs-blog/)*

## § 1 Introduction

Ask a person whether 5,975 is bigger than 5,974 and they answer instantly. Ask them how many parcels are left after two depots each receive 6,063 and dispatch 3, and they slow down and work it out. Kahneman called these System 1 and System 2 [1]. What matters for us is less the two systems than the switch between them. Thinking slowly costs effort, so people only do it when the fast answer is not good enough.

Language models have mostly lost that switch. Over the last two years the frontier slid into *always* reasoning. Reinforcement learning with verifiable rewards taught models that writing a chain of thought before answering pays off [2, 3], and test-time compute became something to scale [4]. That works, and it also means a model now writes paragraphs to tell you that 2 + 3 = 5 [5]. The industry's answers so far are a switch the *user* flips [6], or a separate router that sends each request to a fast model or a thinking model [7].

This post is about a narrower question, which I spent the last weeks working on with a small model and a few dollars of spot TPU time:

> Can one small model learn, *question by question*, whether to decide in a single forward pass, answer in a few words, or reason first, all inside the same network?

The model is Qwen3-4B with a LoRA adapter [8, 9], trained with supervised fine-tuning and then GRPO [10] using Google's Tunix library [11] on spot TPU v5e chips. It answers every question by first writing a mode tag:

- `<mode:jev>` means *decide now*. The runtime does not generate any text. It runs one forward pass and reads the answer off the logits of the option letters.
- `<mode:direct>` means *answer in a few words*.
- `<mode:cot>` means *reason first*. The model writes a short chain of thought and then either states an answer or hands off to a one-step decision.

![**Figure 1. One network, three ways to answer.** Real questions from the sealed test set and the trained model's own out](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-hero.png)
***Figure 1. One network, three ways to answer.** Real questions from the sealed test set and the trained model's own outputs. The first token the model writes picks the mode, and the runtime then executes that mode natively. In the one-step mode nothing is generated: a single forward pass scores the option letters. Use the buttons to replay each mode.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-hero)

Here is what I found, in the order the post explains it:

1. **One network can host the fast decision and the slow reasoning, and learn when to use which.** A one-step decision is just a constrained next-token step in the model's own vocabulary, so ordinary token-level RL trains reasoning, decisions and the choice between them together ([§3](#idea)).
2. **The routing labels have to be measured on the model, not assigned.** My first attempt labelled each question with the mode it "should" use, decided when the data was generated, and learned routing ended up 10.2 points *below* always reasoning. In v2 I ran Qwen3-4B on every question in every mode and took the mode with the best accuracy minus token cost ([§4](#v1), [§5](#measure)).
3. **The model largely knows in advance whether it can answer without thinking.** A linear probe on its hidden state, read before any token is generated, predicts each mode's success with AUROC 0.89–0.90. As a router on its own, though, the probe is mediocre ([§7](#probe)).
4. **Learned routing matches always-reasoning accuracy with about 31% fewer tokens.** Averaged over five RL seeds on 799 sealed test questions: 88.6% accuracy at 54.4 tokens, against 88.5% at 78.5 tokens when the same weights are forced to reason every time ([§10](#learned)).
5. **A reward for the gap between the chosen mode and the measured best mode makes RL stable.** With it, five seeds land at 88.6% ± 0.3 points. With an outcome-only reward, the same five seeds spread from 80.3% to 88.4%: some runs slide into under-thinking ([§9](#reward), [§10](#learned)).
6. **Whether it is "worth it" depends on what a token costs, and it never loses.** Pricing tokens the way training did, learned routing beats always reasoning by a small but significant margin. At higher prices the gap widens steadily ([§11](#worth)).

Much of the interesting material is in what did *not* work: an RL run that collapsed, labels that lied, a probe that predicted well and routed badly, and two rewards that did nothing. I have kept those in the story where they happened, because they explain why the final recipe looks the way it does. Every number on this page is generated from the experiment outputs. None is typed by hand.[^1]

## § 2 Background: how models spend thinking

### 2.1 Thinking is just more tokens

An autoregressive language model does a fixed amount of computation per token: one forward pass through every layer. A model that answers "C" spends one forward pass. A model that writes 200 tokens of reasoning first spends 201. So in a decoder, *thinking longer* just means *emitting more tokens before the answer*. Chain-of-thought prompting [12] showed those extra tokens are not wasted: they let the model carry intermediate results forward, and accuracy on multi-step problems jumps. It also showed the gains are uneven. A large meta-analysis later found that chain of thought helps mostly on maths and symbolic problems and barely moves commonsense or knowledge questions [13]. Hold on to that; it is exactly the pattern our model ends up learning by itself.

### 2.2 How RL taught models to always think

Reasoning models are built by reinforcement learning with *verifiable* rewards: sample an answer, check it against the gold value, and push up the probability of whatever led to a correct answer. GRPO [10] does this without a value network. For each prompt it samples a group of $G$ completions, scores them, and uses each completion's reward relative to its group as the advantage:

$$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}, \qquad \nabla J \approx \frac{1}{G}\sum_i A_i \,\nabla \log \pi(o_i \mid q).$$

Here is the first-principles reason this produces ever-longer reasoning. Imagine a question the model can answer three ways: guess immediately (right 35% of the time), reason briefly (80%), or reason at length (90%). Long reasoning has the highest expected reward, so its completions get positive advantage more often, and every token in them, including every "wait, let me check", gets pushed up. Run the update a few hundred times and the probability mass flows toward long reasoning, regardless of whether the question needed it. The figure below runs exactly this update on a three-way policy.

![**Figure 2. RL with a verifiable reward, from first principles.** A softmax policy over three ways of answering one kind](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-rlvr.png)
***Figure 2. RL with a verifiable reward, from first principles.** A softmax policy over three ways of answering one kind of question, updated with the exact expected policy gradient. Each logit moves by $\pi_a (R_a - \mathbb{E}[R])$. With no token price, mass flows to the most accurate option, long reasoning, even though short reasoning is nearly as good. Raise the token price (charged only on correct answers, as in our real reward) and press the button again to see the optimum move to the cheaper option.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-rlvr)

Two caveats from the literature shape everything later. First, RL mostly *re-weights* behaviour the base model can already produce, rather than inventing new abilities [14]: if a mode is never sampled, RL cannot discover it. Second, plain GRPO has quirks when completions differ wildly in length. Groups where every sample gets the same reward carry no gradient [15], and length normalisation biases the update [16]. Both matter when one of your modes is one token long and another is two hundred.

### 2.3 Four ways people have tried to get the fast path back

**A switch the user flips.** Qwen3 merged a thinking and a non-thinking mode into one model: you pass `enable_thinking` or write `/think` and `/no_think` [6]. Our base model has both modes built in. But the *user* decides, not the model.

**A switch the model learns.** A fast-growing line of work trains the model to emit a control token first: Thinkless picks `<short>` or `<think>` [17]. AdaptThink [18], AutoThink [19], hybrid reasoning models [20], AdaCoT [21] and others [22] learn when to skip the think block. ARM chooses among four formats, including a direct answer [23]. The closest prior work to mine, which I found while writing this post, picks one of three modes (no-think, short, long) as the first token inside GRPO, with no external router [24]. Two lessons recur across these papers. Vanilla GRPO tends to *collapse* onto one mode, because one control token is easily outvoted by hundreds of response tokens. And in every one of them, the fast path still *generates* an answer.

**Thinking without words.** Latent reasoning moves the chain of thought into hidden states, feeding the last hidden vector back in as the next input instead of a token [25, 26, 27], or adding silent "pause" tokens [28]. That saves text, not forward passes: the model still iterates.

**Two models and a router.** Cascades and routers send each query to a cheap or an expensive model [29, 30, 31]. GPT-5 ships this way, a fast model and a thinking model behind a real-time router, and its system card says the plan is to merge them into one model [7].

![**Figure 3. Where the fast/slow decision happens.** Every approach above keeps its fast path generating text, splits the](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-landscape.png)
***Figure 3. Where the fast/slow decision happens.** Every approach above keeps its fast path generating text, splits the work across separate models, or never decides at all. Decision heads on a pretrained model (bottom-but-one) do the opposite: they only ever decide in one step. This work puts that one-step decision and the reasoning in the same network, and lets the network choose between them.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-landscape)

### 2.4 The overlooked ingredient: one-step decisions on a language model

There is an older and almost boring idea that turns out to be the key. A pretrained language model is already a very good one-step classifier. GPT-1 and BERT showed that a tiny head on top of pretrained representations beats task-specific models [32, 33]. The standard MMLU evaluation scores a model by reading the probability it assigns to the letters A–D after the question, with no generation at all [34]. Showing all options and scoring one answer symbol works well once a model is good at binding options to letters [35], and it sidesteps the problem of scoring long answer strings [36].

The idea for the fast mode came from `bev-train` [37], which trains "JEV" decision networks on a Qwen3 backbone: typed choices (pick an option, yes/no, or a score on a scale) answered in one pass, with no reasoning. Its training set, the BEV decision mix [38], is a large collection of exactly this kind of decision.

> **Borrowed.** JEV-style decision models are trained on top of an already-trained language model. They reuse the representations the model built for language and read a decision off them in one step.
>
> **The insight.** If the decision lives in the same representation space the model reasons in, the model does not need a second network to decide. It needs to learn *when* to stop and decide. Fast and slow become two ways of using one set of weights, not two systems.
>
> **Still open.** Whether this holds for decisions that are not multiple choice, such as free-form actions or tool calls, is not tested here.

## § 3 The idea: one network, native switching

Put concretely, every answer starts with a short tag. The model sees a system prompt that names the three modes and a control line, `Mode: auto`, or a forced mode such as `Mode: cot`. It writes `<genre:arith><mode:cot>`, or another genre and mode, and the runtime then does one of three things:

| tag | what the runtime does | typical cost |
|---|---|---|
| `<mode:jev>` | No generation. One forward pass; take the logits of the shown option letters only; softmax; argmax. | 1 token |
| `<mode:direct>` | Generate a short `<answer>…</answer>` (capped at 48 tokens). | ~10–20 tokens |
| `<mode:cot>` | Generate reasoning (capped at 256 tokens). If it writes `<mode:jev>` partway, stop generating and finish with a one-step decision. | ~50–150 tokens |

The trick that makes this trainable is that **the one-step decision is a single next-token step in the model's own vocabulary**. Each option letter A–F is a single token. Choosing an option is the same operation as predicting one token with the rest of the vocabulary masked out. So the whole policy — which mode, what reasoning, which letter — is just a distribution over token sequences, and stock token-level GRPO can train all of it at once. Only the *runtime* treats the modes differently.

> **Borrowed.** My first version did what `bev-train` does: a separate order-invariant scoring head on top of Qwen, with its own decision tokens.
>
> **Broke.** Supervised training worked. Reinforcement learning never did. Mixing token log-probabilities with a categorical head's actions needed a custom RL learner, and on TPU it hit NaNs, compiler stalls and memory failures. Most of the budget went into making that learner stable.
>
> **Reframed.** Drop the head. Make the decision an option-letter token in the language model's own output layer. The decision is then just another token, and an off-the-shelf GRPO implementation trains it with no custom code.
>
> **Still open.** A dedicated head is order-invariant by construction; letter logits are not, and models are known to have option-position biases. I did not measure position bias here.

One more piece of design matters later. The model can be *told* which mode to use. Half of the supervised training data has a forced `Mode: X` line, so the tags become real switches: the same weights can be run as "always decide", "always answer briefly" or "always reason". That gives every experiment its own controls. When I compare learned routing with "always reason", it is the same model, same weights, only the switch differs.

## § 4 A failure worth learning from

The first full run of this idea used Qwen3-1.7B, and the results were sobering. Starting RL straight from the base model collapsed: the base model almost never wrote a valid mode tag, the cheapest valid format was the only thing that ever scored, and GRPO pushed the whole policy onto it. Accuracy on the sealed test fell from 9.8% to 6.8%, a textbook case of RL only being able to sharpen what is already there [14].

A short supervised warm-start fixed the collapse and lifted learned routing to 50.5%. But the same model, simply *forced* to reason every time, scored 60.8%. RL at a higher learning rate made the model cheaper (23.6 tokens instead of 29.5) but no better (50.2%). Learned routing was *worse* than not routing at all.

The reason was in the data, not the algorithm. Each training question carried a label saying which mode it "needed", and those labels were assigned when the dataset was generated. The same family of arithmetic problem appeared labelled "fast", "direct", "reason" and "reason then decide". A question like *split 10,014 and 92 items equally* was labelled "direct", but a 1.7B model cannot do that without working it out: on questions labelled direct, it was right only 24% of the time. The model was being taught to route on wording, and RL had nothing clean to correct it with.

> The lesson that shaped everything in v2: **which mode a question needs is not a property of the question. It is a property of the question *and the model***, and it has to be measured.

## § 5 Measuring what the model can do

For v2 I built a pool of 13,076 single-turn questions across 6 kinds of question, chosen so that the right mode should differ *within* each kind:

- **decisions** from the BEV decision mix [38]: sentiment, support routing, code, spatial puzzles, rated qualities;
- **knowledge** multiple choice from ARC, OpenBookQA and CommonsenseQA [39, 40, 41];
- **word problems** from GSM8K and AQuA [42, 43];
- generated **comparisons**, **arithmetic** and **scoring rules**, each with a difficulty knob (digits, steps, nesting depth).

Then, instead of deciding which mode each question needs, I asked the model. Base Qwen3-4B answered every question in every mode it could use: four samples each for "direct" and "reason", and an exact probability for the one-step decision. For each question $x$ and mode $m$ this gives an accuracy $p_m(x)$ and a mean cost $t_m(x)$ in tokens. The label is the mode with the best *utility*:

$$U_m(x) = p_m(x) - \lambda\, t_m(x), \qquad \text{suggested}(x) = \arg\max_m U_m(x).$$

At $\lambda = 0.0005$, a hundred tokens of reasoning are worth five points of accuracy. That price is a value judgement, not a fact, so the labels store the full $p$ and $t$ vectors and can be re-priced for free. The slider below re-prices them live.

![**Figure 4. The measured best mode, by kind of question.** Each bar is up to 500 questions, with the base model's measur](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-labels.png)
***Figure 4. The measured best mode, by kind of question.** Each bar is up to 500 questions, with the base model's measured per-mode accuracy and token cost. Drag the token price: at zero, reasoning wins almost everywhere it helps at all; as tokens get expensive, the one-step decision takes over. Comparisons and knowledge questions are one-step territory at any sensible price, which is what "To CoT or not to CoT" predicts [13]. Arithmetic, word problems and scoring rules genuinely split.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-labels)

The labels came out *capability-relative* in a way no hand-labelling would have produced. Number comparisons, which the 1.7B model needed to reason about, are a one-step decision for 4B on 94% of questions at every digit count. Scoring rules flip from "just pick" at depth one to "reason" at depth two and three. Word problems need reasoning more often as they get longer.

### 5.1 Is there anything to learn? The winner's curse

Before training anything, I wanted to know whether per-question routing could beat the obvious simple rules. The naive check is misleading. Picking the best of four noisy estimates for each question and then scoring that pick on the *same* samples overstates what any router can achieve, the same winner's curse that inflates any "best of N" estimate. The honest version is cross-fitted: choose the mode on two of the four samples and score it on the other two.

![**Figure 5. How much a perfect router could gain.** Utility on all 12,998 questions. "Best mode per question type" is th](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-headroom.png)
***Figure 5. How much a perfect router could gain.** Utility on all 12,998 questions. "Best mode per question type" is the strongest rule that ignores the question itself: always use the mode with the best average utility for that kind of question. A perfect per-question router (cross-fitted, so not flattered by noise) gains 0.064 utility over it. That is real headroom, but not a huge prize.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-headroom)

I set three checks the labels had to pass before I would train on them. **Stability:** labels computed from two halves of the samples agree on 89% of the questions where there is a clear winner. **Headroom:** above 0.05 utility, as just shown. **Within-type variation:** the label should not be predictable from the kind of question alone. That third check *failed* for comparisons, knowledge and decisions, where one-step decisions win on 83%–94% of questions. I kept them anyway, because they are the questions where the right answer is "don't think", but it means much of the routing problem is easy and the interesting part is concentrated in maths and rules.

> **What this doesn't show.** The labels describe the *base* model. After supervised training the model reasons noticeably better, so by the time RL runs, the labels are partly out of date. Re-measuring them on the trained model is the obvious next iteration.

## § 6 A small data-science detour: checking the answer key

Routing labels built on wrong gold answers would teach the model to avoid modes that were actually right. So before profiling, I checked the answer key. Schema checks caught the easy problems: a few percent of AQuA questions had duplicate options or the right answer listed twice. Removing near-duplicates *before* splitting the data guaranteed no test question leaked into training. My own generators had their own bugs: "yes" was always option A, and one scoring rule put most gold answers at its clipped extremes.

For the subtle errors I used the model as a reviewer. Qwen3-4B with its full thinking mode re-answered every validation and test question plus a sample of training questions, and I hand-reviewed every case where it confidently disagreed with the answer key: 276 flags. Most were the model being wrong. Some were not. One GSM8K problem has an official answer of 115 where the arithmetic gives 95; an AQuA problem has no correct option at all. The estimated error rates are small, 0.4% to 1.7% in the maths and science sets. But about one CommonsenseQA question in ten (9.6% of those audited) has two defensible answers, which is worth knowing before reading anything into accuracy on that dataset. I fixed 2 test answers and dropped 4 questions, leaving 12,998.

> **What this doesn't show.** "Confidently disagrees" only finds errors the model can spot. Questions where the model and the key are wrong together are invisible to this audit, so these error rates are lower bounds.

## § 7 Does the model know when it needs to think?

If the right mode is a property of the model, perhaps the model already represents it. Language models are known to be reasonably calibrated about what they know [44], and their hidden states carry signals about whether an answer will be true [45, 46, 47], even mid-reasoning [48]. So I took the base model's final hidden state at the very end of the question, *before it generates anything*, and trained one linear probe per mode to predict that mode's measured accuracy.

![**Figure 6. A probe on the hidden state predicts each mode's success before any token is generated.** Calibration on the](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-probe.png)
***Figure 6. A probe on the hidden state predicts each mode's success before any token is generated.** Calibration on the sealed test questions: predicted probability of being correct (x) against measured accuracy (y), with dot size showing the number of questions. Points near the diagonal mean the probe's confidence can be taken at face value. The probe never saw these questions.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-probe)

It works surprisingly well: AUROC 0.89 for the one-step decision, 0.90 for direct answers and 0.87 for reasoning, with expected calibration errors of a few points. In a meaningful sense, the model largely knows in advance whether it can answer without thinking.

> **Borrowed.** The natural next step is to use the probe as the router: predict each mode's utility and pick the best.
>
> **Broke.** As a router it only tied the simple per-question-type rule on validation (0.818 against 0.819, with a perfect router at 0.870). The probe is good at ranking questions, but routing is decided by *small differences* between modes. When reasoning and a one-step pick are both around 90%, a few points of error flip the choice.
>
> **Reframed.** Do not let the probe decide. Use it to *shape* training, as a soft signal of how much worse one mode is than another, which RL can then override with real outcomes (§9).

## § 8 Teaching the tags as switches

RL can only reinforce behaviour it samples (§4), so the model first has to be able to produce every mode on command. The supervised mix has two halves:

- **Obedience.** The prompt says `Mode: cot`, `jev` or `direct`, and the target is a correct answer in that mode. This is what makes the tags real switches.
- **Routing.** The prompt says `Mode: auto`, and the target uses the mode measured best for that question.

![**Figure 7. The supervised training mix** (14,168 examples). The routing half reflects the measured labels, mostly one-s](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-sft.png)
***Figure 7. The supervised training mix** (14,168 examples). The routing half reflects the measured labels, mostly one-step decisions with reasoning where it pays. The obedience half covers every mode on every kind of question.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-sft)

The reasoning examples are the model's own. For each question I sampled four short chains of thought and kept the shortest one that was correct. Correctness turned out not to be enough. Of 28,604 samples, 8.2% of those containing checkable arithmetic contained a *false* equation, and 78% of those still reached the right answer. The written reasoning is not always the computation that produced the answer, a known property of chain of thought [49, 50]. I filtered those traces out, so the model never imitates a wrong step that happened to land on the right answer.

Supervised training worked as a switch: forced-mode obedience is 100% in every one of the 66 forced-mode evaluations I ran, before and after RL. As a router it under-thinks. Left to choose, the model takes the one-step decision on 71% of test questions and scores 78.1%, 9.6 points below the same weights forced to reason (87.7%). Imitating labels from the base model taught it the base model's caution about thinking, while the fine-tuned model's reasoning had become noticeably better. Closing that gap is the job of RL.

## § 9 Rewarding the right decision

The obvious RL reward is the one every reasoning model uses: 1 for a correct answer, minus a little for every token. Here is why that is not enough for routing, from first principles.

When a model meets a question, the context *shapes* its distribution over modes before it writes anything. Outcome-only RL nudges that distribution using one signal: whether the mode it happened to sample this time was right. For a one-step decision on an easy question, that is usually yes. So early in training, cheap decisions collect a lot of reward, and the policy drifts toward deciding without thinking — *including on the questions where thinking was the better bet*, because a single lucky correct guess looks exactly like a good decision. How far a run drifts depends on the luck of its samples. Nothing in the reward says that, for *this* question, a different mode is measurably better.

So I added the missing signal. The reward has the usual verifiable terms, plus one more:

$$r_{\text{route}} = -\beta_t \,\bigl(\max_{m} U_m(x) - U_{\hat m}(x)\bigr),$$

where $\hat m$ is the mode the model chose and $U_m(x)$ are the per-mode utilities: predicted by the probe from §7, or measured directly. This is the *regret* of the decision, how much utility the chosen mode leaves on the table compared with the best mode for this question. Three properties make it the right shape:

- **It is zero when the choice is as good as the best,** including near-ties. Unlike a reward for imitating the suggested label, it never punishes a different but equally good mode, so RL is free to find cheaper routes than the labels suggest.
- **It is graded.** Picking a one-step decision on a word problem where it is right 25% of the time costs far more than on a comparison where it is right 99% of the time.
- **It fades out.** $\beta_t$ starts at 0.5 and decays to zero over the first 60% of training. The regret term steers early exploration; the final policy is decided by real outcomes.

The other terms carry over from standard practice, with two changes that came from v1's failures. The token cost is charged only on *correct* answers, so a short wrong answer never beats a long right one. Disobeying a forced mode is the worst possible outcome, so obedience cannot decay during RL; 30% of RL prompts force a mode. I also tried two extra terms: a penalty for false equations in the reasoning, and a bonus for naming the right kind of question.

![**Figure 8. The reward, term by term.** This reproduces the training reward exactly (`kwt/rewards_v2.py`). Pick a kind o](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-reward.png)
***Figure 8. The reward, term by term.** This reproduces the training reward exactly (`kwt/rewards_v2.py`). Pick a kind of question and the mode the model chose; flip whether the answer was right; drag training progress to watch the regret term fade out. On a word problem, a correct one-step decision still loses most of its advantage early in training, because reasoning was the measurably better bet.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-reward)

> **What this doesn't show.** The utilities behind the regret come from the base model (or a probe trained on it), so the regret term inherits the staleness noted in §5. Fading it out limits the damage, but a regret target that is updated as the policy improves would be better.

## § 10 What RL learned

Every RL run starts from the same supervised checkpoint and trains for 150 GRPO steps: four questions per step, eight samples each, on a four-chip TPU v5e slice. I trained the two main configurations five times each with different sampling seeds, plus single runs of the ablations. Everything below is on the sealed test split, opened once, and on an out-of-distribution split: a decision domain never seen in training, plus 6-digit arithmetic where training only had up to 5 digits.

### 10.1 Training dynamics

![**Figure 9. How the policy moves during RL.** Each line is one training run; colour is the reward. Both rewards start fr](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-dynamics.png)
***Figure 9. How the policy moves during RL.** Each line is one training run; colour is the reward. Both rewards start from the same under-thinking checkpoint and both learn to reason more. The outcome-only runs wander: each settles at a different level of reasoning. Logs for three of the ten seeds were overwritten by an accidental relaunch after training finished and are not shown; their evaluation results are unaffected.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-dynamics)

### 10.2 Accuracy against tokens

![**Figure 10. Every model on the accuracy–token plane.** Circles are the regret reward, squares outcome-only, one point p](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-frontier.png)
***Figure 10. Every model on the accuracy–token plane.** Circles are the regret reward, squares outcome-only, one point per seed; diamonds are the same weights forced to always reason. Hover for details. With the regret reward, all five seeds sit together at the top, about a third cheaper than always reasoning. Outcome-only seeds lie along a line: the fewer tokens a run learned to spend, the lower its accuracy.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-frontier)

With the regret reward, learned routing reaches 88.6% on the test set at 54.4 tokens per question. The same weights forced to reason every time reach 88.5% at 78.5 tokens. That is the same accuracy for 31% fewer tokens, and +10.5 points over the supervised router it started from. Out of distribution the pattern repeats: 78.3% at 45.5 tokens, against 78.6% at 80.8 for always reasoning.

### 10.3 The regret reward buys stability

![**Figure 11. Five seeds per reward.** Each dot is one complete RL run evaluated on the full split; bars are means.](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-seeds.png)
***Figure 11. Five seeds per reward.** Each dot is one complete RL run evaluated on the full split; bars are means.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-seeds)

This is the result I did not expect going in. With the regret reward, five seeds score 88.6% ± 0.3 points (range 88.2%–89.0%). With an outcome-only reward, they score 85.7% ± 3.2 points (range 80.3%–88.4%). The regret reward is +2.9 points better on average with 12× less spread between runs. With five runs on each side, a one-sided exact permutation test gives p = 0.008. Out of distribution the direction is the same but the gap is small (+1.0 points, p = 0.071), so I would not claim it there.

The ablations tell a consistent story. Using the *measured* utilities for the regret instead of the probe's predictions gives the same result (88.2% against 88.6%). The cheap probe is a valid stand-in for profiling every training question. The two extra rewards did not help. The false-equation penalty left the rate of reasoning traces with only true equations essentially unchanged (96.7% with it, 96.5% without), and adding it with the genre bonus *cost* 0.8 points (87.9% against 88.6%). On the same seed, the regret term alone beats outcome-only by +1.6 points [+0.6, +2.8].

> **What this doesn't show.** Five seeds is enough to see that outcome-only runs vary a lot, not enough to say how often they fail badly. The stability claim is about this model, this data and 150 steps. Longer training might let outcome-only runs converge on their own.

### 10.4 What it learned to do

![**Figure 12. When the model chooses to reason, by kind of question** (test, mean over five seeds). RL teaches the model ](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-genres.png)
***Figure 12. When the model chooses to reason, by kind of question** (test, mean over five seeds). RL teaches the model to reason on multi-step scoring rules and arithmetic, and to keep making one-step decisions on comparisons, knowledge questions and most decisions. Switch to "accuracy" to see where it pays.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-genres)

The behaviour RL found matches the meta-analysis from §2 [13], learned from scratch by a 4B model with no hint about which questions are "mathy". Scoring rules go from 55% to 100% accuracy as the model learns to reason through rules of depth two and three. Arithmetic goes from 83% to 94%. Comparisons stay at 100% with almost no reasoning at all.

![**Figure 13. Real outputs from the trained model** (best seed, sealed test). Filter by kind of question, chosen mode and](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-traces.png)
***Figure 13. Real outputs from the trained model** (best seed, sealed test). Filter by kind of question, chosen mode and outcome; each card also shows the same question answered with reasoning forced on. Look for the hand-offs: a line or two of arithmetic, then `<mode:jev>` and a one-step pick. Questions from the BEV decision mix, OpenBookQA and ARC are left out here because their licences do not clearly allow republishing the text.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-traces)

## § 11 Is it worth it?

"Same accuracy, fewer tokens" only matters if tokens cost something, and how much they cost is a choice. The same saved outputs can be re-priced at any token price $\lambda$ without retraining:

![**Figure 14. Learned routing against always reasoning, as tokens get more expensive.** Utility difference between the mo](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-sweep.png)
***Figure 14. Learned routing against always reasoning, as tokens get more expensive.** Utility difference between the model choosing its mode and the same weights always reasoning, averaged over the five regret-reward seeds, with a 95% bootstrap interval over questions. Above zero means routing wins.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-sweep)

At a token price of zero, where only accuracy counts, the two tie (0.001). At the price used in training, routing is ahead by 0.013 utility (95% interval 0.001 to 0.025). At $\lambda = 0.001$, where a hundred tokens are worth ten accuracy points, the gap is 0.025. Learned routing never loses.

Tokens are not time, so I also timed 160 test questions one at a time on a single v5e chip, after a warm-up pass so that compilation does not count.

![**Figure 15. Wall-clock time per question at batch size 1.** The same trained weights on 160 test questions, once choosi](https://tinevimbomusingadi.github.io/kwtfs-blog/img/fig-latency.png)
***Figure 15. Wall-clock time per question at batch size 1.** The same trained weights on 160 test questions, once choosing their own mode and once forced to reason, on one v5e chip after a warm-up pass.* [Interactive version](https://tinevimbomusingadi.github.io/kwtfs-blog/#fig-latency)

Learned routing answers in 1.74 seconds on average (1.59 median), against 2.19 (2.28 median) for always reasoning: 20% faster on average and 30% at the median. That is less than the token saving, for a reason worth spelling out. A one-step decision should cost one forward pass, but in my runtime it took 1.67 seconds. The runtime decodes a short header before switching, re-reads the prompt at every switch instead of reusing the cache, and pays a fixed setup cost per generation call. A serving stack that stops at the mode tag and keeps the KV cache would turn most of the token saving into time.

The whole project cost roughly forty dollars of spot TPU time. Data work, supervised training and evaluation ran on single v5e chips at about fifty cents an hour; RL needed a four-chip slice. The training step in Tunix returns a full second copy of the frozen 4B weights, which does not fit next to the first copy on a 16 GB chip.

## § 12 Discussion and open questions

The thread through this project is that knowing when to think is not one decision but a chain of them. You have to measure what the model can do, teach it that it *can* switch, and reward it for switching well, not just for being right. The specific contributions, stated carefully against the work in §2:

- **A fast path that generates nothing.** Learned think switches exist [17, 23, 24], but their fast paths still write an answer. Here the fastest mode is a single forward pass reading the option letters, in the same network as the reasoning.
- **Labels measured on the model.** The mode a question needs depends on the model; measuring it per question, with cross-fitting and stability checks, is what made learned routing match always-reasoning instead of trailing it.
- **A regret reward against measured utilities**, which in these experiments turned an unstable RL procedure into a reproducible one.

Questions I would most like to see answered, by me or anyone:

- **Re-measure as you train.** Labels and regret come from the base model. Measure, train, re-measure, retrain: does the router keep improving as the model's own abilities change?
- **Scale.** Does the stability benefit hold for 8B and larger models, and for longer RL runs?
- **Beyond multiple choice.** The one-step mode needs a set of options. Typed decisions, such as a score on a scale, a tool call or an action, are the natural extension.
- **Other modalities.** In an omni model, the modes become "answer from the image directly", "describe it, then reason" and "pick an option". The same recipe should apply: measure each mode, teach the switches, reward the regret. That is the next project.

## References

1. Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux. [https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
2. OpenAI (2024). Learning to reason with LLMs. [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/)
3. DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
4. Snell, C. et al. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv:2408.03314. [https://arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)
5. Chen, X. et al. (2024). Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. arXiv:2412.21187. [https://arxiv.org/abs/2412.21187](https://arxiv.org/abs/2412.21187)
6. Yang, A. et al. (2025). Qwen3 Technical Report. arXiv:2505.09388. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
7. OpenAI (2025). OpenAI GPT-5 System Card. arXiv:2601.03267. [https://arxiv.org/abs/2601.03267](https://arxiv.org/abs/2601.03267)
8. Qwen Team (2025). Qwen3-4B model card. Hugging Face. [https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
9. Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
10. Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (introduces GRPO). arXiv:2402.03300. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
11. Google (2025). Tunix: a JAX-based LLM post-training library. GitHub. [https://github.com/google/tunix](https://github.com/google/tunix)
12. Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
13. Sprague, Z. et al. (2024). To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning. ICLR 2025. arXiv:2409.12183. [https://arxiv.org/abs/2409.12183](https://arxiv.org/abs/2409.12183)
14. Yue, Y. et al. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? NeurIPS 2025. arXiv:2504.13837. [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837)
15. Yu, Q. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476. [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
16. Liu, Z. et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783. [https://arxiv.org/abs/2503.20783](https://arxiv.org/abs/2503.20783)
17. Fang, G., Ma, X., Wang, X. (2025). Thinkless: LLM Learns When to Think. arXiv:2505.13379. [https://arxiv.org/abs/2505.13379](https://arxiv.org/abs/2505.13379)
18. Zhang, J. et al. (2025). AdaptThink: Reasoning Models Can Learn When to Think. arXiv:2505.13417. [https://arxiv.org/abs/2505.13417](https://arxiv.org/abs/2505.13417)
19. Tu, S. et al. (2025). Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL. NeurIPS 2025. arXiv:2505.10832. [https://arxiv.org/abs/2505.10832](https://arxiv.org/abs/2505.10832)
20. Jiang, L. et al. (2025). Think Only When You Need with Large Hybrid-Reasoning Models. arXiv:2505.14631. [https://arxiv.org/abs/2505.14631](https://arxiv.org/abs/2505.14631)
21. Lou, C. et al. (2025). AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning. arXiv:2505.11896. [https://arxiv.org/abs/2505.11896](https://arxiv.org/abs/2505.11896)
22. Xu, Z. et al. (2026). Stable Adaptive Thinking via Advantage Shaping and Length-Aware Gradient Regulation. arXiv:2602.22556. [https://arxiv.org/abs/2602.22556](https://arxiv.org/abs/2602.22556)
23. Wu, S. et al. (2025). ARM: Adaptive Reasoning Model. NeurIPS 2025. arXiv:2505.20258. [https://arxiv.org/abs/2505.20258](https://arxiv.org/abs/2505.20258)
24. Kassenaar, G., Yang, Z., Francois-Lavet, V. (2026). Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation. arXiv:2608.20256. [https://arxiv.org/abs/2608.20256](https://arxiv.org/abs/2608.20256)
25. Hao, S. et al. (2024). Training Large Language Models to Reason in a Continuous Latent Space. COLM 2025. arXiv:2412.06769. [https://arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769)
26. Zhang, Z. et al. (2025). Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space. arXiv:2505.15778. [https://arxiv.org/abs/2505.15778](https://arxiv.org/abs/2505.15778)
27. Zhu, R.-J. et al. (2025). A Survey on Latent Reasoning. arXiv:2507.06203. [https://arxiv.org/abs/2507.06203](https://arxiv.org/abs/2507.06203)
28. Goyal, S. et al. (2023). Think before you speak: Training Language Models With Pause Tokens. arXiv:2310.02226. [https://arxiv.org/abs/2310.02226](https://arxiv.org/abs/2310.02226)
29. Chen, L., Zaharia, M., Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176. [https://arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)
30. Ding, D. et al. (2024). Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. ICLR 2024. arXiv:2404.14618. [https://arxiv.org/abs/2404.14618](https://arxiv.org/abs/2404.14618)
31. Ong, I. et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665. [https://arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665)
32. Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI. [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
33. Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. arXiv:1810.04805. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
34. Hendrycks, D. et al. (2020). Measuring Massive Multitask Language Understanding. ICLR 2021. arXiv:2009.03300. [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)
35. Robinson, J., Rytting, C. M., Wingate, D. (2022). Leveraging Large Language Models for Multiple Choice Question Answering. ICLR 2023. arXiv:2210.12353. [https://arxiv.org/abs/2210.12353](https://arxiv.org/abs/2210.12353)
36. Holtzman, A. et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. arXiv:2104.08315. [https://arxiv.org/abs/2104.08315](https://arxiv.org/abs/2104.08315)
37. avbiswas. bev-train: Training JEV-like decision models using Qwen3 with choice-order invariance. GitHub repository. [https://github.com/avbiswas/bev-train](https://github.com/avbiswas/bev-train)
38. avbiswas. BEV Decision Mix (Hugging Face dataset avbiswas/bev-decision; licence mixed/unknown). [https://huggingface.co/datasets/avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
39. Clark, P. et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457. [https://arxiv.org/abs/1803.05457](https://arxiv.org/abs/1803.05457)
40. Mihaylov, T. et al. (2018). Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. EMNLP 2018. [https://arxiv.org/abs/1809.02789](https://arxiv.org/abs/1809.02789)
41. Talmor, A. et al. (2019). CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. NAACL 2019. [https://arxiv.org/abs/1811.00937](https://arxiv.org/abs/1811.00937)
42. Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K). arXiv:2110.14168. [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
43. Ling, W. et al. (2017). Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems (AQuA-RAT). ACL 2017. [https://arxiv.org/abs/1705.04146](https://arxiv.org/abs/1705.04146)
44. Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221. [https://arxiv.org/abs/2207.05221](https://arxiv.org/abs/2207.05221)
45. Azaria, A., Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. arXiv:2304.13734. [https://arxiv.org/abs/2304.13734](https://arxiv.org/abs/2304.13734)
46. Kossen, J. et al. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927. [https://arxiv.org/abs/2406.15927](https://arxiv.org/abs/2406.15927)
47. Orgad, H. et al. (2024). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707. [https://arxiv.org/abs/2410.02707](https://arxiv.org/abs/2410.02707)
48. Zhang, A. et al. (2025). Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification. arXiv:2504.05419. [https://arxiv.org/abs/2504.05419](https://arxiv.org/abs/2504.05419)
49. Turpin, M. et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023. arXiv:2305.04388. [https://arxiv.org/abs/2305.04388](https://arxiv.org/abs/2305.04388)
50. Lanham, T. et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702. [https://arxiv.org/abs/2307.13702](https://arxiv.org/abs/2307.13702)

## § A Appendix: engineering notes

**Reproducibility.** The code, configs and a Tunix notebook are in [kwtfs-public](https://github.com/TinevimboMusingadi/kwtfs-public). Data and labels are on Hugging Face, with question text omitted for sources whose licence does not allow redistribution. Every number on this page is regenerated from saved model outputs by the scripts in this site's repository.

**Training details.** Base model: Qwen3-4B in bfloat16, LoRA rank 16 on all attention and MLP projections. SFT: one epoch over 14,168 examples, batch 2, learning rate 1e-4, with the mode-tag tokens weighted 5× in the loss. GRPO: 150 steps, 4 prompts × 8 samples, learning rate 3e-5, KL weight 0.02, clip 0.2, sampling temperature 1.0, generation capped at 256 tokens, on a TPU v5litepod-4 slice.

**Small TPUs teach you things.** A dozen bugs were caught by tiny smoke runs before they could waste a full one. Tunix computes in float32 unless told otherwise, so the KV cache alone overflowed a 16 GB chip. The one-step decision built logits over the entire 152k vocabulary at every position. Compiled programs piled up in memory until the decoder could not load. My favourite: the reasoning prompt said "end with `<answer>FINAL</answer>`", and the model dutifully wrote the word FINAL. Accuracy for that mode was zero until I gave it a real example.

**Citation.**

```
@misc{musingadi2026kwtfs,
  author = {Musingadi, Tinevimbo},
  title  = {Knowing When to Think Fast or Slow at Test Time},
  year   = {2026},
  url    = {https://tinevimbomusingadi.github.io/kwtfs-blog/}
}
```

1. Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux. [https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
2. OpenAI (2024). Learning to reason with LLMs. [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/)
3. DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
4. Snell, C. et al. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv:2408.03314. [https://arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)
5. Chen, X. et al. (2024). Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. arXiv:2412.21187. [https://arxiv.org/abs/2412.21187](https://arxiv.org/abs/2412.21187)
6. Yang, A. et al. (2025). Qwen3 Technical Report. arXiv:2505.09388. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
7. OpenAI (2025). OpenAI GPT-5 System Card. arXiv:2601.03267. [https://arxiv.org/abs/2601.03267](https://arxiv.org/abs/2601.03267)
8. Qwen Team (2025). Qwen3-4B model card. Hugging Face. [https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
9. Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
10. Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (introduces GRPO). arXiv:2402.03300. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
11. Google (2025). Tunix: a JAX-based LLM post-training library. GitHub. [https://github.com/google/tunix](https://github.com/google/tunix)
12. Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
13. Sprague, Z. et al. (2024). To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning. ICLR 2025. arXiv:2409.12183. [https://arxiv.org/abs/2409.12183](https://arxiv.org/abs/2409.12183)
14. Yue, Y. et al. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? NeurIPS 2025. arXiv:2504.13837. [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837)
15. Yu, Q. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476. [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
16. Liu, Z. et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783. [https://arxiv.org/abs/2503.20783](https://arxiv.org/abs/2503.20783)
17. Fang, G., Ma, X., Wang, X. (2025). Thinkless: LLM Learns When to Think. arXiv:2505.13379. [https://arxiv.org/abs/2505.13379](https://arxiv.org/abs/2505.13379)
18. Zhang, J. et al. (2025). AdaptThink: Reasoning Models Can Learn When to Think. arXiv:2505.13417. [https://arxiv.org/abs/2505.13417](https://arxiv.org/abs/2505.13417)
19. Tu, S. et al. (2025). Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL. NeurIPS 2025. arXiv:2505.10832. [https://arxiv.org/abs/2505.10832](https://arxiv.org/abs/2505.10832)
20. Jiang, L. et al. (2025). Think Only When You Need with Large Hybrid-Reasoning Models. arXiv:2505.14631. [https://arxiv.org/abs/2505.14631](https://arxiv.org/abs/2505.14631)
21. Lou, C. et al. (2025). AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning. arXiv:2505.11896. [https://arxiv.org/abs/2505.11896](https://arxiv.org/abs/2505.11896)
22. Xu, Z. et al. (2026). Stable Adaptive Thinking via Advantage Shaping and Length-Aware Gradient Regulation. arXiv:2602.22556. [https://arxiv.org/abs/2602.22556](https://arxiv.org/abs/2602.22556)
23. Wu, S. et al. (2025). ARM: Adaptive Reasoning Model. NeurIPS 2025. arXiv:2505.20258. [https://arxiv.org/abs/2505.20258](https://arxiv.org/abs/2505.20258)
24. Kassenaar, G., Yang, Z., Francois-Lavet, V. (2026). Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation. arXiv:2608.20256. [https://arxiv.org/abs/2608.20256](https://arxiv.org/abs/2608.20256)
25. Hao, S. et al. (2024). Training Large Language Models to Reason in a Continuous Latent Space. COLM 2025. arXiv:2412.06769. [https://arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769)
26. Zhang, Z. et al. (2025). Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space. arXiv:2505.15778. [https://arxiv.org/abs/2505.15778](https://arxiv.org/abs/2505.15778)
27. Zhu, R.-J. et al. (2025). A Survey on Latent Reasoning. arXiv:2507.06203. [https://arxiv.org/abs/2507.06203](https://arxiv.org/abs/2507.06203)
28. Goyal, S. et al. (2023). Think before you speak: Training Language Models With Pause Tokens. arXiv:2310.02226. [https://arxiv.org/abs/2310.02226](https://arxiv.org/abs/2310.02226)
29. Chen, L., Zaharia, M., Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176. [https://arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)
30. Ding, D. et al. (2024). Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. ICLR 2024. arXiv:2404.14618. [https://arxiv.org/abs/2404.14618](https://arxiv.org/abs/2404.14618)
31. Ong, I. et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665. [https://arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665)
32. Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI. [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
33. Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. arXiv:1810.04805. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
34. Hendrycks, D. et al. (2020). Measuring Massive Multitask Language Understanding. ICLR 2021. arXiv:2009.03300. [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)
35. Robinson, J., Rytting, C. M., Wingate, D. (2022). Leveraging Large Language Models for Multiple Choice Question Answering. ICLR 2023. arXiv:2210.12353. [https://arxiv.org/abs/2210.12353](https://arxiv.org/abs/2210.12353)
36. Holtzman, A. et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. arXiv:2104.08315. [https://arxiv.org/abs/2104.08315](https://arxiv.org/abs/2104.08315)
37. avbiswas. bev-train: Training JEV-like decision models using Qwen3 with choice-order invariance. GitHub repository. [https://github.com/avbiswas/bev-train](https://github.com/avbiswas/bev-train)
38. avbiswas. BEV Decision Mix (Hugging Face dataset avbiswas/bev-decision; licence mixed/unknown). [https://huggingface.co/datasets/avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
39. Clark, P. et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457. [https://arxiv.org/abs/1803.05457](https://arxiv.org/abs/1803.05457)
40. Mihaylov, T. et al. (2018). Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. EMNLP 2018. [https://arxiv.org/abs/1809.02789](https://arxiv.org/abs/1809.02789)
41. Talmor, A. et al. (2019). CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. NAACL 2019. [https://arxiv.org/abs/1811.00937](https://arxiv.org/abs/1811.00937)
42. Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K). arXiv:2110.14168. [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
43. Ling, W. et al. (2017). Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems (AQuA-RAT). ACL 2017. [https://arxiv.org/abs/1705.04146](https://arxiv.org/abs/1705.04146)
44. Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221. [https://arxiv.org/abs/2207.05221](https://arxiv.org/abs/2207.05221)
45. Azaria, A., Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. arXiv:2304.13734. [https://arxiv.org/abs/2304.13734](https://arxiv.org/abs/2304.13734)
46. Kossen, J. et al. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927. [https://arxiv.org/abs/2406.15927](https://arxiv.org/abs/2406.15927)
47. Orgad, H. et al. (2024). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707. [https://arxiv.org/abs/2410.02707](https://arxiv.org/abs/2410.02707)
48. Zhang, A. et al. (2025). Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification. arXiv:2504.05419. [https://arxiv.org/abs/2504.05419](https://arxiv.org/abs/2504.05419)
49. Turpin, M. et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023. arXiv:2305.04388. [https://arxiv.org/abs/2305.04388](https://arxiv.org/abs/2305.04388)
50. Lanham, T. et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702. [https://arxiv.org/abs/2307.13702](https://arxiv.org/abs/2307.13702)

[^1]: The site is built by `tools/build_data.py`, which re-scores every saved model completion with the same code the experiments used. `tools/render.py` then fills each number into the text and refuses to render if any placeholder is missing.
