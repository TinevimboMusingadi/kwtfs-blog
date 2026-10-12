# Annotated bibliography: "Knowing when to think fast or slow at test time"

Verified 2026-10-12. arXiv entries were checked against the arXiv export API (title, authors, first-submission date, journal-ref/comments). Non-arXiv items were checked by fetching the page or by web search. Venues are given only where arXiv metadata or the source page states them; otherwise the entry says "arXiv".

Relation tags: **[borrowed]** we use it directly; **[motivates]** it motivates the problem; **[contrast]** it is a contrasting approach; **[explains]** it explains one of our observations.

---

## 1. Dual-process framing and test-time compute

1. **[kahneman2011]** Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux (ISBN 978-0374275631). https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
   The book describes human judgment as two modes. "System 1" is fast, automatic and associative. "System 2" is slow, effortful and deliberate, and it is engaged selectively because effort is costly. **[motivates]** It gives the title and the framing: the JEV one-step decision plays System 1, CoT plays System 2, and the learned switch is the selective engagement. The analogy is loose, and the blog should not claim more for it.

2. **[snell2024]** Snell, C. et al. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv:2408.03314. https://arxiv.org/abs/2408.03314
   The paper shows that the best way to spend test-time compute (revisions vs. search against a process verifier) depends on prompt difficulty. A "compute-optimal" strategy that allocates per prompt by estimated difficulty is >4x more efficient than best-of-N. On some problems, a smaller model with extra test-time compute matches a 14x larger model. **[motivates]** This is the closest classic statement of "allocate compute per question by difficulty". We take that idea down to a discrete choice between zero-token and CoT modes inside one model.

3. **[openai2024reason]** OpenAI (2024, Sept 12). Learning to reason with LLMs (blog post introducing o1-preview). https://openai.com/index/learning-to-reason-with-llms/
   The post states that large-scale RL teaches the model to use its chain of thought productively. It reports that o1's performance rises smoothly with both train-time RL compute and test-time "thinking" compute, and that o1 was evaluated at its maximal test-time compute setting. **[motivates]** This is the "always think, think longer" paradigm. Our work asks the opposite question: when to spend no thinking at all.

4. **[openai2024o1card]** OpenAI (2024). OpenAI o1 System Card. arXiv:2412.16720. https://arxiv.org/abs/2412.16720
   The system card for o1 and o1-mini, models trained with RL to produce a long chain of thought before answering. It is mainly a safety and evaluation document. **[motivates]** Cite it alongside the blog post as the canonical always-reasoning model.

## 2. RL with verifiable rewards (RLVR), its limits, and overthinking

5. **[wei2022cot]** Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903. https://arxiv.org/abs/2201.11903
   Few-shot exemplars with intermediate reasoning steps improve large models on arithmetic, commonsense and symbolic tasks. A 540B model reached SOTA on GSM8K with eight exemplars. **[motivates]** This is the origin of the "slow" mode we choose between.

6. **[shao2024deepseekmath]** Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. https://arxiv.org/abs/2402.03300
   The paper introduces Group Relative Policy Optimization (GRPO), a PPO variant that drops the value network. It samples a group of responses per prompt and uses group-normalised rewards as the advantage. **[borrowed]** GRPO is our RL algorithm. Our regret reward is fed through its group-relative advantage.

7. **[deepseek2025r1]** DeepSeek-AI (Guo, D. et al.) (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948; Nature 645, 633-638 (2025). https://arxiv.org/abs/2501.12948
   With rule-based (verifiable) accuracy and format rewards, GRPO on a base model (R1-Zero) yields long, reflective CoT without SFT. R1 adds cold-start SFT and multi-stage RL. **[borrowed]** We use the same recipe shape (SFT cold start, then GRPO with a verifiable reward). Our reward targets mode choice, not CoT length or quality.

8. **[yu2025dapo]** Yu, Q. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476. https://arxiv.org/abs/2503.14476
   DAPO modifies GRPO with four changes. Clip-higher uses decoupled clip ranges. Dynamic sampling drops groups whose rewards are all equal and so give zero advantage. Token-level policy-gradient loss replaces the sample-level loss. Overlong reward shaping handles truncated responses. Together these give stable large-scale RLVR. **[explains]** Groups with all-equal rewards carry no gradient under GRPO. This is relevant when our per-question regret is zero for every sampled mode, as on easy questions, and it also bears on mode collapse. (Tunix ships a DAPO implementation.)

9. **[liu2025drgrpo]** Liu, Z. et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783. https://arxiv.org/abs/2503.20783
   The paper shows that GRPO's per-response length normalisation and per-group std normalisation introduce an optimisation bias. The bias artificially grows the length of incorrect responses and weights questions unevenly. Dr. GRPO removes both terms. The paper also finds that base models such as Qwen2.5 already show reasoning ability and "aha" behaviour before RL. **[explains]** The length bias matters when modes differ hugely in length (0 tokens vs. CoT). It also bears on std normalisation of our regret reward. Dr. GRPO is available in Tunix.

10. **[yue2025rlvr]** Yue, Y. et al. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? arXiv:2504.13837. NeurIPS 2025 (oral). https://arxiv.org/abs/2504.13837
    RLVR models beat their base models at pass@1, but base models reach higher pass@k at large k. Coverage and perplexity analyses suggest that RLVR reweights reasoning paths already in the base distribution rather than creating new ones. Distillation, by contrast, does add new patterns. **[explains]** This explains why our RL mostly reshapes which mode is chosen and gains little per-mode accuracy. It also supports framing GRPO here as a selector, not a teacher.

11. **[chen2024overthinking]** Chen, X. et al. (2024). Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. arXiv:2412.21187. https://arxiv.org/abs/2412.21187
    The paper documents "overthinking": o1-like models spend many tokens on trivial problems, and later solution rounds add little accuracy. It proposes outcome and process efficiency metrics, and self-training that trims redundant solutions. **[motivates]** This is direct evidence that always-on CoT wastes compute on easy items.

12. **[sui2025survey]** Sui, Y. et al. (2025). Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models. arXiv:2503.16419. https://arxiv.org/abs/2503.16419
    The survey sorts efficient-reasoning work into three groups: model-based (training concise reasoners), output-based (dynamically cutting steps at inference) and prompt-based (difficulty- or length-aware prompting). **[motivates]** It is a good pointer for readers. Our method falls under model-based, but with a zero-generation fast path that the survey's categories do not really cover.

## 3. Hybrid and adaptive thinking switches (one model chooses whether to think)

13. **[qwen3]** Yang, A. et al. (Qwen Team) (2025). Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388
    Qwen3 merges a thinking mode and a non-thinking mode in one model through a four-stage post-training pipeline (long-CoT cold start, reasoning RL, thinking-mode fusion, general RL). Users switch modes with `enable_thinking` or the `/think` and `/no_think` tags, and a thinking budget is available. **[borrowed]** Our base model has both modes built in. The user or the template chooses the switch, never the model. Our work makes that choice learned and per question, and adds a third, zero-token mode.

14. **[fang2025thinkless]** Fang, G., Ma, X., Wang, X. (2025). Thinkless: LLM Learns When to Think. arXiv:2505.13379. https://arxiv.org/abs/2505.13379
    The model emits a control token (`<short>` or `<think>`) and then the response. Decoupled GRPO (DeGRPO) splits the loss into a control-token term and a response term with separate weights. This prevents the mode collapse that vanilla GRPO shows, because the single control token gets swamped by long responses. Long-chain use drops by 50-90% on Minerva Algebra, MATH-500 and GSM8K. **[contrast]** This is the nearest single-model analogue to our setup. The differences: our fast path generates no tokens at all, our modes include a one-pass letter decision, and our reward is regret against measured per-mode utilities, not plain correctness plus length. Their collapse diagnosis also **[explains]** why mode-choice tokens need special handling in GRPO.

15. **[zhang2025adaptthink]** Zhang, J. et al. (2025). AdaptThink: Reasoning Models Can Learn When to Think. arXiv:2505.13417. https://arxiv.org/abs/2505.13417
    The paper shows that "NoThinking" (prompting an R1-distill model to skip the think block) is better on easy problems. RL then uses a constrained objective that favours NoThinking subject to not losing accuracy, plus importance sampling to keep both modes on-policy from a cold start. Response length falls 53% and accuracy rises 2.4% on R1-Distill-Qwen-1.5B. **[contrast]** It is a binary think/no-think switch whose fast path still generates an answer. Their importance-sampling fix is relevant to exploring rare modes.

16. **[tu2025autothink]** Tu, S. et al. (2025). Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL. arXiv:2505.10832. NeurIPS 2025. https://arxiv.org/abs/2505.10832
    The authors observe that putting an ellipsis ("...") in the prompt makes R1-style models stochastically pick thinking or no-thinking. AutoThink uses multi-stage RL with stage-wise reward shaping to make that choice adaptive: +6.4% relative accuracy and -52% tokens on R1-Distill-Qwen-1.5B. **[contrast]** The mode is chosen implicitly by generation, not by an explicit typed decision. **[explains]** Latent mode controllability already exists in the base policy, in line with "RL reweights what is there".

17. **[wu2025arm]** Wu, S. et al. (2025). ARM: Adaptive Reasoning Model. arXiv:2505.20258. NeurIPS 2025 (spotlight). https://arxiv.org/abs/2505.20258
    ARM chooses among four reasoning formats: Direct Answer, Short CoT, Code and Long CoT. It is trained with Ada-GRPO, which up-weights rare formats to avoid "format collapse". Tokens fall about 30% on average (up to 70%) at Long-CoT-level accuracy. The paper also offers an instruction-guided mode and a consensus-guided mode. **[contrast]** Its multi-format menu is the closest to our three modes. Even ARM's "Direct Answer" still generates text, and its reward is format-frequency-scaled correctness, not regret against per-mode utilities.

18. **[jiang2025lhrm]** Jiang, L. et al. (2025). Think Only When You Need with Large Hybrid-Reasoning Models. arXiv:2505.14631. https://arxiv.org/abs/2505.14631
    LHRMs are trained in two stages: Hybrid Fine-Tuning (HFT) on both thinking and no-thinking data as a cold start, then Hybrid Group Policy Optimization (HGPO) to choose the mode implicitly. The paper also proposes a "Hybrid Accuracy" metric for mode selection. **[contrast]** This is a closely parallel SFT-then-RL pipeline. Their mode-selection metric is a useful comparison point for our regret measure.

19. **[lou2025adacot]** Lou, C. et al. (ByteDance) (2025). AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning. arXiv:2505.11896. https://arxiv.org/abs/2505.11896
    AdaCoT treats CoT triggering as Pareto optimisation of accuracy against CoT cost, trained with PPO and tunable penalty coefficients. Selective Loss Masking prevents the decision boundary from collapsing in multi-stage RL. On production traffic, CoT triggering fell to as low as 3.18% and response tokens fell 69%. **[contrast]** It is a production-scale binary trigger. Its penalty-coefficient sweep is the analogue of our token-price sweep.

20. **[xu2026stable]** Xu, Z. et al. (2026). Stable Adaptive Thinking via Advantage Shaping and Length-Aware Gradient Regulation. arXiv:2602.22556. https://arxiv.org/abs/2602.22556
    The method has two stages. Hybrid Fine-Tuning exposes the model to both thinking and no-thinking behaviour. Adaptive RL then adds Correctness-Preserving Advantage Shaping (CPAS), so that correct long reasoning is not suppressed, and Length-Aware Gradient Regulation (LAGR), to stabilise training under very different response lengths. On Qwen2.5-1.5B/7B, accuracy rises up to +3.7 points with about 40% fewer tokens. **[explains]** It names the same instabilities we face (length heterogeneity across modes, punishing correct slow answers). Their fixes are alternatives to our regret reward.

21. **[kassenaar2026when]** Kassenaar, G., Yang, Z., Francois-Lavet, V. (2026). Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation. arXiv:2608.20256. https://arxiv.org/abs/2608.20256
    The model chooses one of three modes (NoThink, Short, Long) as the first token of its response. The choice is learned inside GRPO "with no separate router", using a shaped reward that makes each mode worthwhile at a different length, plus hard per-mode token caps. On a 1.5B distilled model trained on MATH, all three modes survive. MATH500 accuracy is 0.782 vs. 0.796 for the base, with 41% fewer tokens. Transfer to GSM8K gives 76% token savings. **[contrast]** This is the closest prior work to ours: a three-way mode token learned in GRPO within one network. Our differences: our fastest mode is a single forward pass with no generated tokens, the decision is typed (letter logits), the reward is regret against measured per-mode utilities, and we evaluate on multiple-choice/QA mixes, not only math.

22. **[sprague2024cot]** Sprague, Z. et al. (2024). To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning. arXiv:2409.12183. ICLR 2025. https://arxiv.org/abs/2409.12183
    A meta-analysis of over 100 papers plus experiments on 20 datasets finds that CoT's gains are concentrated in math and symbolic/logical tasks. On other tasks (e.g. commonsense MCQ) the gains are small. On MMLU, much of the gain comes from questions containing an "=" sign. **[explains]** This predicts our observation that CoT pays on GSM8K and AQuA but rarely on CommonsenseQA, OpenBookQA and ARC, where the fast modes should win.

23. **[aggarwal2025l1]** Aggarwal, P., Welleck, S. (2025). L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning. arXiv:2503.04697. COLM 2025. https://arxiv.org/abs/2503.04697
    Length-Controlled Policy Optimization (LCPO) trains a model to obey a target length given in the prompt ("think for N tokens"). The reward combines correctness with a penalty on deviation from the target, which gives a smooth compute/accuracy trade-off. **[contrast]** It offers continuous, user-specified budget control. We have discrete, model-chosen modes.

## 4. Latent / continuous reasoning

24. **[hao2024coconut]** Hao, S. et al. (2024). Training Large Language Models to Reason in a Continuous Latent Space (Coconut). arXiv:2412.06769. COLM 2025. https://arxiv.org/abs/2412.06769
    Coconut feeds the last hidden state back as the next input embedding ("continuous thought") instead of decoding a token. A curriculum gradually replaces language CoT steps with latent steps. This enables breadth-first-like search, and on some logic tasks it beats CoT with fewer tokens. **[contrast]** It also moves reasoning out of generated text, but it still takes many sequential forward passes. Our fast path takes one forward pass and does no iterative latent computation.

25. **[zhang2025soft]** Zhang, Z. et al. (2025). Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space. arXiv:2505.15778. https://arxiv.org/abs/2505.15778
    Soft Thinking is training-free. At each reasoning step it feeds the probability-weighted mixture of token embeddings (a "concept token") instead of a sampled token, and it stops early when entropy is low. It reports higher pass@1 with fewer tokens. **[contrast]** It is another way to make the slow path cheaper, not a way to skip it.

26. **[chen2025latentsurvey]** Chen, X. et al. (2025). Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning. arXiv:2505.16782. EMNLP 2026 Findings (per arXiv comment). https://arxiv.org/abs/2505.16782
    A taxonomy of latent CoT methods: token-wise strategies, internal mechanisms, analysis and applications. **[contrast]** It is a pointer for readers.

27. **[zhu2025latentsurvey]** Zhu, R.-J. et al. (2025). A Survey on Latent Reasoning. arXiv:2507.06203. https://arxiv.org/abs/2507.06203
    A second survey covering activation-based recurrence, hidden-state propagation and diffusion-style latent reasoning. **[contrast]** Cite either survey; both exist.

28. **[goyal2023pause]** Goyal, S. et al. (2023). Think before you speak: Training Language Models With Pause Tokens. arXiv:2310.02226. https://arxiv.org/abs/2310.02226
    Appending learnable `<pause>` tokens delays the output and gives the model extra hidden computation. The gains appear only when the model is both pre-trained and fine-tuned with pauses (e.g. +18% EM on SQuAD, +8% on CommonsenseQA for a 1B model). **[contrast]** It adds compute without readable text. Our JEV mode goes the other way and uses the minimum compute.

## 5. Routing between separate models

29. **[chen2023frugalgpt]** Chen, L., Zaharia, M., Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176. https://arxiv.org/abs/2305.05176
    FrugalGPT is an LLM cascade: query cheaper APIs first, and use a learned scorer to decide whether to accept the answer or escalate to a pricier model. It reports matching GPT-4 with up to 98% cost reduction. **[contrast]** The fast/slow decision is made outside the models, after a cheap model has already generated an answer.

30. **[ding2024hybrid]** Ding, D. et al. (2024). Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. arXiv:2404.14618. ICLR 2024. https://arxiv.org/abs/2404.14618
    A BERT-style router predicts the quality gap between a small and a large model for each query, and sends the query to the small model when the predicted gap is small. A tunable threshold trades quality for cost. **[contrast]** Its predicted quality gap is conceptually close to our per-mode utility/regret target, but the router is a separate model choosing between separate LLMs.

31. **[ong2024routellm]** Ong, I. et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665. https://arxiv.org/abs/2406.18665
    RouteLLM trains routers (matrix factorisation, BERT classifier, causal-LLM classifier, similarity-weighted ranking) on Chatbot Arena preference data plus augmentation, to send each query to a strong or a weak model. It cuts cost by more than 2x on MT-Bench etc. without quality loss. **[contrast]** This is the standard separate-router baseline.

32. **[openai2025gpt5card]** OpenAI (Singh, A. et al.) (2025). OpenAI GPT-5 System Card. Published Aug 2025 at openai.com; arXiv copy arXiv:2601.03267. https://arxiv.org/abs/2601.03267 (PDF: https://cdn.openai.com/pdf/8124a3ce-ab78-4f06-96eb-49ea29ffb52f/gpt5-system-card-aug7.pdf)
    GPT-5 is described as "a unified system" made of a fast model (gpt-5-main), a deeper reasoning model (gpt-5-thinking) and a real-time router. The router picks between them based on conversation type, complexity, tool needs and explicit intent, and it is trained continually on user model-switches, preference rates and measured correctness. OpenAI says it plans to merge these capabilities into a single model. **[motivates]**/**[contrast]** This is the highest-profile deployed fast/slow switch, and it uses separate models with an external router. OpenAI's stated aim of folding the router into one model is exactly the single-network direction we study.

## 6. Decision heads and one-step typed decisions on pretrained LMs

33. **[biswas_bevtrain]** avbiswas (n.d.; full author name not stated on the repo page). bev-train: Training JEV-like decision models using Qwen3 with choice-order invariance. GitHub repository. https://github.com/avbiswas/bev-train
    The repo trains "JEV networks" on a Qwen3-0.6B backbone (optionally truncated to 20 layers, with LoRA on the last layers). Three question types are supported: CHOICE, NOUL (yes/no) and SCORE. Options share starting position IDs, and an attention mask hides each option from the others. A small attention head scores the options, so the result does not depend on option order. There is no paper; the repo links video walkthroughs. **[borrowed]** Our "JEV" one-step decision mode takes its name and idea from here. Note one difference: bev-train uses a dedicated order-invariant scoring head, while our JEV mode (as described in the brief) reads option-letter logits in one forward pass of the same LM.

34. **[biswas_bevdata]** avbiswas (n.d.). BEV Decision Mix (Hugging Face dataset; formerly `avbiswas/bev-decision-150K`, now `avbiswas/bev-decision`). https://huggingface.co/datasets/avbiswas/bev-decision-150K
    The default config has 150,000 rows (125,614 train / 24,386 test), and an `all` config has 276,188 rows. Fields are `state`, `questions_json`, `domain`, `question_types` and `question_count`. Each row has CHOICE, NOUL or SCORE questions across domains such as sentiment, retail, tool use, scientific claims and software engineering. Labels come from upstream annotations or rule-based checks, with LLM-rewritten questions. The licence is "unknown"/mixed, so check each upstream licence. **[borrowed]** This is the source data for typed one-step decisions. The blog should cite the new name and the licence caveat.

35. **[radford2018gpt]** Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI technical report. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
    GPT-1 shows that a generatively pre-trained Transformer, fine-tuned with only a linear output layer and task-specific input transformations, improves SOTA on classification, entailment and multiple-choice tasks. **[borrowed]** This is the classical basis for "pretrained LM plus a tiny head is a good one-step classifier".

36. **[devlin2018bert]** Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805. NAACL 2019. https://arxiv.org/abs/1810.04805
    Fine-tuning a pre-trained bidirectional encoder with a single classification layer on the [CLS] representation reached SOTA on GLUE, SQuAD and SWAG. **[borrowed]** It is the canonical evidence that pretrained representations support one-pass decisions.

37. **[hendrycks2020mmlu]** Hendrycks, D. et al. (2020). Measuring Massive Multitask Language Understanding. arXiv:2009.03300. ICLR 2021. https://arxiv.org/abs/2009.03300
    MMLU is a 57-subject multiple-choice benchmark. Its reference evaluation scores the model's next-token probabilities for the answer letters "A"-"D" after the prompt, with no generation. **[borrowed]** Our JEV mode is the same letter-logit readout, used as an action, not only as an evaluation protocol.

38. **[robinson2022mcp]** Robinson, J., Rytting, C. M., Wingate, D. (2022). Leveraging Large Language Models for Multiple Choice Question Answering. arXiv:2210.12353. ICLR 2023. https://arxiv.org/abs/2210.12353
    The paper contrasts "cloze prompting" (scoring each option's text) with "multiple choice prompting" (MCP: show all options and score the answer-symbol token). MCP needs one forward pass and avoids normalisation problems, but it needs high "multiple choice symbol binding" ability, which strong models have. **[borrowed]**/**[explains]** It justifies the single-pass letter decision. Symbol binding can explain JEV failures or position bias on a 4B model.

39. **[holtzman2021sfc]** Holtzman, A. et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. arXiv:2104.08315. https://arxiv.org/abs/2104.08315
    In zero-shot scoring of answer strings, probability mass is split across different surface forms of the same answer. The authors propose Domain-Conditional PMI scoring as a fix. **[explains]** This is why scoring single option letters (one token each) is better than scoring full option strings in our fast path.

## 7. "Models know what they know": self-knowledge and probes

40. **[kadavath2022know]** Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221. https://arxiv.org/abs/2207.05221
    Large LMs are well calibrated on multiple-choice and true/false formats. They can be trained to predict P(IK), the probability that "I know" the answer before answering, with partial generalisation across tasks. **[explains]** This is the core premise that a model can predict, from the question alone, whether its fast answer will be right. Our mode policy is a P(IK)-like decision with three actions.

41. **[azaria2023lying]** Azaria, A., Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. arXiv:2304.13734. https://arxiv.org/abs/2304.13734
    The paper introduces SAPLMA, a classifier on hidden-layer activations that predicts the truthfulness of statements. It beats using the model's own output probabilities, with 71-83% accuracy. **[explains]** Hidden states encode correctness signals that the policy can read when picking a mode.

42. **[kossen2024sep]** Kossen, J. et al. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927. https://arxiv.org/abs/2406.15927
    Linear probes on a single generation's hidden states predict semantic entropy, which normally needs many samples. This approximates uncertainty almost for free and generalises better than accuracy probes. **[explains]** Uncertainty can be read in one pass. This supports making the fast/slow decision before generating.

43. **[orgad2024know]** Orgad, H. et al. (2024). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707. https://arxiv.org/abs/2410.02707
    Truthfulness signals concentrate in specific (exact-answer) tokens. Error detectors do not generalise across datasets. Models sometimes encode the correct answer internally yet generate a wrong one. **[explains]** Two caveats for us: per-dataset calibration of the mode policy may not transfer, and the JEV readout can sometimes beat generated answers.

44. **[zhang2025probe]** Zhang, A. et al. (2025). Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification. arXiv:2504.05419. https://arxiv.org/abs/2504.05419
    Probes on reasoning-model hidden states verify intermediate answers with high accuracy and calibration, and can predict correctness before the answer is fully formed. Using the probe for early exit saves 24% of tokens. **[explains]** It is evidence that "should I keep thinking?" is decodable from hidden states.

## 8. Chain-of-thought faithfulness

45. **[turpin2023unfaithful]** Turpin, M. et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. arXiv:2305.04388. NeurIPS 2023. https://arxiv.org/abs/2305.04388
    Biasing features in the prompt (e.g. reordering few-shot answers so the answer is always "(A)") shift model answers, but the CoT explanations never mention the bias. Accuracy drops by up to 36% on BIG-Bench Hard. **[explains]** The CoT is not necessarily the computation. This weakens the argument that the slow mode is "more transparent" and supports judging modes by utility alone.

46. **[lanham2023faithful]** Lanham, T. et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702. https://arxiv.org/abs/2307.13702
    The authors intervene on CoT by truncating it, adding mistakes, paraphrasing it, or replacing it with filler. How much the answer depends on the CoT varies widely by task, and larger models often depend on it less ("inverse scaling" of faithfulness). **[explains]** On many tasks the answer is fixed before the CoT is written. This matches our finding that a zero-token decision often equals the CoT answer.

## 9. Tools, model and datasets used

47. **[tunix]** Google (2025). Tunix (Tune-in-JAX): a JAX-based LLM post-training library. GitHub, Apache-2.0. https://github.com/google/tunix
    Tunix supports SFT (full and PEFT/LoRA), DPO/ORPO, and RL (PPO, GRPO, GSPO-Token, DAPO, Dr. GRPO), plus agentic multi-turn RL. It provides a BibTeX entry pointing to the repo; there is no paper. **[borrowed]** It is our SFT and GRPO training stack.

48. **[qwen3_4b]** Qwen Team (2025). Qwen3-4B model card. Hugging Face, Apache-2.0. https://huggingface.co/Qwen/Qwen3-4B
    The model has 4.0B parameters (3.6B non-embedding), 36 layers, and a 32K native context (131K with YaRN). Thinking is hard-disabled by `enable_thinking=False`, or toggled per turn by `/think` and `/no_think` when it is enabled. **[borrowed]** This is our base model.

49. **[hu2021lora]** Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
    LoRA freezes pretrained weights and learns low-rank updates ΔW = BA in attention projections. It matches full fine-tuning with orders of magnitude fewer trainable parameters and no inference latency once merged. **[borrowed]** It is our parameter-efficient adaptation.

50. **[cobbe2021gsm8k]** Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168. https://arxiv.org/abs/2110.14168
    The paper introduces GSM8K (8.5K grade-school math word problems with natural-language solutions) and shows that training verifiers to rerank samples beats fine-tuning alone. **[borrowed]** It is a dataset (CoT-favouring).

51. **[ling2017aqua]** Ling, W. et al. (2017). Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems. arXiv:1705.04146 (ACL 2017). https://arxiv.org/abs/1705.04146
    The paper introduces AQuA-RAT, about 100K algebraic word problems with five options and natural-language rationales, and generates rationales and answers jointly. **[borrowed]** It is a dataset (multiple-choice math, where letter decisions and CoT both apply).

52. **[clark2018arc]** Clark, P. et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457. https://arxiv.org/abs/1803.05457
    ARC is a set of grade-school science MCQs split into Challenge (questions that retrieval and co-occurrence baselines fail) and Easy, with a 14M-sentence science corpus. **[borrowed]** It is a dataset.

53. **[mihaylov2018obqa]** Mihaylov, T. et al. (2018). Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. arXiv:1809.02789. EMNLP 2018. https://arxiv.org/abs/1809.02789
    OpenBookQA has about 6K elementary science MCQs that require combining an "open book" core fact with broad common knowledge. **[borrowed]** It is a dataset.

54. **[talmor2019csqa]** Talmor, A. et al. (2019). CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. arXiv:1811.00937. NAACL 2019. https://arxiv.org/abs/1811.00937
    CommonsenseQA has 12,247 five-way MCQs built from ConceptNet relations, with crowd-authored distractors. **[borrowed]** It is a dataset (one where CoT is expected to help little, per [sprague2024cot]).

---

## UNVERIFIED

None. Every reference requested in the brief was found and confirmed. Caveats:
- The OpenAI pages (`openai.com/index/learning-to-reason-with-llms/`, `openai.com/index/gpt-5-system-card/`) return HTTP 403 to automated fetches. Their existence, date and content were confirmed through web search results and, for GPT-5, through the arXiv copy (2601.03267) and the cdn.openai.com PDF (HTTP 200).
- Some venues are not listed above (e.g. for RouteLLM, Thinkless, AdaptThink, Azaria & Mitchell, Kossen et al.), because arXiv metadata does not state them. Add them only after checking.
- The Kahneman book was confirmed through Wikipedia and bookseller/library catalogues (FSG, 2011, ISBN 978-0374275631).

---

## What is known vs what is new here

| Approach | Where the fast/slow decision happens | One network does both? | Does the fast path generate tokens? | How the decision is supervised |
|---|---|---|---|---|
| Always-reasoning RLVR models (o1, DeepSeek-R1, DAPO/Dr. GRPO-trained) | Nowhere explicit: always slow. Length emerges from RL; budgets are set by the user (L1) | Yes (only one mode) | n/a (no fast path) | Outcome correctness reward (RLVR); length penalties or targets in L1-style work |
| Think switches / adaptive thinking (Qwen3 `/think`, Thinkless, AdaptThink, AutoThink, ARM, LHRM, AdaCoT, CPAS/LAGR, Kassenaar et al.) | Inside the model: the first generated control token or think block (Qwen3: user or template) | Yes | Yes. "No-think" still decodes a full answer (ARM's Direct Answer, NoThink, `<short>`) | RL on correctness plus length/cost terms, often after hybrid SFT. Fixes for mode collapse: DeGRPO, Ada-GRPO, importance sampling, SLM, CPAS/LAGR |
| Latent CoT (Coconut, Soft Thinking, pause tokens) | Mostly no fast/slow decision; reasoning moves into hidden space | Yes | Fewer or no *text* tokens, but still many sequential forward passes | Curriculum SFT from language CoT (Coconut); training-free (Soft Thinking); pause pre-training |
| Routers with separate models (FrugalGPT, Hybrid LLM, RouteLLM, GPT-5 router) | Outside the models: a separate router or cascade scorer | No (2+ models plus router) | Yes. The cheap model is a full generator | Router trained on preference data, predicted quality gaps, or user signals (switches, preferences, correctness) |
| Decision heads (GPT-1/BERT heads, MMLU letter scoring, MCP, bev-train JEV) | No fast/slow decision: always the one-step decision | Yes, but only the fast path | No: one forward pass, readout from logits or a head | Supervised labels (cross-entropy) on choices, yes/no or scores |
| **Our approach** (Qwen3-4B: JEV / direct / brief CoT) | Inside the model, per question, as an explicit typed mode choice before any answer text | **Yes**: the same network does the zero-token decision, the short answer and the CoT | **Fastest mode does not**: a single forward pass picks the option letter. The middle mode generates a short answer | SFT then GRPO, with a reward that penalises **regret** of the chosen mode against **measured per-mode utilities** (accuracy minus token cost) |

**What is new, stated carefully.** Adaptive mode choice inside one model is well established (more than 8 papers in 2025-26), and three-way mode menus exist: ARM (4 formats) and Kassenaar et al. 2026 (NoThink/Short/Long, first-token choice inside GRPO). The defensible novelties are therefore:
1. The fast path is a typed zero-generation decision (a letter-logit readout in one forward pass) inside the same model as its CoT mode.
2. The reward is regret against measured per-mode utilities, rather than correctness plus a length penalty.
3. The setting is multiple-choice and decision data, with JEV-style typed decisions, rather than math-only.
