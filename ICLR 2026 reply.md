# ICLR 2026 reply



## Experiments to add

1. Try one experiment with large m or small d or 画饼
2. Show compression to log(d) when m=1 (fig2 task)
3. Show the time complexity of compression vs d (fig2 task)
4. Try compressing a simple attention module

## List of changes

1. Fixed the broken reference in line 190: "Appendix ??”
2. Replaced $\omega(d)$ by $\varepsilon(d)$ to avoid ambiguity. Added examples ($\varepsilon(d) = d^{-\alpha}$ ) to help understand the results of Theorems 4 and 7
3. 
4. 



## Reviewer YxjE

**Summary:**

This work addresses dataset and neural network compression from a moment-matching perspective. Under certain assumptions, this approach establishes novel compression rates and power laws for these tasks. It also enables the boosting of neural power laws, which describe performance versus dataset size dynamics. A number of low-dimensional experiments are conducted to support the claims.

**Soundness:** 3: good

**Presentation:** 3: good

**Contribution:** 2: fair

**Strengths:**

The work is mathematically sound and easy to follow. The text is clear, supported by a decent and concise background overview. The authors provide both rigorous derivations and intuitive explanations for their theoretical results, and the experiments support their claims across a number of settings.

**Weaknesses:**

My main criticism revolves around the **curse of dimensionality**, which the authors underaddress several times throughout the paper.

1. Both (9) and (10) have dimensionality-dependent exponents, which explode when given that other constants are fixed. This is later combated by selecting , which, in turn, explodes . Through some trickery in Theorem 7 (unfortunately, due to time constraints, I was not able to fully verify the math), the authors miraculously balance these issues by attaining a poly-log compression rate.

   That said, one might expect that substituting from (45) into (44) should yield errors which are (asymptotically) under some fixed . However, when done numerically for , , , and any multiplicative constant in (45), I always get an exploding upper bound on the compression error. Reasonable variations of and do not alleviate the issue, which only worsens as grows.

   ==Curse of dimensionality is indeed present and is unavoidable given o==

   ==Stress that our most important point to show that such a compression exists, rather than giving some optimal approach. Indeed we do not expect this to work well when m is large, and it is not possible to do better because we proved optimality. There are ways to go around the curse of dimensionality that are mentioned in Conclusion. We can elaborate more on it.==

2. Since in Theorem 7 grows with increasing , is required to be increasingly smooth. While most contemporary NNs are $\infty$-smooth almost everywhere, their numerical smoothness degrades with increasing dimensionality or a decreasing learning rate [1]. In practice, this will take a toll on the derived bounds in terms of asymptotic constants or other parameters (e.g., in (44)). This problem remains unaddressed in the main text.

   ==Add reference. Admit this as a caveat in the main text==

3. The experimental setups are toy, with the dimensionality being orders of magnitude lower than in real-world tasks. In my opinion, this might lead to the following problems:

   - While showing decent performance in low-dimensional regimes, the proposed compression method might entail overfitting in high-dimensional setups. Stochastic gradient descent (SGD) is known to apply implicit regularization during training [2], thus selecting less overfitting solutions. Your method, however, might "overcompress" a NN/dataset: among all solutions, a non-generalizable one is selected (train error or even dynamics are the same, but test error is not).

     ==Overfitting is irrelevant to how to compress. Because our theory applies to almost any symmetric functions.==

   - It is known that some problems in ML have exponential (in dimensionality) sample complexity (e.g., density estimation). Your result, however, suggests that these problems are also log-exponential in dimensionality (Theorem 7 applied to dataset compression) given the train error is preserved. The only logical conclusion I can arrive at is that such compression almost always entails overfitting when considering complex problems.

     ==WTF==

4. While the authors briefly mention the manifold hypothesis in Section 7, it is not clear how one can use it to improve the method. Moment matching is agnostic to manifolds: i.e., it generally cannot capture such intricate structures. Therefore, another manifold learning strategy must be employed beforehand to decrease the dimensionality. Such a strategy typically requires the full dataset, as manifold learning is usually of exponential sample complexity.

[1] Cohen et al. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability". Proc. of ICLR 2021.

[2] Smith et al. "On the Origin of Implicit Regularization in Stochastic Gradient Descent". Proc. of ICLR 2021.

**Minor issues:**

1. Broken reference in line 190: "Appendix ??"

**Questions:**

1. Can you, please, provide additional experiments (e.g., for high dataset dimensionality or low sampling sizes) proving that your method avoids overfitting?

   ==overfitting is irrelevant to large m==

   ==Add an experiment: 尝试一个高维实验/low sample size/画饼==

2. I kindly ask to address my concerns in Weakness 1. In particular, I am interested in the numerical verification of the bounds provided.

   ==首先fig2已经是在直接做verification了（not for polylog） 但是我们可以做更好==

   ==尝试做个m=1实验 达到polylog==

**Rating:** 4: marginally below the acceptance threshold. But would not mind if paper is accepted

**Confidence:** 3



## Reviewer 1dtD

**Summary:**

The paper proves a universal compression theorem, showing that almost any symmetric function of $d$ elements can be compressed to a function with $O({\rm polylog}$ (d)) elements losslessly. The theory leads to two key applications. First is the dynamical lottery ticket hypothesis, proving that large networks can be compressed to polylogarithmic width while preserving their training dynamics. Second is dataset compression, demonstrating that neural scaling laws can be theoretically improved from power-law to stretched-exponential decay.

**Soundness:** 3: good

**Presentation:** 4: excellent

**Contribution:** 4: excellent

**Strengths:**

- The paper delivers a rigorous theoretical result that proves the dynamical lottery ticket hypothesis by showing that large networks can be compressed while preserving their original training dynamics.
- Provides a generalized compression theory with broad applicability across diverse domains (e.g., dataset and model compression), demonstrating strong theoretical versatility and significant potential for cross-domain impact.
- Establishes clear practical advantages, such as improved scaling laws and model compression, that are well grounded in the proposed theoretical framework.

**Weaknesses:**

- The paper lacks a thorough discussion on the applicability of the proposed theory to complex neural architectures such as Transformer blocks, which integrate linear projections, attention mechanisms, and normalization layers.

  ==考虑做一个简单transformer实验 4-6个参数==

  ==展开讲line 101-103==

- There seems to be a missing reference link to the Appendix at line 190 on page 4 (“Appendix ??”).

**Questions:**

- The model assumes neuron permutation symmetry. Does the assumption is applicable to complex modules in neural networks, such as Transformer block?

- In experiments such as Figure 3 or 4, how much real computation time does the proposed compression take?

  ==加个小实验scale压缩所需时间==

**Rating:** 6: marginally above the acceptance threshold. But would not mind if paper is rejected

**Confidence:** 3



## Reviewer ui4S

**Summary:**

The paper introduces the universal compression theorem as a step towards the dynamical lottery ticket hypothesis (LTH), which claims that in a dense network there exists a subnetwork, which when trained in isolation exhibits the same training dynamics as the original one. The theorem states (informally) that a permutation-invariant function of variables each of dimensionality can be asymptotically compressed to a function of variables. The authors argue that, because many model / dataset objects are symmetric in parameters / datapoints, these results imply polylog-rate network and dataset compression under the assumptions of the theorem. Another implication of polylog compression is the scaling law changing from power law form to stretched-exponential form , both for model and dataset size.

**Soundness:** 3: good

**Presentation:** 4: excellent

**Contribution:** 3: good

**Strengths:**

1. The paper provides theoretical guarantees on asymptotic polylogarithmic compression for symmetrical functions. The authors provide Algorithm 1 for compression of symmetric functions using moment-matching and validate it numerically.
2. An important feature is the universality of the result: the implications of the theorem include both neural networks and datasets.
3. A major practical consequence of the work is the potential speed up guarantees on the power-law scaling laws, which are known to "be slow", i.e. have small power exponentials.
4. Although the main result is theoretical, the authors back each claim with numerical experiments: they show on a synthetic function that compression error drops with in agreement with the theoretical bound (Fig. 2); that training dynamics on a compressed dataset follows training on the full dataset (Fig. 3); training performances of full and compressed models are identical to support dynamical LTH (Fig. 4); and compressing a network or dataset leads to a larger scaling law exponent (Fig. 5). These comprehensive validations neatly complement the theoretical backbone of the paper.

**Weaknesses:**

1. Further empirical evaluation would strengthen this work, as the authors note.

2. The proposed moment-matching algorithm scales poorly with moment order and dimension (via ), which limits immediate practical effects despite the asymptotic guarantees.

3. The theoretical claim of polylogarithmic compression yielding a stretched-exponential scaling is not supported with evidence. The numerical experiments in Section 6 demonstrate how the scaling laws can be improved only for quadratic compression.

   ==用1维压缩到polylog的实验来回答这个问题==

**Questions:**

1. Can you show an example with the scaling laws of a form to illustrate the stretched-exponential regime?

2. In numerical experiments in Section 6 the exponent should have improved by a factor of 2: . The reported values are close but lower, 1.271 vs and 0.608 vs . Why does this difference appear? And why is it larger for dataset compression? 

3. Many elements of modern neural networks do not fall under the smoothness assumptions, like ReLU, top-k selections, sparse \ quantized representations. How do you imagine expanding your work around those limitations and how would compression rates be affected?

   ==你说的对。光滑性和可压缩性的定量关系是很有意思的future work. Empirically用很小的k就已经很有用了==

**Rating:** 8: accept, good paper (poster)

**Confidence:** 2



## Reviewer LiWj

**Summary:**

The paper studies how neural networks and datasets can be compressed by exploiting permutation symmetries. The authors show that symmetric functions can be represented using fewer variables, which implies that both the model and the data can be reduced to polylogarithmic size without significantly changing the loss. This leads to what the authors call a dynamical lottery ticket hypothesis and stronger scaling laws.

**Soundness:** 2: fair

**Presentation:** 2: fair

**Contribution:** 2: fair

**Strengths:**

The paper presents an interesting idea: using permutation symmetry to achieve strong compression of both networks and datasets. The theoretical argument (that symmetric functions can be represented with fewer variables), is promising. The results aim to connect model compression, scaling laws, and the lottery ticket hypothesis in a unified framework.

**Weaknesses:**

The paper proposes a theoretical link between symmetry, compression, and scaling laws. However, the lack of clear algorithmic formulation and the absence of fair experimental baselines limit its current practical relevance.

The main limitation of the paper is the lack of rigor and clarity. The compression process is described only at a high level. It is not clear how one would actually construct the compressed network or dataset in practice. The paper does not include pseudocode or complexity estimates, making it hard to evaluate the tractability of the proposed methods. 

==Fuck you (complain to AC)==

The experimental comparison is incomplete. The proposed compressed network is compared with both the original network and a random sparse network. However, it is already known that random sparse networks perform poorly, while sparse networks obtained with *Iterative Magnitude Pruning* (IMP, Frankle & Carbin 2019) can match the performance of dense ones. A fair comparison should therefore include IMP or other modern sparse training methods.

==I think comparing IMP with our method is unfair, because IMP is like train->prune->wind back->train->…, but my approach does not rely on training at all so there’s one compression map that works for training anything. Though IMP could be better, but it mainly benefits from using info from the training so the comparison is unfair.==

The compression-error trade-off is not clearly quantified. The claim that a network with (d) parameters can be reduced to polylogarithmic size should be expressed as a function of the error, and possibly compared to existing theoretical bounds. Finally, some parts of the theoretical presentation are unclear. The meaning of the function in Theorem 5 is not explained, and the notation is confusing, since can mean any function that grows faster than , but such a bound would be vacuous.

==example: when $\omega(d)=d^{-\alpha}$, derive the bound for d’, which is still polylog==

换字母

**Questions:**

- Could you provide a concrete description of the compression algorithm? How are the compressed parameters and datasets obtained from the original ones?

- How does your method compare, both in compression ratio and performance, with Iterative Magnitude Pruning or other sparse training techniques?

- Can you explicitly state the trade-off between compression and approximation error, and how it compares with previous results (e.g. to those for the Strong LTH such as Pensia et al., 2020)?

- What exactly does the function f represent in Theorem 5? You didn't define it. Can you clarify the notation?

- Can you argue that the bound is not vacuous?

  ==Answered by (1) proof of optimality (2) Find a function f that matches the error upper bound in Eq. (9)==

**Rating:** 4: marginally below the acceptance threshold. But would not mind if paper is accepted

**Confidence:** 4





## Official comment

Stress that we focus on the existence of a universal compression. The polylog compression rate is already the first theoretical result and is unparalleled (maybe except Surya’s paper). We are not here to propose something without theoretical ground and just trying to show it is superior to previous approaches by numerics. 