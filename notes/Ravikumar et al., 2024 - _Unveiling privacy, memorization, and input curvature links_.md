# Unveiling Privacy, Memorization, and Input Curvature Links 

Deepak Ravikumar ${ }^{1}$ Efstathia Soufleri ${ }^{1}$ Abolfazl Hashemi ${ }^{1}$ Kaushik Roy ${ }^{1}$


#### Abstract

Deep Neural Nets (DNNs) have become a pervasive tool for solving many emerging problems. However, they tend to overfit to and memorize the training set. Memorization is of keen interest since it is closely related to several concepts such as generalization, noisy learning, and privacy. To study memorization, Feldman (2019) proposed a formal score, however its computational requirements limit its practical use. Recent research has shown empirical evidence linking input loss curvature (measured by the trace of the loss Hessian w.r.t inputs) and memorization. It was shown to be $\sim 3$ orders of magnitude more efficient than calculating the memorization score. However, there is a lack of theoretical understanding linking memorization with input loss curvature. In this paper, we not only investigate this connection but also extend our analysis to establish theoretical links between differential privacy, memorization, and input loss curvature. First, we derive an upper bound on memorization characterized by both differential privacy and input loss curvature. Second, we present a novel insight showing that input loss curvature is upper-bounded by the differential privacy parameter. Our theoretical findings are further empirically validated using deep models on CIFAR and ImageNet datasets, showing a strong correlation between our theoretical predictions and results observed in practice.


## 1. Introduction

Machine learning and deep learning approaches have become state-of-the-art solutions in many learning tasks such as computer vision, natural language processing, etc. However, Deep Neural Nets (DNNs) are prone to over-fitting and memorization. An increasingly larger number of recent literature has focused on understanding memorization in

[^0]![](https://cdn.mathpix.com/cropped/2025_11_08_eddebc1656fc9aee8f2fg-01.jpg?height=457&width=709&top_left_y=601&top_left_x=1116)
Figure 1. Our theoretical framework provides upper bounds in Theorems 5.1, 5.3, and 5.4. These are visualized as links between Differential Privacy, Memorization, and Input Loss Curvature.

DNNs (Zhang et al., 2017; Arpit et al., 2017; Carlini et al., 2019; Feldman \& Vondrak, 2019; Feldman \& Zhang, 2020; Feldman, 2019). This is crucial given the implications to several connected areas such as generalization (Zhang et al., 2021; Brown et al., 2021), noisy learning (Liu et al., 2020), identifying mislabelled examples (Maini et al., 2022), identifying rare and hard examples (Carlini et al., 2019), privacy (Feldman, 2019), risks from membership inference attacks (Shokri et al., 2017; Carlini et al., 2022) and more.

To study memorization several metrics have been suggested. Carlini et al. (2019) proposed a combination of five metrics to analyze memorization. Alternatively, Jiang et al. (2020) proposed using a computationally efficient proxy to C-score, a metric closely related to the stability-based memorization (Feldman, 2019). The stability-based memorization score proposed by Feldman (2019) measures the change in expected output probability when the sample under investigation is removed from the training dataset. Additionally, unlike other proposed metrics, Feldman (2019) provides a strong theoretical framework for understanding memorization. This theory was then tested in practice in a subsequent paper (Feldman \& Zhang, 2020). However, their method involved training thousands of models and is thus computationally infeasible in most real applications.

In a recent paper, Garg et al. (2023) suggested a new proxy using input loss curvature to measure the stability-based memorization score proposed in Feldman (2019). To measure input loss curvature they suggested using the trace of the loss Hessian with respect to the input. Using this in-

![](https://cdn.mathpix.com/cropped/2025_11_08_eddebc1656fc9aee8f2fg-02.jpg?height=411&width=1446&top_left_y=243&top_left_x=308)
Figure 2. Images from ImageNet ranked using input loss curvature. Input loss curvature was obtained using a single ResNet18 trained on ImageNet. Ten lowest curvature samples (left) and ten highest curvature samples (right) from the training set are visualized for 5 classes (each row is a class) from ImageNet. Low curvature samples are 'prototypical' of their class, while high curvature samples are rare, difficult, and more likely memorized instances.

put loss curvature measurement, they provided empirical evidence on the link between memorization and input loss curvature. They obtained high cosine similarity between input loss curvature and memorization scores from Feldman \& Zhang (2020) while being $\sim 3$ orders of magnitude less compute-intensive. To illustrate the savings we reproduced Garg et al. (2023)'s results on ImageNet and visualized the ten lowest and highest curvature samples in Figure 2. These examples were obtained using a single ResNet18 model trained on ImageNet, compared to 1000's of models trained by Feldman \& Zhang (2020) to compute memorization scores. From Figure 2, we see that low curvature samples are 'prototypical' of their class. While high curvature samples are drawn from rare, hard, or outlier examples which are more likely to be memorized.

Input loss curvature is thus, a promising proxy for stabilitybased memorization score. However, there is a lack of theoretical understanding of the link between memorization, and input loss curvature. In this paper, we develop a theoretical framework to understand this empirical observation and formally unveil the connections between memorization and input loss curvature. Further, we explore the relationship beyond memorization and input loss curvature, our theoretical contributions reveal the underlying link between differential privacy (Dwork et al., 2006), memorization (Feldman, 2019), and input loss curvature (Garg et al., 2023). We present the links as three theorems. The first links memorization and input loss curvature, and the second theorem links input loss curvature and differential privacy. The third theorem links differential privacy and memorization. These links are visualized in Figure 1. Each of the three theoretical links developed in this paper is corroborated by empirical evidence obtained on DNNs used for vision classification tasks.

In summary, the main contributions of this paper include:

- We develop a theoretical framework for analyzing input loss curvature and memorization in a general learning setting and demonstrate its implications to DNNs.
- We show that memorization is upper bounded by (a) input loss curvature and (b) relevant privacy parameters. We also show that input loss curvature is also upper bounded by privacy, completing the theoretical links between memorization, privacy, and input loss curvature.
- We verify the theoretical results with extensive empirical experiments on vision classification tasks using DNNs on CIFAR100 and ImageNet datasets.
- We obtain a tighter bound on private learnability. Namely, we establish that $\epsilon$-differential privacy implies $L\left(1-e^{-\epsilon}\right)$ stability, thereby improving the previous theoretical bound.


## 2. Preliminaries and Notation

Consider a supervised learning problem where the goal is to learn a mapping from some input space $\mathcal{X} \subset \mathbb{R}^{d}$ to an output space $\mathcal{Y} \subset \mathbb{R}$. The learning is performed using a randomized algorithm $\mathcal{A}$ on a training set $S$. A randomized algorithm employs a degree of randomness as a part of its logic. The training set $S$ contains $m$ elements. Each element $z_{i}=\left(x_{i}, y_{i}\right)$ is drawn from an unknown distribution $\mathcal{D}$, where $z_{i} \in \mathcal{Z}, x_{i} \in \mathcal{X}, y_{i} \in \mathcal{Y}$ and $\mathcal{Z}= \mathcal{X} \times \mathcal{Y}$. Thus we define the training set $S \in \mathcal{Z}^{m}$ as $S= \left\{z_{1}, \cdots, z_{m}\right\}$. We assume $m \geq 2$. Another very relevant concept is adjacent datasets. Adjacent datasets are obtained when the $i^{\text {th }}$ element is removed. This is sometimes referred to as a leave-one-out set defined as

$$
S^{\backslash i}=\left\{z_{1}, \cdots, z_{i-1}, z_{i+1}, \cdots, z_{m}\right\}
$$

Related to adjacent (a.k.a neighboring) datasets is the distance between datasets. The distance between any two datasets $S, S^{\prime}$ denoted by $\left\|S-S^{\prime}\right\|_{1}$ is a measure of how many samples differ between $S$ and $S^{\prime}$. Note, $\|S\|_{1}$ denotes the size of a dataset $S$.

A randomized learning algorithm $\mathcal{A}$ when applied on a dataset $S$ results in a hypothesis denoted by $h_{S}^{\phi}=\mathcal{A}(\phi, S)$, where $\phi \sim \Phi$ is the random variable associated with the randomness of the algorithm $\mathcal{A}$. A cost function $c: \mathcal{Y} \times \mathcal{Y} \mapsto \mathbb{R}^{+}$ is used to measure the performance of the hypothesis. The cost of the hypothesis $h$ at a sample $z_{i}$ is also referred to as the loss $\ell$ at $z_{i}$ defined as

$$
\ell\left(h, z_{i}\right)=c\left(h\left(x_{i}\right), y_{i}\right)
$$

In most cases, we are interested in the loss of $h$ over the data distribution, which is referred to as population risk defined as

$$
R(h)=\mathbb{E}_{z \sim \mathcal{D}}[\ell(h, z)]
$$

Since the data distribution $\mathcal{D}$ is unknown in general, it is common to evaluate and study the empirical risk defined as

$$
R_{e m p}(h, S)=\frac{1}{m} \sum_{i=1}^{m} \ell\left(h, z_{i}\right), \quad z_{i} \in S
$$

In this paper, our characterization of curvature involves the gradient and the Hessian of the loss with respect to the input data, which is denoted using the $\nabla$ and $\nabla^{2}$ operators respectively. $\|a\|$ denotes the $\ell_{2}$ norm of $a$.

## 3. Background

Differential Privacy was introduced by Dwork et al. (2006) and here we briefly recall the definition. A randomized algorithm $\mathcal{A}$ with domain $\mathcal{Z}^{m}$ is $\epsilon$-differentially private if for all $\mathcal{R} \subset \operatorname{Range}(\mathcal{A})$ and for all $S, S^{\prime} \in \mathcal{Z}^{m}$ such that $\left\|S-S^{\prime}\right\|_{1} \leq 1$

$$
\begin{equation*}
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi} \in \mathcal{R}\right] \leq e^{\epsilon} \operatorname{Pr}_{\phi}\left[h_{S^{\prime}}^{\phi} \in \mathcal{R}\right] \tag{1}
\end{equation*}
$$

where the probability is taken over the randomness arising from the algorithm $\mathcal{A}, \phi \sim \Phi$.
Memorization of the $i^{\text {th }}$ element $z_{i}=\left(x_{i}, y_{i}\right)$ of the dataset $S$ by an algorithm $\mathcal{A}$ was defined by Feldman (2019) using the notion of stability as:

$$
\begin{equation*}
\operatorname{mem}(\mathcal{A}, S, i)=\operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]-\operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] \tag{2}
\end{equation*}
$$

where the probability is taken over the randomness of algorithm $\mathcal{A}$.

Error Stability of a possibly randomized algorithm $\mathcal{A}$ for some $\beta>0$ is defined as Kearns \& Ron (1997)
$\forall i \in\{1, \cdots, m\},\left|\mathbb{E}_{\phi, z}\left[\ell\left(h_{S}^{\phi}, z\right)\right]-\mathbb{E}_{\phi, z}\left[\ell\left(h_{S \backslash i}^{\phi}, z\right)\right]\right| \leq \beta$,
where $z \sim \mathcal{D}$ and $\phi \sim \Phi$.

Generalization. A randomized algorithm $\mathcal{A}$ is said to generalize with confidence $\delta$ and a rate $\gamma^{\prime}(m)$ if

$$
\begin{equation*}
\operatorname{Pr}\left[\left|R_{e m p}(h, S)-R(h)\right| \leq \gamma^{\prime}(m)\right] \geq \delta . \tag{4}
\end{equation*}
$$

Uniform Model Bias. The hypothesis $h$ resulting from the application of algorithm $\mathcal{A}$ to learn the true conditional $h^{*}=\mathbb{E}[y \mid x]$ from a dataset $S \sim \mathcal{D}^{m}$ has uniform bound on model bias given by $\Delta$ if

$$
\begin{equation*}
\forall S \sim \mathcal{D}^{m}, \quad\left|\mathbb{E}_{\phi}\left[R\left(h_{S}^{\phi}\right)-R\left(h^{*}\right)\right]\right| \leq \Delta \tag{5}
\end{equation*}
$$

$\rho$-Lipschitz Hessian. The Hessian of $\ell$ is Lipschitz continuous on $\mathcal{Z}$ if $\forall z_{1}, z_{2} \in \mathcal{Z}$, and $\forall h \in \operatorname{Range}(\mathcal{A})$ if there exists some $\rho>0$ such that

$$
\begin{equation*}
\left\|\nabla_{z_{1}}^{2} \ell\left(h, z_{1}\right)-\nabla_{z_{2}}^{2} \ell\left(h, z_{2}\right)\right\| \leq \rho\left\|z_{1}-z_{2}\right\| . \tag{6}
\end{equation*}
$$

Input Loss Curvature. Using the notation of curvature from Moosavi-Dezfooli et al. (2019); Garg et al. (2023), input loss curvature is defined as the sum of the absolute eigenvalues of the Hessian $H$ of the loss with respect to input $z_{i}$, conveniently it can be written using the trace as

$$
\begin{equation*}
\operatorname{Curv}_{\phi}\left(z_{i}, S\right)=\operatorname{tr}(H)=\operatorname{tr}\left(\nabla_{z_{i}}^{2} \ell\left(h_{S}^{\phi}, z_{i}\right)\right) \tag{7}
\end{equation*}
$$

$v$-adjacency. A dataset $S$ is said to contain $v$-adjacent (read as upsilon-adjacent) elements if it contains two elements $z_{i}, z_{j}$ such that $z_{j}=z_{i}+\alpha$ for some $\alpha \in B_{p}(v)$ (read as $v$-Ball). Note that this can be ensured through construction. Consider a dataset $S^{\prime}$ which has no $z_{j}$ s.t $z_{j}=z_{i}+\alpha ; z_{j}, z_{i} \in S^{\prime}$. Then we can construct $S$ such that $S=\left\{z \mid z \in S^{\prime}\right\} \cup\left\{z_{i}+\alpha\right\}$ for some $z_{i} \in S^{\prime}, \alpha \in B_{p}(v)$, ensuring $v$-adjacency holds.

## 4. Related Work

Input Loss Curvature is a measure of the sensitivity of the model to a specific input. Loss curvature with respect to weight parameters has received significant attention (Keskar et al., 2017; Wu et al., 2020; Jiang* et al., 2020; Foret et al., 2021; Kwon et al., 2021; Andriushchenko \& Flammarion, 2022), recently regarding its role in characterizing the sharpness of a learning objective and its connection to generalization. However, input loss curvature has received less focus. Input loss curvature has been studied in the context of adversarial robustness (Fawzi et al., 2018; MoosaviDezfooli et al., 2019), coresets (Garg \& Roy, 2023) and recently as a proxy for memorization (Garg et al., 2023). Moosavi-Dezfooli et al. (2019) showed that adversarial training decreases the curvature of the loss surface with respect to inputs. Further, they provided a theoretical link between robustness and curvature and proposed using curvature regularization. Garg \& Roy (2023) identified samples with
low curvature as being more data－efficient and developed a coreset identification and training algorithm based on input loss curvature．In an interesting application of input loss curvature，Garg et al．（2023）provided empirical evidence linking memorization and input loss curvature．

Memorization has garnered increasing research effort with several recent works aiming to add to the understanding of memorization and its implications（Zhang et al．，2017；Arpit et al．，2017；Carlini et al．，2019；Feldman \＆Vondrak，2019； Feldman，2019；Feldman \＆Zhang，2020；Maini et al．，2022； Garg et al．，2023；Lukasik et al．，2023）．The motivation for studying memorization stems from a variety of goals rang－ ing from deriving insights into generalization（Zhang et al．， 2017；Toneva et al．，2019；Brown et al．，2021；Zhang et al．， 2021），identifying mislabeled examples（Pleiss et al．，2020； Maini et al．，2022），and identifying challenging or rare sub－ populations（Carlini et al．，2019），to understanding privacy （Feldman，2019）and robustness risks from memorization （Shokri et al．，2017；Carlini et al．，2022）．While several metrics have been proposed to study memorization（Carlini et al．，2019；Jiang et al．，2020），the stability－based memoriza－ tion score proposed by Feldman（2019）provides a strong theoretical framework to understand memorization along with strong empirical evidence（Feldman \＆Zhang，2020）． However，since the score proposed by Feldman（2019）is computationally expensive，Garg et al．（2023）proposed us－ ing input loss curvature as a more compute－efficient proxy． In this paper，we develop the theoretical framework to under－ stand the links between input loss curvature，memorization， and differential privacy．

Influence Functions were applied to deep learning by Koh \＆Liang（2017）and are closely related to memorization．In－ fluence functions aim to identify the impact of one training point on the model predictions．Influence functions try to answer the counterfactual：what would have happened if a training point were absent，or if its values were changed slightly？While recent approaches（Schioppa et al．，2022） have applied influence functions to large deep models，in－ fluence functions have been criticized（Basu et al．，2021； Bae et al．，2022；Schioppa et al．，2023）since the underlying theory assumes strong convexity and positive definiteness of the Hessian，conditions that are not met in the context of DNNs．On the other hand，the theoretical framework we present in this paper does not make any assumptions about the convexity or the definiteness of the Hessian and is more suitable for studying deep learning．

## 5．Linking Privacy，Memorization and Input Loss Curvature

In this section，we discuss our theoretical contributions as three links．First，we present Theorem 5.1 which links memorization and curvature．Second，we present Theorem
5.3 which links privacy and curvature．Finally，we present Theorem 5.4 which links memorization and privacy．

## 5．1．Memorization and Input Loss Curvature

The association between memorization and input loss curva－ ture may initially appear counterintuitive at first，but a closer examination reveals a fundamental connection．Both met－ rics intrinsically capture the sensitivity of a model to input perturbations．Here we provide a theoretical link between memorization and input curvature in the form of Theorem 5．1．Theorem 5.1 is one of our core contributions．
Theorem 5.1 （Curvature Upper Bounds Memorization）．Let the assumptions of error stability 3，generalization 4，and uniform model bias 5 hold and assume the $v$－adjacency of the dataset and that the loss is bounded such that $0 \leq \ell \leq L$ ． Then with probability at least $1-\delta$ it holds

$$
\begin{gather*}
|\operatorname{mem}(\mathcal{A}, S, i)| \leq \frac{1}{L} \mathbb{E}_{\phi}\left[\operatorname{Curv}_{\phi}\left(z_{i}, S^{\backslash i}\right)\right]+c_{1}  \tag{8}\\
c_{1}=\frac{\rho}{6 L} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]+\frac{m \beta}{L}+\frac{(4 m-1) \gamma}{L}+\frac{2(m-1) \Delta}{L} \tag{9}
\end{gather*}
$$

Sketch of Proof．Using the result from Nesterov \＆Polyak （2006）we obtain an upper bound on the loss at $z_{j}$ involving the Hessian of the loss．By choosing $\alpha$ such that $\mathbb{E}[\alpha]=0$ we get rid of the first－order terms．Then by taking expecta－ tion over the randomness of the algorithm and then perform－ ing algebraic manipulation we can show that the expected difference in loss at $z_{i}$ for two different models is upper bound by the result in Theorem 5．1．The final step is to make the connection that for bounded loss，the difference in loss at $z_{i}$ for two different models is a scaled version of memorization．The full proof is provided in Appendix A．3．
Interpreting the Theory．Theorem 5.1 （Equation 8）in－ dicates a linear relationship between memorization and input loss curvature．Observe that the upper bound is de－ pendent on the input loss curvature of a sample $z_{i}$ and the offset factor $c_{1}$ ．However，the offset factor $c_{1}$ is data in－ dependent，i．e．$c_{1}$ has no dependence on $z_{i}$ ．The offset factor $c_{1}$（Equation 9）consists of the following components． The third moment of the perturbation random variable $\alpha$ ， which is a measure of the skewness of the distribution．By choosing the distribution of $\alpha$ carefully，e．g．a centralized Gaussian，this can be made zero．The second and third terms of Equation 9 are properties of the training algorithm，i．e． the algorithm＇s stability $\beta$ and ability to generalize $\gamma$ ．The last term is dependent on model bias $\Delta$ ．Thus $c_{1}$ is roughly

$$
c_{1}=\text { Stability }+ \text { Generalization }+ \text { Model Bias }
$$

To answer the question，＇How tight is the upper bound？＇， we use empirical evaluation of curvature and memorization scores in Section 6.3 and find that the linear relationship
from Equation 8 does hold true. We briefly and qualitatively discuss the validity of our assumptions in practical settings. Research (Hardt et al., 2016) has shown that using stochastic gradient methods (such as stochastic gradient descent) to train models attains small generalization error. Further, it has been shown that stochastic gradient is uniformly stable (Hardt et al., 2016). Thus the assumptions of stability (Equation 3) and generalization (Equation 4) are reasonable. Model bias is a property of the model, and a uniform bound across different datasets seems reasonable. And finally, the $v$-adjacency can be ensured through construction. In practice, this might not be needed because the size of the ball $B_{p}(v)$ is unconstrained. Thus, two samples from the same class that are 'similar' may be sufficient to satisfy this requirement (note that this will result in a non-zero first term of Equation 9). With the size of modern datasets, this assumption is also reasonable.

Remark. Without assuming loss boundedness, we can state Theorem 5.1 for cross-entropy replacing $L$ with 1 , if

$$
\forall h \in \operatorname{Range}(\mathcal{A}), \forall k \quad 0<\operatorname{Pr}\left[h\left(x_{k}\right)=y_{k}\right]<1 .
$$

Note the boundary condition that probability cannot be exactly 0 or 1 . This is a reasonable assumption in a practical setting. The proof is provided in Appendix A.4. The main takeaway is that when the loss is bounded the expected difference in loss is the same as the memorization score, however when the loss is cross entropy the expected difference in loss upper bounds the memorization score.

### 5.2. Privacy and Input Loss Curvature

In this section, we present the second link between input loss curvature and privacy. To make the connection between input curvature and privacy we leverage stability. To establish the curvature-privacy link we first present Lemma 5.2 which links the stability constant and privacy. In doing so, we further improve the bounds in Wang et al. (2016), from $L\left(e^{\epsilon}-1\right)$ to $L\left(1-e^{-\epsilon}\right)$.
Lemma 5.2 (Privacy $\Longrightarrow$ Stability). Assume boundedness of the loss, i.e., $0 \leq \ell \leq L$. Then, any $\epsilon$-differential private algorithm satisfies $L\left(1-e^{-\epsilon}\right)$-stability.

Sketch of Proof We start with the difference in the expected loss of adjacent datasets. Next, we assume that models resulting from training on $S$ and $S^{\backslash i}$ for some $i$ have distributions $p$ and $p^{\prime}$, respectively. We use the properties of the expectation operator to expand the resultant terms. Next, by upper bounding the expectation using loss boundedness and performing some algebraic manipulations we arrive at the result. The full proof is provided in Appendix A.5.

Here we present our second main contribution in the form of Theorem 5.3 linking privacy and input loss curvature.
Theorem 5.3 (Privacy $\Longrightarrow$ Low Input Loss Curvature). Let $\mathcal{A}$ be a randomized algorithm which is $\epsilon$-differentially
private and the assumptions of error stability 3, generalization 4, and uniform model bias 5 hold. Further, assume $0 \leq \ell \leq L$. Then for two adjacent datasets $S, S^{\backslash i} \sim \mathcal{D}$ with a probability at least $1-\delta$ we have

$$
\begin{gather*}
\mathbb{E}_{z, \phi}\left[\operatorname{Curv}_{\phi}(z, S)\right] \leq L(m+1)\left(1-e^{-\epsilon}\right)+c_{2}  \tag{10}\\
c_{2}=(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right] \tag{11}
\end{gather*}
$$

Sketch of Proof Starting at Lemma A. 4 we take the expectation over $z$, then we use the stability assumption. Rearranging the expressions and using Lemma 5.2 we arrive at the result. The full proof is provided in Appendix A.7.

Interpreting Theorem 5.3. Focusing on Equation 10, we see that a stronger privacy guarantee ensures reduced average input loss curvature. To validate the tightness of the bound we use empirical evaluations of curvature and privacy in Section 6.4. Similar to Theorem 5.1, $c_{2}$ can be thought of as having two components, the generalization term $\gamma$ and the model bias term $\Delta$. By choosing $\alpha$ carefully (see previous discussion on Theorem 5.1) the last term can be ignored. Thus $c_{2}$ can be thought as $c_{2}=$ Generalization + Model Bias. The validity of our assumptions in practical settings is reasonable as previously discussed for Theorem 5.1.

### 5.3. Privacy and Memorization

The definition of stability-based memorization (Feldman, 2019) is very much related to privacy. Notably, Feldman (2019) explored this link, demonstrating that under specific conditions, algorithms that do not memorize cannot achieve optimal generalization performance. Feldman (2019) showed that this memorization-generalization result stems from the long-tailed nature of data. Our exploration in determining how memorization and privacy are related, is, however, different. In particular, we show that the memorization score is upper bounded by $1-e^{-\epsilon}$ for any $\epsilon$-DP algorithm. While this result is relatively straightforward, we state it for completeness as it is still a critical link in understanding memorization.
Theorem 5.4 (Privacy $\Longrightarrow$ Less Memorization). Let $\mathcal{A}$ be an $\epsilon$-differentially private algorithm and $z_{i}$ be the $i^{\text {th }}$ element of $S \in \mathcal{Z}^{m}$. Then, we have

$$
\begin{equation*}
\forall i \in\{1, \cdots, m\}, \quad \operatorname{mem}(\mathcal{A}, S, i) \leq 1-e^{-\epsilon} \tag{12}
\end{equation*}
$$

Sketch of Proof We start with the definition of $\epsilon$-differential privacy, with simple algebraic manipulation, and repetitively using the definition of $\epsilon$-differential privacy we arrive at the result. Note that this result can also be readily extended to ( $\epsilon, \delta_{p}$ )-differential privacy, i.e. Theorem 5.4 holds for an ( $\epsilon, \delta_{p}$ )-differential private algorithm with a probability $1-\delta_{p}$. The full proof is provided in Appendix A.1.

## 6. Experiments

### 6.1. Experimental Setup

Datasets. To evaluate our theory we consider the classification task using standard vision datasets as the pre-computed stability-based memorization scores from Feldman \& Zhang (2020) are available for CIFAR100 (Krizhevsky et al., 2009) and ImageNet (Russakovsky et al., 2015) datasets.

Architectures. For some experiments we train ResNet18 (He et al., 2016) models from scratch, while for others we use pre-trained Small Inception (Szegedy et al., 2015) and ResNet50 models released by Feldman \& Zhang (2020). Details regarding the model used are specified at the beginning of each experiment section.

Training. For experiments that use private models, we use the Opacus library (Yousefpour et al., 2021) to train ResNet18 models for 20 epochs till the privacy budget is reached. We use DP-SGD (Abadi et al., 2016) with the maximum gradient norm set to 1.0 and privacy parameter $\delta=1 \times 10^{-5}$. The initial learning rate was set to 0.001 . The learning rate is decreased by 10 at epochs 12 and 16 . When training on CIFAR10 and CIFAR100 datasets the batch size is set to 128 . For both CIFAR10 and CIFAR100 datasets, we used the following sequence of data augmentations for training: resize ( $32 \times 32$ ), random crop, and random horizontal flip, this is followed by normalization.
Testing. During testing no augmentations were used, i.e. we used resize followed by normalization. When using pretrained models from Feldman \& Zhang (2020) we validated the accuracy of the models before performing experiments. To improve reproducibility, we have provided the code in the supplementary material.

### 6.2. Estimating Input Loss Curvature

To corroborate the theoretical findings presented in the prior section, an efficient methodology for computing input loss curvature is needed as computing the full Hessian is computationally intensive. Since we are interested in the trace of the Hessian, it can be efficiently computed using Hutchinson's trace estimator (Hutchinson, 1989; Garg et al., 2023) from which we have

$$
\begin{equation*}
\operatorname{tr}(H)=\mathbb{E}_{v}\left[v^{T} H v\right], \tag{13}
\end{equation*}
$$

where $v \in \mathbb{R}^{d}$ belongs to a Rademacher distribution. Using the finite step approximation similar to Moosavi-Dezfooli et al. (2019); Garg et al. (2023) and the symmetric nature of the Hessian we have

$$
\begin{aligned}
\operatorname{tr}\left(H^{2}\right) & =\frac{1}{n} \sum_{i=0}^{n}\left\|H v_{i}\right\|_{2}^{2} \\
H v & \propto \frac{\partial(L(x+h v)-L(x))}{\partial x}
\end{aligned}
$$

$$
\begin{align*}
\operatorname{tr}\left(H^{2}\right) & \propto \frac{1}{n} \sum_{i=0}^{n}\left\|\frac{\partial(L(x+h v)-L(x))}{\partial x}\right\|_{2}^{2} \\
\operatorname{Curv}(x) & \propto \frac{1}{n} \sum_{i=0}^{n}\left\|\frac{\partial(L(x+h v)-L(x))}{\partial x}\right\|_{2}^{2} \tag{14}
\end{align*}
$$

where $n$ is the number of Rademacher vectors to average. For all our experiments we used $h=1 \times 10^{-3}$ and $n=10$. We found the results to be robust to changes in $h$; we varied it from $1 \times 10^{-1}$ to $1 \times 10^{-3}$. We also varied $n$ from $5,10,20$ and found the results to be robust to changes in $n$.

### 6.3. Input Curvature and Memorization

In this section, we present the empirical results on CIFAR100 and ImageNet datasets for the first link between memorization and input loss curvature (Theorem 5.1).

Experiment. Here we aim to understand how memorization changes with curvature. The experiment aims to plot the memorization score vs curvature score to validate our theoretical results. We calculate curvature scores by averaging over many seeds at the end of training. This empirical measurement is proportional to the expected curvature score, i.e. $\mathbb{E}_{\phi}\left[\operatorname{Curv}_{\phi}\left(z_{i}, S^{\backslash i}\right)\right]$ in Theorem 5.1.

For this experiment, we used 1000 models trained on CIFAR100 and 100 models trained on ImageNet obtained from Feldman \& Zhang (2020)'s 0.7 subset ratio repository. We calculated the curvature score for each sample in the training set using Equation 14. We then compiled a dataset comprising each sample's memorization score and curvature score. Precomputed memorization scores were obtained from Feldman \& Zhang (2020)'s repository. We averaged these scores across all models (1000 for CIFAR100 and 100 for ImageNet) to form an averaged dataset, which was divided into 50 bins based on memorization score. For example, bin 0 includes samples with memorization scores from 0 to 0.02 and the corresponding curvature scores, bin 1 includes samples in the memorization score range of 0.02 to 0.04 , and so on. The average memorization score and maximum curvature score (since curvature is an upper bound) for each bin were used to create a scatter plot as shown in Figures 3(a) and 3(b). For CIFAR100, the Small Inception model was used, and for ImageNet, the ResNet50 model was used, both sourced from Feldman \& Zhang (2020).

Results. We provide the results for CIFAR100 and ImageNet datasets in Figure 3(a) and 3(b) respectively. The figures also visualize the best-fit (shown in red) based on Theorem 5.1. From the results, we see a clear linear relation. The results from the experiment show that the curvature scores have a strong linear trend with respect to memorization, in line with Theorem 5.1.

Accounting for Variables in Practice. Notably, the linearity of the relation between memorization and curvature

![](https://cdn.mathpix.com/cropped/2025_11_08_eddebc1656fc9aee8f2fg-07.jpg?height=592&width=1663&top_left_y=246&top_left_x=197)
Figure 3. Plot of memorization score vs. input loss curvature at the end of training for CIFAR100 (average over 1000 Small Inception models) and ImageNet (average over 100 ResNet50) datasets.

diminishes at the extreme ends of the data range. This phenomenon is particularly pronounced in ImageNet results, as shown in Figure 3(b). This is because the loss bound $L$ (refer to Equation 8) is not constant and the bound changes for each sub-population. Here, we can treat each memorization bin as a sub-population. Hence, when using cross entropy loss we found a better fit, if the loss boundedness is assumed, and the loss bound at convergence is empirically modeled. Accounting for the change in loss bound with sub-population size we see a much improved match. This is observed when comparing the best-fit results in green (assuming sub-population loss bound) vs red (no sub-population loss bound) in Figure 3.

To obtain an improved fit seen in Figure 3 we assumed the loss bound reduces in the square root of the sub-population size (Bousquet \& Elisseeff, 2002). Since the theoretical curvature score from Equation 7 is proportional to the computed curvature score (Equation 14), we can rewrite Equation 8 from Theorm 5.1 using two parameters $p_{1}, c_{1}$ as

$$
|\operatorname{mem}(\mathcal{A}, S, i)| \leq \frac{p_{1}}{L} \cdot \mathbb{E}_{\phi}\left[\operatorname{Curv}_{\phi}\left(z_{i}, S^{\backslash i}\right)\right]+c_{1}
$$

Using $L \propto m_{s u b}^{-0.5}$, where $m_{s u b}$ is the number of samples in each sub-population we can model the relation as

$$
\begin{gather*}
|\operatorname{mem}(\mathcal{A}, S, i)| \leq p_{1} \cdot \sqrt{m_{s u b}} \cdot \mathbb{E}_{\phi}\left[\operatorname{Curv}_{\phi}\left(z_{i}, S^{\backslash i}\right)\right]+c_{1} \\
\text { s.t. } \quad p_{1}, c_{1}>0 \tag{15}
\end{gather*}
$$

Fitting parameters $p_{1}, c_{1}$ to the data results in the green plot in Figures 3(a) and 3(b), where we see much improved match between empirical results and our theory. Thus, these results strongly agree with and validate Theorem 5.1.

### 6.4. Privacy and Input Loss Curvature

In this section, we present the empirical results on CIFAR10 and CIFAR100 datasets to verify the link between privacy and input loss curvature (Theorem 5.3).

Experiment. To study the relation between privacy and curvature, we train private ResNet18 models on CIFAR10 and CIFAR100 using DP-SGD (Abadi et al., 2016) and calculate the curvature scores. We aim to plot privacy budget vs curvature score and validate Theorem 5.3. Specifically, we train models with privacy budgets $\epsilon$ ranging from 5 to 100 , in increments of 5 . We train 10 seeds for every privacy budget, and the curvature score is averaged over the 10 seeds and all the dataset samples.
Accounting for Variables in Practice. For our experiments, we use cross entropy trained private models, where the loss is unbounded. However, Theorem 5.3 assumes bounded loss. Thus, we obtain an empirical bound on the loss at convergence for each privacy budget. We model the loss bound as a function of privacy using $L(\epsilon)=a+b e^{-c \epsilon}$. The fit of this model is shown in Figure 4. Using this loss bound model, Theorem 5.3 can be re-written as

$$
\begin{align*}
\mathbb{E}_{z, \phi} & {\left[\operatorname{Curv}_{\phi}(z, S)\right] \leq L(\epsilon) \cdot(m+1) \cdot\left(1-e^{-\epsilon}\right)+c_{2} } \\
& \leq\left(a+b e^{-c \epsilon}\right) \cdot(m+1) \cdot\left(1-e^{-\epsilon}\right)+c_{2} \tag{16}
\end{align*}
$$

where $c_{2}$ is treated as a constant when trying to fit the data to Equation 16. The empirical data and the best fit model using Equation 16 are shown in Figure 5.
Results. The result of plotting the average convergence loss and privacy budget is shown in Figure 4 along with the best-fit model (in dashed blue line), demonstrating a strong match. Next, Figure 5 shows the result of studying the link between input loss curvature and privacy budget. The scatter plot shows curvature vs privacy. We visualize the empirical data and the best fit (dashed line) using the model from Equation 16. Again, we see a very strong match. All these results strongly correlate with theory and validate Theorem 5.3.

![](https://cdn.mathpix.com/cropped/2025_11_08_eddebc1656fc9aee8f2fg-08.jpg?height=576&width=787&top_left_y=238&top_left_x=203)
Figure 4. Plot of differential privacy vs loss bound for CIFAR100 trained with cross-entropy and the best fit curve (dashed).

![](https://cdn.mathpix.com/cropped/2025_11_08_eddebc1656fc9aee8f2fg-08.jpg?height=573&width=804&top_left_y=951&top_left_x=197)
Figure 5. Plot of privacy vs loss curvature for CIFAR10 and CIFAR100. The best-fit curve (dashed) is predicted by Theorem 5.3.

### 6.5. Memorization and Privacy

In this section, we present the results for the link between memorization and privacy (Theorem 5.4).

Experiment. The goal of the study is to estimate the memorization score of samples when the models have differential privacy guarantees. Since Theorem 5.4 provides an upper bound, we are interested in how privacy affects most memorized examples. This enables us to reduce the compute requirement, and we consider the top 500 most memorized samples from CIFAR100 as reported in Feldman \& Zhang (2020) and study how these scores change as privacy guarantees are varied. For this experiment we first split the CIFAR100 training set into two, set $a$ contains all examples that are not the top 500 most memorized examples, and set $b$ contains the top 500 most memorized examples as reported by Feldman \& Zhang (2020). From $b$ we randomly sample half the dataset called $b^{0.5}$. We concatenate $a$ and $b^{0.5}$ to get our training set. This is used to train a ResNet18 model using DP-SGD (Abadi et al., 2016, Differentially Private

![](https://cdn.mathpix.com/cropped/2025_11_08_eddebc1656fc9aee8f2fg-08.jpg?height=570&width=763&top_left_y=238&top_left_x=1089)
Figure 6. Plot of differential privacy vs memorization for CIFAR100 and the upper bound from the Theorem 5.4.

SGD). We repeat the process of random sub-sampling of $b$ and training for 40 seeds. By keeping track of what samples of $b$ were present in each training run we can estimate the memorization score of the top 500 most memorized examples. This process is repeated 6 times for privacy budgets $\epsilon=1,10,20,30,40,50$ with $\delta=1 \times 10^{-5}$ to train a total of 240 private models (previously described in Section 6.1).
Results. The average memorization scores for the top 500 most memorized examples across various privacy budgets $(\epsilon)$ are presented in Figure 6. As a reference, we also plot the upper bound from Theorem 5.4 in the same plot. Note that Figure 6 is a semi-log plot. The results align with Theorem 5.4, showing an increase in memorization score as the privacy budget increases (i.e. privacy budget $\epsilon \uparrow$ ). Further, the memorization scores are significantly lower than the bound from Theorem 5.4 supporting Nasr et al. (2021)'s observation that DP-SGD may be overly conservative.

## 7. Conclusion

This paper explores the theoretical link between memorization, curvature, and privacy. Understanding this link is critical since input curvature offers $\sim 3$ orders of magnitude compute efficiency when calculating memorization scores. The theoretical analysis relies on three assumptions, stability, generalization, and Lipshitzness, and thus can be applied in non-convex settings such as DNNs. Our main result shows that memorization is upper-bounded by curvature and privacy. Further, we presented two theorems that complete the links between memorization, privacy, and input loss curvature. To empirically test the theory we use standard DNNs for image classification using CIFAR100 and ImageNet datasets. Our results show a very strong match between our theoretical findings and empirical results. Results in this paper provide evidence for the link between memorization, input loss curvature, and privacy strengthening the understanding of DNNs and their properties.

## Impact Statement

The research presented in this paper fills significant gaps in our understanding of DNNs. We focused on the relationship between memorization, input loss curvature, and privacy. This finding is key for various applications, as it provides a clearer framework for leveraging the significant ( $\sim 3$ orders of magnitude) efficiencies in computing memorization scores when using input loss curvature. This work, therefore, not only develops our theoretical understanding of DNNs but also offers practical insights for developing more effective machine learning and deep learning models and algorithms.

## References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pp. 308-318, 2016.

Andriushchenko, M. and Flammarion, N. Towards understanding sharpness-aware minimization. In International Conference on Machine Learning, pp. 639-668. PMLR, 2022.

Arpit, D., Jastrzebski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., Maharaj, T., Fischer, A., Courville, A., Bengio, Y., et al. A closer look at memorization in deep networks. In International conference on machine learning, pp. 233-242. PMLR, 2017.

Bae, J., Ng, N., Lo, A., Ghassemi, M., and Grosse, R. B. If influence functions are the answer, then what is the question? Advances in Neural Information Processing Systems, 35:17953-17967, 2022.

Basu, S., Pope, P., and Feizi, S. Influence functions in deep learning are fragile. In International Conference on Learning Representations, 2021. URL https: / / openreview.net/forum?id=xHKVVHGDOEk.

Bousquet, O. and Elisseeff, A. Stability and generalization. The Journal of Machine Learning Research, 2:499-526, 2002.

Brown, G., Bun, M., Feldman, V., Smith, A., and Talwar, K. When is memorization of irrelevant training data necessary for high-accuracy learning? In Proceedings of the 53rd annual ACM SIGACT symposium on theory of computing, pp. 123-132, 2021.

Carlini, N., Erlingsson, U., and Papernot, N. Distribution density, tails, and outliers in machine learning: Metrics and applications. arXiv preprint arXiv:1910.13427, 2019.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy (SP), pp. 1897-1914. IEEE, 2022.

Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3, pp. 265-284. Springer, 2006.

Fawzi, A., Moosavi-Dezfooli, S.-M., Frossard, P., and Soatto, S. Empirical study of the topology and geometry of deep networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3762-3770, 2018.

Feldman, V. Does learning require memorization? a short tale about a long tail. arXiv preprint arXiv:1906.05271, 2019.

Feldman, V. and Vondrak, J. High probability generalization bounds for uniformly stable algorithms with nearly optimal rate. In Conference on Learning Theory, pp. 1270-1279. PMLR, 2019.

Feldman, V. and Zhang, C. What neural networks memorize and why: Discovering the long tail via influence estimation. Advances in Neural Information Processing Systems, 33:2881-2891, 2020.

Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations, 2021. URL https: / /openreview. net/forum?id=6Tm1mposlrM.

Garg, I. and Roy, K. Samples with low loss curvature improve data efficiency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20290-20300, 2023.

Garg, I., Ravikumar, D., and Roy, K. Memorization through the lens of curvature of loss function around samples. arXiv preprint arXiv:2307.05831, 2023.

Hardt, M., Recht, B., and Singer, Y. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning, pp. 1225-1234. PMLR, 2016.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.

Hutchinson, M. F. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 18(3):1059-1076, 1989.

Jiang*, Y., Neyshabur*, B., Mobahi, H., Krishnan, D., and Bengio, S. Fantastic generalization measures and where to find them. In International Conference on Learning Representations, 2020. URL https: / /openreview. net/forum?id=SJgIPJBFvH.

Jiang, Z., Zhang, C., Talwar, K., and Mozer, M. C. Characterizing structural regularities of labeled data in overparameterized models. arXiv preprint arXiv:2002.03206, 2020.

Kearns, M. and Ron, D. Algorithmic stability and sanitycheck bounds for leave-one-out cross-validation. In Proceedings of the tenth annual conference on Computational learning theory, pp. 152-162, 1997.

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P. On large-batch training for deep learning: Generalization gap and sharp minima. In International Conference on Learning Representations, 2017. URL https: / /openreview.net/forum? id=H1oyRlYgg.

Koh, P. W. and Liang, P. Understanding black-box predictions via influence functions. In International conference on machine learning, pp. 1885-1894. PMLR, 2017.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images, 2009.

Kwon, J., Kim, J., Park, H., and Choi, I. K. Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. In International Conference on Machine Learning, pp. 5905-5914. PMLR, 2021.

Liu, S., Niles-Weed, J., Razavian, N., and FernandezGranda, C. Early-learning regularization prevents memorization of noisy labels. Advances in neural information processing systems, 33:20331-20342, 2020.

Lukasik, M., Nagarajan, V., Rawat, A. S., Menon, A. K., and Kumar, S. What do larger image classifiers memorise? arXiv preprint arXiv:2310.05337, 2023.

Maini, P., Garg, S., Lipton, Z., and Kolter, J. Z. Characterizing datapoints via second-split forgetting. Advances in Neural Information Processing Systems, 35:3004430057, 2022.

Moosavi-Dezfooli, S.-M., Fawzi, A., Uesato, J., and Frossard, P. Robustness via curvature regularization, and vice versa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 90789086, 2019.

Nasr, M., Songi, S., Thakurta, A., Papernot, N., and Carlin, N. Adversary instantiation: Lower bounds for differentially private machine learning. In 2021 IEEE Symposium on security and privacy (SP), pp. 866-882. IEEE, 2021.

Nesterov, Y. and Polyak, B. T. Cubic regularization of newton method and its global performance. Mathematical Programming, 108(1):177-205, 2006.

Pleiss, G., Zhang, T., Elenberg, E., and Weinberger, K. Q. Identifying mislabeled data using the area under the margin ranking. Advances in Neural Information Processing Systems, 33:17044-17056, 2020.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211-252, 2015. doi: 10.1007/s11263-015-0816-y.

Schioppa, A., Zablotskaia, P., Vilar, D., and Sokolov, A. Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8179-8186, 2022.

Schioppa, A., Filippova, K., Titov, I., and Zablotskaia, P. Theoretical and practical perspectives on what influence functions do. arXiv preprint arXiv:2305.16971, 2023.

Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pp. 3-18. IEEE, 2017.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9, 2015.

Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and Gordon, G. J. An empirical study of example forgetting during deep neural network learning. In International Conference on Learning Representations, 2019. URL https: / /openreview.net/forum? id=BJlxm30cKm.

Wang, Y.-X., Lei, J., and Fienberg, S. E. Learning with differential privacy: Stability, learnability and the sufficiency and necessity of erm principle. The Journal of Machine Learning Research, 17(1):6353-6392, 2016.

Wu, D., Xia, S.-T., and Wang, Y. Adversarial weight perturbation helps robust generalization. Advances in Neural Information Processing Systems, 33:2958-2969, 2020.

Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., and Mironov, I. Opacus: Userfriendly differential privacy library in PyTorch. arXiv preprint arXiv:2109.12298, 2021.

Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations, 2017. URL https: / /openreview. net/forum?id=Sy8gdB9xx.

Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107115, 2021.

## A. Proofs

## A.1. Proof of Theorem 5.4

Consider $S, S^{\backslash i}$ from construction we have $\left\|S-S^{\backslash i}\right\|=1$. 
Next let $\mathcal{R} \subset \operatorname{Range}(\mathcal{A})$ such that $\mathcal{R}=\left\{h \mid h\left(x_{i}\right)=y_{i}\right\}$. 
Since $\mathcal{A}$ is $\epsilon$-differentially private then it follows from the definition of differential privacy in Equation 1 that

$$
\begin{equation*}
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi} \in \mathcal{R}\right] \leq e^{\epsilon} \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi} \in \mathcal{R}\right] \tag{17}
\end{equation*}
$$

Since $\mathcal{R}=\left\{h \mid h\left(x_{i}\right)=y_{i}\right\}$ we have

$$
\begin{equation*}
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi} \in \mathcal{R}\right]=\operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right] \tag{18}
\end{equation*}
$$

Using Equations 17 and 18 we have

$$
\begin{align*}
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right] & \leq e^{\epsilon} \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right]  \tag{19}\\
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right] & \leq e^{\epsilon} \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] \pm \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] \\
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]-\operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] & \leq e^{\epsilon} \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right]-\operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] \\
\operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]-\operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] & \leq\left(e^{\epsilon}-1\right) \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] \\
\operatorname{mem}(\mathcal{A}, S, i) & \leq\left(e^{\epsilon}-1\right) \operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right]
\end{align*}
$$

Using Equation 19, we have the lower bound on $\operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right]$ as

$$
\operatorname{Pr}_{\phi}\left[h_{S \backslash i}^{\phi}\left(x_{i}\right)=y_{i}\right] \geq e^{-\epsilon} \operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]
$$

Thus we have

$$
\begin{aligned}
& \operatorname{mem}(\mathcal{A}, S, i) \leq\left(e^{\epsilon}-1\right) e^{-\epsilon} \operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right] \\
& \operatorname{mem}(\mathcal{A}, S, i) \leq\left(1-e^{-\epsilon}\right) \operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]
\end{aligned}
$$

Since $\sup \operatorname{Pr}_{\phi}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]=1$ we have the result

$$
\operatorname{mem}(\mathcal{A}, S, i) \leq 1-e^{-\epsilon}
$$

## A.2. Proof of Lemma A. 2

Lemma A.1. If the generalization assumption 4 holds then we know here exists $\gamma$ such that with probability $1-\delta$

$$
\begin{align*}
\mathbb{E}_{\phi}\left[\left|R_{\text {emp }}\left(h_{S \backslash i}^{\phi}, S\right)-R\left(h_{S \backslash i}^{\phi}\right)\right|\right] & \leq \gamma  \tag{20}\\
\mathbb{E}_{\phi}\left[\left|R_{\text {emp }}\left(h_{S}^{\phi}, S\right)-R\left(h_{S}^{\phi}\right)\right|\right] & \leq \gamma  \tag{21}\\
\mathbb{E}_{\phi}\left[\left|R_{\text {emp }}\left(h_{S}^{\phi}, S^{\backslash i}\right)-R\left(h_{S}^{\phi}\right)\right|\right] & \leq \gamma \tag{22}
\end{align*}
$$

Proof of Lemma A. 1 From (Feldman \& Vondrak, 2019) we know that with a confidence $\delta$ we have

$$
\operatorname{Pr}_{S \sim \mathcal{D}^{m}}\left[\left|R_{e m p}(h, S)-R(h)\right| \geq c\left(\beta^{\prime} \ln (m) \ln (m / \delta)+\frac{\sqrt{\ln (1 / \delta)}}{\sqrt{m}}\right)\right] \leq \delta
$$

Where $\beta^{\prime}$ is the uniform stability bound. Thus with a confidence of at least $1-\delta$ we can say:

$$
\left|R_{e m p}(h, S)-R(h)\right|<c\left(\beta^{\prime} \ln (m) \ln (m / \delta)+\frac{\sqrt{\ln (1 / \delta)}}{\sqrt{m}}\right)
$$

Thus if we set

$$
\gamma^{\prime}(m)=c\left(\beta^{\prime} \ln (m) \ln (m / \delta)+\frac{\sqrt{\ln (1 / \delta)}}{\sqrt{m}}\right)
$$

we have

$$
\begin{equation*}
\left|R_{e m p}(h, S)-R(h)\right|<\gamma^{\prime}(m) \tag{23}
\end{equation*}
$$

Thus as a direct consequence of Equation 23 we can say

$$
\begin{align*}
\forall S, S^{\backslash i} \sim \mathcal{D}^{m}, \mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-R\left(h_{S \backslash i}^{\phi}\right)\right|\right] \leq \gamma^{\prime}(m-1) & \forall S \sim \mathcal{D}^{m}, \mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S}^{\phi}, S\right)-R\left(h_{S}^{\phi}\right)\right|\right] \leq \gamma^{\prime}(m)  \tag{24}\\
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S \backslash i}^{\phi}, S\right)-R\left(h_{S \backslash i}^{\phi}\right)\right|\right] & =\mathbb{E}_{\phi}\left[\left|\frac{1}{m} \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right|\right]+\mathbb{E}_{\phi}\left[\left|\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-R\left(h_{S \backslash i}^{\phi}\right)\right|\right]  \tag{25}\\
& =\frac{1}{m} \mathbb{E}_{\phi}\left[\left|\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right|\right]+\mathbb{E}_{\phi}\left[\left|\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-\frac{m-1}{m} R\left(h_{S \backslash i}^{\phi}\right)-\frac{1}{m} R\left(h_{S \backslash i}^{\phi}\right)\right|\right] \\
& \leq \frac{L}{m}+\mathbb{E}_{\phi}\left[\left|\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-\frac{m-1}{m} R\left(h_{S \backslash i}^{\phi}\right)-\frac{1}{m} R\left(h_{S \backslash i}^{\phi}\right)\right|\right] \\
& \leq \frac{L}{m}+\mathbb{E}_{\phi}\left[\left|\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-\frac{m-1}{m} R\left(h_{S \backslash i}^{\phi}\right)-\frac{1}{m} R\left(h_{S \backslash i}^{\phi}\right) \pm \frac{1}{m} R\left(h^{*}\right)\right|\right] \\
& \leq \frac{L}{m}+\mathbb{E}_{\phi}\left[\left|\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-\frac{m-1}{m} R\left(h_{S \backslash i}^{\phi}\right)-\frac{1}{m} R\left(h^{*}\right)\right|\right]+\Delta \\
& \leq \frac{L}{m}+\mathbb{E}_{\phi}\left[\left|\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-\frac{m-1}{m} R\left(h_{S \backslash i}^{\phi}\right)-\frac{1}{m} R\left(h^{*}\right)\right|\right]+\Delta \\
& \leq \frac{L}{m}+\left|\frac{m-1}{m} \gamma^{\prime}(m-1)\right|+\left|\frac{1}{m} R\left(h^{*}\right)\right|+\Delta \\
& \leq \frac{L}{m}+\left|\frac{m-1}{m} \gamma^{\prime}(m-1)\right|+\frac{L}{m}+\Delta \\
& \leq \frac{2 L}{m}+\frac{m-1}{m} \gamma^{\prime}(m-1)+\Delta
\end{align*}
$$

Now consider

$$
\begin{aligned}
R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)-R\left(h_{S}^{\phi}\right) & =R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)-R\left(h_{S}^{\phi}\right) \\
& =\frac{1}{m-1} \sum_{j=1, j \neq i}^{m} \ell\left(h_{S}^{\phi}, z_{j}\right)-R\left(h_{S}^{\phi}\right) \\
& =\frac{1}{m-1} \sum_{j=1}^{m} \ell\left(h_{S}^{\phi}, z_{j}\right)-\frac{1}{m-1} \ell\left(h_{S}^{\phi}, z_{i}\right)-R\left(h_{S}^{\phi}\right) \\
& \leq \frac{m}{m-1} R_{e m p}\left(h_{S}^{\phi}, S\right)-R\left(h_{S}^{\phi}\right) \\
& \leq \gamma^{\prime}(m)+\frac{1}{m-1} R_{e m p}\left(h_{S}^{\phi}, S\right) \\
& \leq \gamma^{\prime}(m)+\frac{1}{m-1} L \\
\left|R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)-R\left(h_{S}^{\phi}\right)\right| & \leq \gamma^{\prime}(m)+\frac{L}{m-1}
\end{aligned}
$$

Thus if we set $\gamma(m)=\max \left\{\frac{2 L}{m}+\frac{m-1}{m} \gamma^{\prime}(m-1)+\Delta, \gamma^{\prime}(m)+\frac{L}{m-1}\right\}$ we get $\forall S, S^{\backslash i} \sim \mathcal{D}^{m}$

$$
\begin{align*}
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S \backslash i}^{\phi}, S\right)-R\left(h_{S \backslash i}^{\phi}\right)\right|\right] & \leq \gamma  \tag{26}\\
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S}^{\phi}, S\right)-R\left(h_{S}^{\phi}\right)\right|\right] & \leq \gamma  \tag{27}\\
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)-R\left(h_{S}^{\phi}\right)\right|\right] & \leq \gamma \tag{28}
\end{align*}
$$

Using error stability from assumption (see Equation 3) introduced by Kearns \& Ron (1997) without loss of generality, we can write

$$
\begin{array}{r}
\mathbb{E}_{\phi, z \sim \mathcal{D}}\left[\ell\left(h_{S}^{\phi}, z\right)\right]-\mathbb{E}_{\phi, z \sim \mathcal{D}}\left[\ell\left(h_{S \backslash i}^{\phi}, z\right)\right] \leq \beta \\
\mathbb{E}_{\phi}\left[R\left(h_{S}^{\phi}\right)-R\left(h_{S \backslash i}^{\phi}\right)\right] \leq \beta
\end{array}
$$

Lemma A.2. If assumptions of error stability 3, generalization 4, and uniform model bias 5 hold, then for all $i, j$ and two adjacent datasets $S, S^{\backslash i} \sim \mathcal{D}$ with a probability at least $1-\delta$ it holds that

$$
\begin{align*}
\left|\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right]\right| \leq & m \beta+(4 m-1) \gamma \\
& +2(m-1) \Delta . \tag{29}
\end{align*}
$$

Lemma A. 2 provides an upper bound on the expected loss difference between two adjacent datasets evaluated at any data point in the training set.

Proof of Lemma A. 2 Using Lemma A. 1 we know here exists $\gamma$ such that with probability $1-\delta$

$$
\begin{aligned}
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S \backslash i}^{\phi}, S\right)-R\left(h_{S \backslash i}^{\phi}\right)\right|\right] & \leq \gamma \\
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S}^{\phi}, S\right)-R\left(h_{S}^{\phi}\right)\right|\right] & \leq \gamma \\
\mathbb{E}_{\phi}\left[\left|R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)-R\left(h_{S}^{\phi}\right)\right|\right] & \leq \gamma
\end{aligned}
$$

Using Equations 20 and 21 we can upper bound the expected difference in empirical risk of adjacent datasets as

$$
\begin{aligned}
\mathbb{E}_{\phi}\left[R_{e m p}\left(h_{S}^{\phi}, S\right)-R_{e m p}\left(h_{S \backslash i}^{\phi}, S\right)\right] & \leq \beta+2 \gamma \\
\mathbb{E}_{\phi}\left[\frac{1}{m} \ell\left(h_{S}^{\phi}, z_{i}\right)+\frac{m-1}{m} R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)-\frac{1}{m} \ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)-\frac{m-1}{m} R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)\right] & \leq \beta+2 \gamma \\
\frac{1}{m} \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\frac{1}{m} \mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] & \leq \beta+2 \gamma+\frac{m-1}{m} \mathbb{E}_{\phi}\left[R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)\right] \\
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] & \leq m \beta+2 m \gamma+(m-1) \mathbb{E}_{\phi}\left[R_{e m p}\left(h_{S \backslash i}^{\phi}, S^{\backslash i}\right)-R_{e m p}\left(h_{S}^{\phi}, S^{\backslash i}\right)\right]
\end{aligned}
$$

We obtain the upper and lower bound for empirical risk using Equations 20 and 22 to get

$$
\begin{aligned}
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] & \leq m \beta+2 m \gamma+(m-1) \mathbb{E}_{\phi}\left[R\left(h_{S \backslash i}^{\phi}\right)+\gamma-R\left(h_{S}^{\phi}\right)+\gamma\right] \\
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] & \leq m \beta+(4 m-1) \gamma+(m-1) \mathbb{E}_{\phi}\left[R\left(h_{S \backslash i}^{\phi}\right)-R\left(h_{S}^{\phi}\right)\right]
\end{aligned}
$$

We add an subtract the risk of $h^{*}=\mathbb{E}[y \mid x]$ which is the true conditional of $\mathcal{D}^{m}$

$$
\begin{aligned}
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+(m-1) \mathbb{E}_{\phi}\left[R\left(h_{S \backslash i}^{\phi}\right)-R\left(h_{S}^{\phi}\right) \pm R\left(h^{*}\right)\right] \\
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+(m-1) \mathbb{E}_{\phi}\left[R\left(h_{S \backslash i}^{\phi}\right)-R\left(h^{*}\right)-\left(R\left(h_{S}^{\phi}\right)-R\left(h^{*}\right)\right)\right]
\end{aligned}
$$

Using the uniform model bias bound from assumption 5 we have

$$
\begin{aligned}
& \mathbb{E}_{\phi}\left[R\left(h_{S \backslash i}^{\phi}\right)-R\left(h^{*}\right)\right] \leq \Delta \\
& \mathbb{E}_{\phi}\left[R\left(h_{S}^{\phi}\right)-R\left(h^{*}\right)\right] \geq-\Delta
\end{aligned}
$$

Hence we get

$$
\begin{aligned}
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+(m-1)[\Delta-(-\Delta)] \\
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta
\end{aligned}
$$

Since we can interchange $\ell\left(h_{S}^{\phi}, z_{i}\right)$ and $\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)$ i.e. start with $\mathbb{E}_{z \sim \mathcal{D}}\left[\ell\left(h_{S \backslash i}^{\phi}, z\right)\right]-\mathbb{E}_{z \sim \mathcal{D}}\left[\ell\left(h_{S}^{\phi}, z\right)\right]$ we have the result

$$
\left|\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right]\right| \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta
$$

Lemma A.3. If Lipschitz assumption 6 on the Hessian of $\ell$ holds from Nesterov \& Polyak (2006) we have

$$
\begin{equation*}
\left|\ell\left(h, z_{1}\right)-\ell\left(h, z_{2}\right)-\left\langle\nabla \ell\left(h, z_{2}\right), z_{1}-z_{2}\right\rangle-\left\langle\nabla^{2} \ell\left(h, z_{2}\right)\left(z_{1}-z_{2}\right), z_{1}-z_{2}\right\rangle\right| \leq \frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3} \tag{30}
\end{equation*}
$$

## A.3. Proof of Theorem 5.1

From Lemma A. 3 we have

$$
-\frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3} \leq \ell\left(h, z_{1}\right)-\ell\left(h, z_{2}\right)-\left\langle\nabla \ell\left(h, z_{2}\right), z_{1}-z_{2}\right\rangle-\left\langle\nabla^{2} \ell\left(h, z_{2}\right)\left(z_{1}-z_{2}\right), z_{1}-z_{2}\right\rangle \leq \frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3}
$$

This gives us an upper bound on $\ell\left(h, z_{1}\right)$

$$
\begin{equation*}
\ell\left(h, z_{1}\right) \leq \frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3}+\ell\left(h, z_{2}\right)+\left\langle\nabla \ell\left(h, z_{2}\right), z_{1}-z_{2}\right\rangle+\left\langle\nabla^{2} \ell\left(h, z_{2}\right)\left(z_{1}-z_{2}\right), z_{1}-z_{2}\right\rangle \tag{31}
\end{equation*}
$$

Consider $z_{j} \in S$ such that $z_{j}=z_{i}+\alpha$ for some $j \neq i$ where $\alpha \in B_{p}(v)$ such that $\mathbb{E}[\alpha]=0$ and $\mathbb{E}\left[\alpha^{T} \alpha\right]=1$.
Without loss of generality from Lemma A. 2 we have:

$$
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta
$$

Using the upper bound from Equation 31, setting $z_{1}=z_{j}, z_{2}=z_{i}$ we have

$$
\begin{array}{r}
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)-\frac{\rho}{6}\|\alpha\|^{3}-\left\langle\nabla \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right), \alpha\right\rangle-\left\langle\nabla^{2} \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right) \alpha, \alpha\right\rangle\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)-\frac{\rho}{6}\|\alpha\|^{3}-\left\langle\nabla \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right), \alpha\right\rangle-\alpha^{T} H^{T} \alpha\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta
\end{array}
$$

Where $H=\nabla^{2} \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)$. Next, we take expectation over $\alpha$ we get

$$
\begin{aligned}
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right]-\frac{\rho}{6} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]-\mathbb{E}_{\phi, \alpha}\left[\alpha^{T} H^{T} \alpha\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right]-\frac{\rho}{6} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]-\mathbb{E}_{\phi, \alpha}\left[\operatorname{tr}\left(H^{T} \mathbb{E}_{\alpha}\left[\alpha^{T} \alpha\right]\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right] \leq \frac{\rho}{6} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]+\mathbb{E}_{\phi}\left[\operatorname{tr}\left(H^{T} \mathbb{E}_{\alpha}\left[\alpha^{T} \alpha\right]\right)\right]+m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right] \leq \frac{\rho}{6} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]+\mathbb{E}_{\phi}[\operatorname{tr}(H)]+m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
& \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right] \leq \frac{\rho}{6} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]+\mathbb{E}_{\phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right)\right]+m \beta+(4 m-1) \gamma+2(m-1) \Delta
\end{aligned}
$$

If we have $0 \leq \ell \leq L$ then:

$$
\begin{equation*}
\frac{1}{L}\left[\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right]\right]=\operatorname{mem}(\mathcal{A}, S, i) \tag{32}
\end{equation*}
$$

Since we can exchange $S$ and $S^{\backslash i}$ Hence we have the result

$$
|\operatorname{mem}(\mathcal{A}, S, i)| \leq \frac{\rho}{6 L} \mathbb{E}_{\alpha}\left[\|\alpha\|^{3}\right]+\frac{1}{L} \mathbb{E}_{\phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S \backslash i}^{\phi}, z_{i}\right)\right)\right]+\frac{m \beta}{L}+\frac{(4 m-1) \gamma}{L}+\frac{2(m-1) \Delta}{L}
$$ $\square$

## A.4. Proof of Theorem 5.1 for Cross-Entropy

For classification with one-hot ground truth labels we have cross entropy we have.

$$
\begin{aligned}
& \ell\left(h_{S}^{\phi}, z_{i}\right)=-\ln \left(\operatorname{Pr}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]\right) \\
& \ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)=\ln \left(\frac{\operatorname{Pr}\left[h_{S \backslash i}^{\phi}\left(x_{j}\right)=y_{j}\right]}{\operatorname{Pr}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]}\right) \\
&=\ln \left(\frac{a}{b}\right) \\
& \text { for } \quad \theta>-1 \quad \text { we have, } \quad \frac{\theta}{\theta+1} \leq \ln (1+\theta) \\
& \frac{\frac{a}{b}-1}{\frac{a}{b}} \leq \ln \left(\frac{a}{b}\right) \\
& \frac{a-b}{a} \leq \ln \left(\frac{a}{b}\right) \\
& a-b \leq \frac{a-b}{a} \leq \ln \left(\frac{a}{b}\right) \quad \text { For } 0<a \leq 1
\end{aligned}
$$

Thus we have

$$
\operatorname{Pr}\left[h_{S \backslash i}^{\phi}\left(x_{j}\right)=y_{j}\right]-\operatorname{Pr}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right] \leq \ell\left(h_{S}^{\phi}, z_{i}\right)-\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)
$$

Taking expectation over the randomness of $\mathcal{A}$ we have

$$
\begin{aligned}
\mathbb{E}_{\phi}\left[\operatorname{Pr}\left[h_{S \backslash i}^{\phi}\left(x_{j}\right)=y_{j}\right]\right]-\mathbb{E}_{\phi}\left[\operatorname{Pr}\left[h_{S}^{\phi}\left(x_{i}\right)=y_{i}\right]\right] & \leq \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \\
\operatorname{mem}(\mathcal{A}, S, i) & \leq \mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right]
\end{aligned}
$$

## A.5. Proof of Lemma 5.2

Let $h \sim \mathcal{A}(\phi, S)$ have a pdf defined as $p(h)$, and $h^{\prime} \sim \mathcal{A}\left(\phi, S^{\backslash i}\right)$ have a pdf defined as $p^{\prime}\left(h^{\prime}\right)$

$$
\begin{aligned}
& \mid \mathbb{E}_{\phi, z}\left[\ell\left(h_{S}^{\phi}, z\right)\right]-\mathbb{E}_{\phi, z}\left[\ell\left(h_{S \backslash i}^{\phi}, z\right)\right]=\left|\mathbb{E}_{z, \phi}\left[\ell\left(h_{S}^{\phi}, z\right)\right]-\mathbb{E}_{z, \phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z\right)\right]\right| \\
&=\left|\mathbb{E}_{z, \phi}[\ell(\mathcal{A}(\phi, S), z)]-\mathbb{E}_{z, \phi}\left[\ell\left(\mathcal{A}\left(\phi, S^{\backslash i}\right), z\right)\right]\right| \\
&=\left|\mathbb{E}_{z, h}[\ell(h, z)]-\mathbb{E}_{z, h^{\prime}}\left[\ell\left(h^{\prime}, z\right)\right]\right| \\
&=\left|\int_{z} \int_{h} \ell(h, z) p(h) d h p(z) d z-\int_{z} \int_{h^{\prime}} \ell\left(h^{\prime}, z\right) p^{\prime}\left(h^{\prime}\right) d h^{\prime} p(z) d z\right| \\
&=\left|\int_{z} \int_{h} \ell(h, z) p(h) d h p(z) d z-\int_{z} \int_{h} \ell(h, z) p^{\prime}(h) d h p(z) d z\right| \\
&=\left|\int_{z} \int_{h} \ell(h, z)\left(p(h)-p^{\prime}(h)\right) d h p(z) d z\right| \\
& \leq\left|\int_{z} \sup \ell(h, z) \int_{h: p(h) \geq p^{\prime}(h)}\left(p(h)-p^{\prime}(h)\right) d h p(z) d z\right| \\
& \leq\left|\sup _{h, z} \ell(h, z) \int_{z} p(z) d z \int_{h: p(h) \geq p^{\prime}(h)}\left(p(h)-p^{\prime}(h)\right) d h\right| \\
& \leq\left|L \int_{h: p(h) \geq p^{\prime}(h)} p(h)-p^{\prime}(h) d h\right| \\
& \leq\left|L \int_{h: p(h) \geq p^{\prime}(h)} p(h)\left(1-\frac{p^{\prime}(h)}{p(h)}\right) d h\right| \\
& \leq\left|L \int_{h: p(h) \geq p^{\prime}(h)} p(h)\left(1-e^{-\epsilon}\right) d h\right| \\
& \leq\left|L\left(1-e^{-\epsilon}\right) \int_{h: p(h) \geq p^{\prime}(h)} p(h) d h\right| \\
& \leq\left|L\left(1-e^{-\epsilon}\right)\right| \\
& \leq L\left(1-e^{-\epsilon}\right) \\
& \text { ■ }
\end{aligned}
$$

Lemma A.4. If the assumptions of error stability 3, generalization 4, and uniform model bias 5 hold, then for two adjacent datasets $S, S^{\backslash i} \sim \mathcal{D}$ and for any $i, j \in\{1, \cdots, m\}$ with a probability at least $1-\delta$ we have

$$
\begin{align*}
\mathbb{E}_{\phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z_{i}\right)\right)\right] \leq & m \beta+(4 m-1) \gamma \\
& +2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right] \\
& +\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] . \tag{33}
\end{align*}
$$

## A.6. Proof of Lemma A. 4

From Lemma A. 3 we have

$$
-\frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3} \leq \ell\left(h, z_{1}\right)-\ell\left(h, z_{2}\right)-\left\langle\nabla \ell\left(h, z_{2}\right), z_{1}-z_{2}\right\rangle-\left\langle\nabla^{2} \ell\left(h, z_{2}\right)\left(z_{1}-z_{2}\right), z_{1}-z_{2}\right\rangle \leq \frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3}
$$

This gives us a lower bound on $\ell\left(h, z_{1}\right)$

$$
\begin{equation*}
-\frac{\rho}{6}\left|z_{1}-z_{2}\right|^{3}+\ell\left(h, z_{2}\right)+\left\langle\nabla \ell\left(h, z_{2}\right), z_{1}-z_{2}\right\rangle+\left\langle\nabla^{2} \ell\left(h, z_{2}\right)\left(z_{1}-z_{2}\right), z_{1}-z_{2}\right\rangle \leq \ell\left(h, z_{1}\right) \tag{34}
\end{equation*}
$$

Consider $z_{j} \in S$ such that $z_{i}=z_{j}+\alpha$ for some $j \neq i$ where $\alpha \in B_{p}(v)$ such that $\mathbb{E}[\alpha]=0$ and $\mathbb{E}\left[\alpha^{T} \alpha\right]=1$. Using the lower bound in Lemma A. 2 with $z_{1}=z_{i}, z_{2}=z_{j}$ we get

$$
\begin{array}{r}
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{i}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
-\frac{\rho}{6}\|\alpha\|^{3}+\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]+\mathbb{E}_{\phi}\left[\left\langle\nabla \ell\left(h_{S}^{\phi}, z_{j}\right), \alpha\right\rangle\right]+\mathbb{E}_{\phi}\left[\left\langle\nabla^{2} \ell\left(h_{S}^{\phi}, z_{i}\right) \alpha, \alpha\right\rangle\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta \\
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]+\mathbb{E}_{\phi}\left[\left\langle\nabla \ell\left(h_{S}^{\phi}, z_{j}\right), \alpha\right\rangle\right]+\mathbb{E}_{\phi}\left[\left\langle\nabla^{2} \ell\left(h_{S}^{\phi}, z_{i}\right) \alpha, \alpha\right\rangle\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6}\|\alpha\|^{3}
\end{array}
$$

Taking Expectation over $\alpha$ we get
$\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]+\mathbb{E}_{\alpha, \phi}\left[\left\langle\nabla \ell\left(h_{S}^{\phi}, z_{j}\right), \alpha\right\rangle\right]+\mathbb{E}_{\alpha, \phi}\left[\left\langle\nabla^{2} \ell\left(h_{S}^{\phi}, z_{i}\right) \alpha, \alpha\right\rangle\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6}\|\alpha\|^{3}$
Note that we can change the order of expectation due to Fubini's theorem

$$
\begin{array}{r}
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]+\mathbb{E}_{\phi, \alpha}\left[\left\langle\nabla^{2} \ell\left(h_{S}^{\phi}, z_{i}\right) \alpha, \alpha\right\rangle\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6}\|\alpha\|^{3} \\
\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]+\mathbb{E}_{\phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z_{i}\right)\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6}\|\alpha\|^{3} \\
\mathbb{E}_{\phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z_{j}\right)\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right]+\mathbb{E}_{\phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]-\mathbb{E}_{\phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right]
\end{array}
$$

## A.7. Proof of Theorem 5.3

We start with the results of Lemma A.4. Taking Expectation over $z \sim \mathcal{D}$ we have

$$
\begin{aligned}
& \mathbb{E}_{z, \phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z\right)\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right]+\mathbb{E}_{z, \phi}\left[\ell\left(h_{S}^{\phi}, z_{j}\right)\right]-\mathbb{E}_{z, \phi}\left[\ell\left(h_{S \backslash i}^{\phi}, z_{j}\right)\right] \\
& \mathbb{E}_{z, \phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z\right)\right)\right] \leq m \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right]+\beta \\
& \mathbb{E}_{z, \phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z\right)\right)\right] \leq(m+1) \beta+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right]
\end{aligned}
$$

Using Lemma 5.2

$$
\mathbb{E}_{z, \phi}\left[\operatorname{tr}\left(\nabla^{2} \ell\left(h_{S}^{\phi}, z\right)\right)\right] \leq L(m+1)\left(1-e^{-\epsilon}\right)+(4 m-1) \gamma+2(m-1) \Delta+\frac{\rho}{6} \mathbb{E}\left[\|\alpha\|^{3}\right]
$$


[^0]:    ${ }^{1}$ Department of ECE, Purdue University, West Lafayette, Indiana. Correspondence to: Deepak Ravikumar [dravikum@purdue.edu](mailto:dravikum@purdue.edu).

