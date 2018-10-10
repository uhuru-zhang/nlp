#Word2Vect
##How to represent words?
* navie representation: one-hot vectors in $R^{\mid vocabulary \mid}$(very large).
* Classical IR: document and query vector are superpositions of word vectors: $$\hat{d_q}=\arg\max_d sim(d,q)$$ simily for word classification problems
* We want richer representations expressing senmantic similarity
* Idea: produce dense vector representation based on the context/use of words.
* Three main approches: **count-based**, **predictive**, and **task-based**.

##Count-based methods
* Define a **biasis vocabulary** $C$ of content words.
* Define a **word window** size $W$.
* **Count the basis vocabulary** occurring w words to the left or right of each instance of a **target word** in the corpus. From a **vector representation** of the target word based on these counts.  

		... and the cute kitten purred and then ...  
		... the cute furry cat purred and miaowed ...   
		... that the small kitten miaowed and she ...   
		... the loud furry dog ran and bit ...  

	Example **basis vocabulary**: {bit, cute, furry, loud, miaowed, purred, ran, small}.

	kitten context words: {cute, purred, small, miaowed}.   
	cat context words: {cute, furry, miaowed}.  
	dog context words: {loud, furry, ran, bit}.  
	$$kitten=[0,1,0,0,1,1,0,1]^T$$
	$$cat=[0,1,1,0,1,0,0,0]^T$$
	$$dog=[1,0,1,1,0,0,1,0]^T$$
	We can use inner product or cosine as **similarity kernel**.E.g.:$$sim(kitten, cat)=cosine(\vec{kitten}, \vec{cat}) \approx 0.58$$ $$consine(\vec{u},\vec{v}) = \frac{\vec{u}\cdot\vec{v}}{\parallel \vec{u} \parallel \times \parallel \vec{v} \parallel}$$
* **Not all features are equel**: we must distinguish counts that are high because they are informative from those that are just ***independently frequent contexts***
* Some **normalisation methods**: **TF-IDF**, **PMI**

##Nerual Embedding Models
* **count based vectors** 产生了一个 **embedding matrix in $R^{|vocab|\times|context|}$**$$
\begin{array}{c|lcr}
n & \text{bit} & \text{cute} & \text{furry} & \cdots\\
\hline
kitten & 0 & 1 & 0 & \cdots\\
cat & 0 & 0 & 1 & \cdots\\
dog & 1 & 1 & 0 & \cdots\\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{array}
$$
	所以 cat 我们就可以表示为$$
	cat = onehot_{cat}^TE;~~~~onehot_{cat} = [0,1,0]^T
	$$
这样我们可以表示为Symbols = unique vectors. Representation = embedding symbols with **E**.
* **Generic idea behind embeding learning**
	1. Collect instances $t_i \in inst(t)$ of a word t of vocab V.
	2. For each instance , collect its context words $c(t_i)$(e.g. k-word window)
	3. Define some score function $score(t_i, c(t_i); \boldsymbol{\theta}, \boldsymbol{E})$ with upper bound on output.
	4. Define a loss: $$\boldsymbol{L}=-\sum_{t\in \boldsymbol{V}}\sum_{t_i\in inst(t)}score(t_i,c(t_i);\boldsymbol{\theta},\boldsymbol{E})$$
	5. Estimate:$$\widehat{\boldsymbol{\theta}},\widehat{\boldsymbol{E}}=\arg\min_{{\boldsymbol{\theta}},\boldsymbol{E}}\boldsymbol{L}$$
	6. Use the estimated **E** as your embedding matrix.

* **Scoring function matter**
	1. Embeds $T_i$ with **E**
	2. Produces a socre which is a function of how well $t_i$ is accounted for by $c(t_i)$, and/or vice versa.
	3. Requires the word to a account for the context(or the reverse) more than another word in the same place.
	4. Produces a loss that is differentiable w.r.t $\boldsymbol{\theta},\boldsymbol{E}$.

* **Neraul Embedding Models: C&W**
	![Alt Text](./img/word2vec/C&W.png 'model C&W')
	$$模型C&W$$
	* Prevents the network from **ingoring input and outputting high score**. During traing, for each sentence s we sample a distrctor sentnce z by randomly corruptting words of s. Minimise hinge loss.$$\boldsymbol{L}=\max(0, 1-(g_{\boldsymbol{\theta},\boldsymbol{E}}(s) - g_{\boldsymbol{\theta},\boldsymbol{E}}(z)))$$ 
	* representations carry information about what **neighbouring representations** should look like.**But**,it is not cheap to train, because it is fairy **deep**, and **convolution capture very local information**.

* **Neraul Embedding Models: CBoW**
	![Alt Text](./img/word2vec/CBoW.png 'model CBow')$$模型 CBoW$$
	**All linear**, so very fast. Bascically a cheap way of applying one matrix to all inputs.
* **Neraul Embedding Models: Skip-gram**
	![Alt Text](./img/word2vec/skip-gram.png 'model Skip-gram')$$模型 Skip-gram$$
	* **Fast**: One embedding versus $\mid C \mid$ embedding.
	* Just read off probalities from softmax.
	* Trade off between efficiency and more structed notion of context.
	
##参考文献
1. [oxford-cs-deepnlp-2017 Word Level Semantics](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%202a-%20Word%20Level%20Semantics.pdf)
2. [skip-gram model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
3. [egative sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
4. [Chinese Word Vectors](https://github.com/zhangsiqi951016/Chinese-Word-Vectors)