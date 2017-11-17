## Arxiv Summaries Week 09/11

### PassGAN: A Deep Learning Approach for Password Guessing

**Authors**:  Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, and Fernando Perez-Cruz.

**Arxiv Link**: https://arxiv.org/pdf/1709.00440v1.pdf

**Published on Arxiv**: September, 1 2017

**Executive Summary**: 
Up until now password generation has focused on relatively simple transformations such as concatenation of words, leetspeak, mixed letter cases. These transformations combined with Markov Models can be used to generate passwords, but depend on prior knowledge. PassGAN uses a Generative Adversarial Network to “provide a principled and theory-grounded take on the generation of password guesses” and compare their results to two common password guessing software: John the Ripper (JTR) and HashCat. 

**Notable Details**: 
This paper uses ideas from Improved training of Wasserstein GANs to improve training and comments on good parameters: batch size (64), iterations (199,000), n_critic (10), gradient penalty (10). Both the generator and discriminator networks have 5 residual layers and dimension 128, and used an Adam Optimizer with β1 (.5), β2 (.9), learning rate (10E-4). The paper also shows that more training iterations beyond 175,000-199,000 produce diminishing returns and may cause overfitting as fewer unique passwords are being generated. When combined with HashCat’s best64 rules, PassGAN outperforms all other methods matching 37.69% unique training passwords and a staggering 60.1% unique passwords in test set. This improvement over previous password generation methods means the PassGAN has learned to generate distinct passwords not previously generatable. 

**Suitable readers**:
Those who work in security/testing of systems. Anybody interested in generative models and unique applications of the newest and most powerful GAN technique. 


### CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices 

**Authors**: Caiwen Ding, Bo Yuan et al.

**Arxiv Link**: https://arxiv.org/pdf/1708.08917.pdf

**Published on Arxiv**: August 29, 2017

**Executive Summary**:
Deep neural networks (DNN) have been rapidly extended to many applications.  In the meantime, the size of DNNs continues to grow quickly. Improving the energy efficiency and performance while maintaining accuracy is very important in large-scale DNNs. In this work, researchers process neural networks and represent weights by using block-circulant matrices. CIRCNN utilizes the Fast Fourier Transform (FFT) to process convolution operations in the neural network. The computation complexity and storage complexity largely decreases in large-scale DNNs. Authors test CIRCNN in different hardware systems.The results show CIRCNN achieves 6- 102X energy efficiency improvement compared to the state-of-art results.

**Notable Details**:
Hardware acceleration methods would suffer from the frequent accesses to off-chip DRAM systems, which would cost 200x energy to access on-chip SRAM. For instance, Block-circulant Matrices would reduce the weight matrix parameters from 18 to 6. Paper assumes that layers could be represented by block-circulant matrices and the training generates a vector for each circulant submatrix. This reduced both storage and computational complexity of the training.

**Suitable readers**:
Those who are interested in implementing large-scale neural networks, especially for imaging processing applications.

### Fast Image Processing with Fully-Convolutional Networks 

**Authors**: Qifeng chen, Vladlen Koltun et al.

**Arxiv Link**: https://arxiv.org/pdf/1709.00643v1.pdf

**Published on Arxiv**: Sep 2, 2017

**Executive Summary**:
In this work, a general deep fully convolutional neural network is used to replace image operators for fast image processing. After training the network, the network could directly process the image without calling the original processing operators. Accuracy, speed, and compactness are the three criteria used for evaluating the performance of the network. The main architecture proposed is a Context Aggregation Network (CAN). CAN uses N layers of the same height and width, but varying number of features. Strided convolution is used to aggregate long-range information in all three dimensions without losing resolution ( information of each feature in a given layer is aggregated from all features of the previous layer). Three alternative fully-convolutional network architectures are also explored. 

**Notable Details**:
The network achieved state-of-the-art accuracy in approximating 10 image processing algorithms, including multiple variational models, multiscale tone and detail manipulation, photographic
style transfer, nonlocal dehazing, and non-photorealistic stylization, at an average speed up factor of nearly 50.

**Suitable readers**:
Those who are interested in implement imaging processing or filter design.

### Unsupervised feature learning with discriminative encoder

**Authors**: Gaurav Pandey, Ambedkar Dukkipati

**Arxiv Link**: https://arxiv.org/pdf/1709.00672.pdf

**Published on Arxiv**: 3 Sep 2017

**Executive Summary**:
A common unsupervised feature extraction technique works with a joint distribution of observation x and latent variables z.  Such an approach often learns an encoding distribution P(z|x) and a decoding distribution p(x|z). With the goal of optimizing the marginal distribution p(x), such a network, given sufficiently flexible decoders, are not guaranteed to learn useful latent-space representations, as x and z can in the extremely limit be independent. This paper thus maximizes joint distribution q(x,z)=p(x|z)p(z) with p(z) a prior distribution to prevent collapse of latent space representation. 
Furthermore, to prevent encoder from mapping to initial (random) states, and decoder reconstructing therefrom, an adversarial regularization is added. Borrowing from ideas of Generative Adversarial Networks, the discriminative network attempts to be unsure what latent vector to assign to  a fake sample belongs to, while the generative network tries to fools the discriminator into mapping the discriminator to the latent vector from which the sample was generated. 


**Notable Details**:

The authors show that the resulting network achieves state-of-the art performance in several unsupervised and semi-supervised tasks such as image generation and clustering. Specifically, the network reportedly achieves 3.12% error on unsupervised clustering of MNIST(20 clusters), 17.24% error on semisupervised  CIFAR-10 (4000 labeled), and 5.22% error on semi-supervised SVHN (4000 labeled).   

**Suitable readers**:
Audiences interested in unsupervised and semi-supervised learning. 

### Unsupervised learning through one-shot image-based shape reconstruction

**Authors**:  Dinesh Jayaraman, Ruohan Gao, and Kristen Grauman

**Arxiv Link**: https://arxiv.org/pdf/1709.00505.pdf

**Published on Arxiv**: September, 1 2017

**Executive Summary**: 
Understanding the structure of 3-D entities from 2-D pictures has previously focused on category specific models, and focuses on the reconstruction as an end goal. In this work, the authors train a category-agnostic neural network from scratch. This neural network is capable of a full 3-D reconstruction of an object from an arbitrary 2-D view, which constitutes one shot learning as only one 2-D view is given. The authors then show that the neural network automatically learns intermediate representations, and that the final neural net is competitive in unsupervised tasks such as object recognition and object retrieval.

**Notable Details**: 
The neural network contains separate modules for processing the input image (implemented as a 5 layer step down convolutional neural network) and sensing elevation. This creates an intermediate representation which is run through two fully connected “fusion” layers. Finally this representation is “decoded” using a nearly identical convolutional neural network to the input processing, but stepping up. This step is “mentally rotating” the viewpoint to produce a viewgrid of various outputs as seen from other angles.  The loss function seeks to minimize the L_2 distance of the generated viewgrids from ground truth up to certain rotations. 

**Suitable readers**:
One shot learning enthusiasts along with those who are interested in representation learning/ unsupervised learning. 

## Arxiv Summaries Week 09/18

### Transform Invariant Auto-encoder

**Authors**:  Tadashi Matsuo, Hiroya Fukuhara and Nobutaka Shimada

**Arxiv Link**: https://arxiv.org/pdf/1709.03754.pdf

**Published on Arxiv**: September, 12 2017

**Executive Summary**: Auto-encoders map data to latent space and vice versa. Typically using L2 norm on the reconstructed data, inputs containing the same spatial subpattern may be mapped to very different latent vectors if the subpatterns appear in different locations. This paper introduces a novel loss function that makes the auto-encoder robust to shifts in input. The authors demonstrated network applied to MNIST and hand-object interactions and showed its robustness against translations. 

**Notable Details**: Notation: {D: decoder, E:encoder, I: input, T: spatial transform}
The new cost function includes a transform variance term, which measures sum of the error |D(E(I))-D(E(T(I)))| for a variety of  transforms T. This in effect forces the auto-encoder to reconstruct similar images regardless of the spatial location of the pattern. The cost function also includes a modified restoration error term. Instead of |D(E(I))-I| as they usually are in auto-encoders, this term becomes min|D(E(I))-T(I)|, where the minimum is taken over all transforms T. In addition, a transform inference model R can be trained with the loss |R(I)-argmin|I-T(D(E(I)))||, where the argmin is again taken over T. In words this trains R to predict the particular T applied to the input image I. 

**Suitable readers**:
Those interested in generative networks, or latent dimensionality reduction/clustering. 

### Deep Subspace Clustering Networks

**Authors**:  Pan Ji , Tong Zhang , Hongdong Li , Mathieu Salzmann , Ian Reid

**Arxiv Link**: https://arxiv.org/pdf/1709.02508.pdf

**Published on Arxiv**: September, 8 2017

**Executive Summary**: Subspace clustering aims to cluster data in low-dimensional subspaces. Most works to date focuses on linear subspaces, and works by forming affinity matrix of every pair of data and cluster the data based on the (assumed Euclidean) affinity. Non-linear subspaces have been explored mostly with a fixed kernel that replaces the Euclidean norm. This paper learns a kernel with an auto-encoder architecture, where the encoder serves as the nonlinear map from data space to subspace. The proposed auto-encoder learns subspace structure with a self-expressive layer, which encourages points in the subspace to be representable as linear combinations of other points in the same subspace. 

**Notable Details**: The authors performed experiments on Extended Yale B, ORL, and COIL20/COIL100 datasets. Each of these datasets explore different images of a given number of  human face or 3D objects under different lighting  or orientation. The authors showed that the network significantly improved upon state-of-the-art results in clustering the correct subject. To enable the network to learn with a limited dataset, the autoencoder is first trained without the self-expressive layer, the fine-tuned with the layer. The self-expressiveness matrix is formulated as a fully connected layer and its parameters are trained directly with back-propagation. 

**Suitable readers**:
Those interested in generative networks, or latent dimensionality reduction/clustering. 

### Explore, Exploit or Listen: Combining Human Feedback and Policy Model to Speed up Deep Reinforcement Learning in 3D Worlds

**Authors**:  Zhiyu Lin, Brent Harrison, Aaron Keech, and Mark O. Riedl

**Arxiv Link**: https://arxiv.org/pdf/1709.03969.pdf

**Published on Arxiv**: September, 12 2017

**Executive Summary**: 
Interactive machine learning allows humans to demonstrate correct behavior in order to improve and speed-up learning in complex environments. This paper demonstrates an approach enabling humans, or oracles, to provide examples of correct behavior to DQN agents. 

The approach adds an “Arbiter” to the pipeline which decides whether to follow the DQN’s actions or the oracle’s actions (if they differ) based on the past performance of the DQN and the oracle. It can also select a random action based on an exponentially decaying probability. The paper tested their approach on a custom Minecraft map, training the agent to pick up a specific object in the map while providing advice through a simulated oracle. When the oracle provided good advice (70% and 90% accuracy), the model’s performance surpassed that of a DQN. When the oracle provided random advice (50% accuracy), the model’s performance matched that of a DQN.

**Notable Details**: 
The agent is limited to four directions: North, East, South, West. When turning, the agent's viewing angles may be perturbed up to 2 degrees after turning.

The Arbiter has three checks: exploration (for selecting a random action), confidence, and consensus. Confidence selects between DQN and oracle actions based on past performance of the DQN. Consensus selects between the DQN and oracle randomly according to a moving probability that favors the DQN when the DQN and oracle actions differ. In addition,  the consensus check has 2 parameters that must be tuned: f1 and f2.

The agent in the test environment is punished with a score of -1 for each step taken and rewarded with a score of 100 when it successfully picks up the object.

**Suitable readers**:
Those interested in methods for training RL agents and in HCI related to machine learning agents.

### Ensemble Methods as a Defense to Adversarial Perturbations Against Deep Neural Networks

**Authors**:  Thilo Strauss, Markus Hanselmann , Andrej Junginger , Holger Ulmer

**Arxiv Link**: https://arxiv.org/pdf/1709.03423.pdf

**Published on Arxiv**: September, 11 2017

**Executive Summary**: 
Deep neural networks are highly vulnerable to adversarial examples, with some perturbations not being visible to human perception. Constructing adversarial examples is done mainly through two examples, fast gradient sign method (FGSM), which adds noise in the direction of the gradient, and basic iterative method (BIM), which iteratively applies FGSM to data points. Several ensemble methods are discussed and then evaluated with adversarial datasets constructed using FGSM and BIM. The methods discussed were training multiple of the same networks but with different weights, training multiple classifiers each with slightly different architectures, bagging of the training data, and adding small gaussian noise to the dataset. These four methods are tested against FGSM and BIM on the MNIST and CIFAR-10 dataset. The test results indicate that applying the ensemble methods, especially applying gaussian noise, make each dataset more robust against adversarial examples, and the following ensemble makes the classification more robust against adversarial examples. 

**Notable Details**: 
When testing against adversarial examples generated using FGSM in MNIST, classification accuracy would drop to 35-56% from roughly 99% on the unperturbed dataset. The ensemble method proposed led to a classification accuracy of 57%-78% on the perturbed data. 

FGSM relies on calculation of the gradient and it is much more difficult to calculate the gradient of an ensemble of networks. The two methods for doing so are using the gradient of one of the networks and applying it across the whole architecture in an effort to fool the whole system, or taking a sum of the gradients of the whole system. 

**Suitable readers**:
Those interested in training for adversarial examples in deep neural networks, with applications including self-driving cars. 

### Learning to Compose Domain-Specific Transformations for Data Augmentation

**Authors**:  Alexander J. Ratner , Henry R. Ehrenberg, Zeshan Hussain, Jared Dunnmon, Christopher Ré

**Arxiv Link**: https://arxiv.org/pdf/1709.01643v1.pdf

**Published on Arxiv**: September, 6 2017

**Executive Summary**:  
A common technique to boost accuracy of machine learning methods is data augmentation. This is frequently found in computer vision where images will often be mirrored, brightened, translated, and generally transformed in various ways that respect a class level invariance. Often these will not just be done in isolation, but rather they will be applied sequentially with different parameters (such as degree of rotation or amount of increase of brightness). However, haphazardly doing these can lead to bizarre and unrealistic images. This paper investigates automated ways to apply these dataset augmentation techniques. To do this, they have a gene network select which transformation to apply and they have a second network test whether or not the example came from the distribution, among other regulatory features. They trained their method using policy gradient and applied their method to CIFAR-10,  NIST Automatic Content Extraction (ACE) corpus, and the Digital Database for Screening Mammography (DDSM) dataset. They show that their method outperform random and heuristic methods for composing data augmentation transformations.

**Notable Details**: 
In order to prevent the generator from always outputting a no-op transformation, the authors included a diversity term in their loss function. The diversity term was a distance metric that was either over the raw input or the features before the softmax layer. Another regularization term looked at the difference in predicted classes both before and after transformation was applied and punished the generator when the gap was large.


**Suitable readers**: 
Those interested in data augmentation techniques or those wishing to get a little extra performance out of a system. 

their loss function. The diversity term was a distance metric that was either over the raw input or the features before the softmax layer. Another regularization term looked at the difference in predicted classes both before and after transformation was applied and punished the generator when the gap was large.


**Suitable readers**: 
Those interested in data augmentation techniques or those wishing to get a little extra performance out of a system. 

## Arxiv Summaries Week 09/25

### A Deep Generative Framework for Paraphrase Generation

**Authors**: Ankush Gupta, Arvind Agarwal, Prawaan Singh, Piyush Rai

**Arxiv Link**: https://arxiv.org/pdf/1709.05074.pdf

**Published On**: September, 150 2017

**Executive Summary**: The authors open the article by emphasizing the importance of paraphrasing (i.e., generate a sentence with similar semantic structure given the original sentence) in the context of Q&A, information retrieval, conversational agents, etc. However, due to the complex nature of the task, the problem suffers from the lack of training data, compelling the authors to leverage semi-supervised learning.

**Notable Details**:  The proposed methodology is as follows:

* An LSTM-based encoder that transforms a sentence to an intermediate representation x, and a decoder that takes an intermediate representation x’ and transforms it back into a sentence.
* A conditioned-VAE  encoder that learns a posterior distribution of a random latent code z conditioned on the intermediate representation of the input sentence x  (i.e., qɸ(z|x)), and a decoder that  learns a posterior distribution of x that takes as input a random latent code z (i.e., pθ(x|z)).

The parameters in the model are learned in two distinct phases. First, auto-encoding capability is learned by training on (sentence, sentence) examples (unsupervised). Then, paraphrasing capability is learned by training on (sentence, paraphrase) examples (supervised). Once trained, the model is able to produce multiple paraphrases with beam-search  procedure.

Quantitatively (i.e., BLEU, METEOR, and TER score), the model beats state-of-the-art baselines on the MSCOCO dataset, but only reports against trivial baselines on the recently-released Quora paraphrasing dataset . Qualitatively, the authors claim the model produces paraphrases with similar relevance and readability scores as to ground truth, however they do not report the qualitative evaluations of their baselines to compare against.

### Exploring Human-like Attention Supervision in Visual Question Answering

**Authors**: Tingting Qiao, Jianfeng Dong, Duanqing Xu

**Arxiv Link**: https://arxiv.org/abs/1709.0630

**Published On**: September 19, 2017

**Executive Summary**: Attention mechanisms in Neural Nets often don’t attend over the same areas humans would. Errors in attention distributions will often lead to incorrect outcomes, for example failure to detect specific objects. Traditionally, attention mechanisms are only optimized indirectly. This paper uses the VQA-HAT (Visual Question Answering Human ATtention) Dataset. For each image, three annotators were given a blurred image and a pixel budget to unblur; their allocation of pixels is the “human attention.” A CNN on the image and a GRU on the question then generate a sequence of predicted attention distributions explicitly. Then a second GRU takes in this sequence of distributions and explicitly predicts the question answer. 

**Notable Details**: Unsurprisingly, correlation to human attention outperforms any other model, and even slightly outperforms human (since the model has lower variance than a single annotator). Comparing supervised attention to unsupervised attention on VQA 2.0, the authors noted a .15% increase in overall performance, and a .42% increase in counting problems.

**Suitable Readers**: Anyone interested in attention or computer vision, and especially visual question answering.

### Mitigating Evasion Attacks to Deep Neural Networks via Region-based Classification

**Authurs**: Xiaoyu Cao, Neil Zhenqiang Gong

**Arxiv Link**: https://arxiv.org/pdf/1709.05583.pdf

**Published On**:  September 17, 2017

**Executive Summary**: Recent studies show that deep neural networks are vulnerable to adversarial examples; that is, given a test sample, an attacker can add a small carefully crafted noise to it, so that this example is wrongly predicted by the DNN classifier. The authors find that using existing adversarial example generation methods, the generated adversarial examples are close to the classification boundary. Therefore, they propose a new defense strategy, called region-based classification. Specifically, traditional classifiers make predictions based on the input alone. In contrast, a region-based classifier makes predictions for a set of samples centered at the input example, then selects the most frequently predicted label as the final prediction.

**Notable Details**: They evaluate their defense proposal on MNIST and CIFAR-10 datasets. They use CW attack to generate adversarial examples, which is the state-of-the-art approach.

* First, they assume that the adversary has full knowledge of the model architecture and parameters, but does not know that the model uses region-based classification method. Their results show that on MNIST, the attack success rate drops from 100% to at most 16%, while on CIFAR-10, it drops from 100% to at most 7%. 

* Further, they assume a stronger adversary, who not only has full knowledge of the model, but also knows that the model is using region-based classification. In this case, the adversary can increase the magnitude of the noise added to the original image, so that it is moved further away from the classification boundary. The results show that when the adversary doubles the magnitude of the noise, the attack success rate is around 60% for both MNIST and CIFAR-10 datasets. 

* In addition, the model’s accuracy on original test set does not drop.

**Suitable Readers**: Those who are interested in adversarial deep learning, in particular, who are working on generating adversarial examples for deep neural networks and developing defense strategies against adversarial examples.

### DropoutDAgger: A Bayesian Approach to Safe Imitation Learning

**Authors**: Kunal Menda, Katherine Driggs-Campbell, and Mykel J. Kochenderfer

**Arxiv Link**: https://arxiv.org/abs/1709.06166

**Published On**: September 18, 2017

**Executive Summary**: In reinforcement learning (RL), we’re concerned with teaching an agent (any trainable actor) to take optimal or near-optimal actions in an environment (world) that is associated with some enumerable state. Each state-action pair is associated with a reward that the agent seeks to maximize. Often, a simple but effective method for training an agent in an environment E is to replicate the behavior of an expert that already functions near optimally in E. This technique is called behavioral cloning, and it’s part of a set of algorithms referred to as imitation learning algorithms. Beyond copying the expert, we can enhance behavioral cloning by allowing the expert to observe the agent’s behavior and provide corrections. The algorithm associated with this general procedure is called DAgger. DAgger works very well in simulated environments; however, agents that are training with DAgger in real world environments may venture into unsafe conditions  (i.e. an autonomous vehicle on the edge of a cliff, a drone headed directly toward a tree, etc) if they fail to perfectly replicate an expert or are exposed to a novel state. DropoutDAgger attempts to  address this problem.

**Notable Details**: For an agent in state  within an environment E, the authors propose to attempt to estimate the uncertainty of the action  that the agent intends to take. Assuming that a is produced by a deep neural network with input s, the proposal makes use of the following method to obtain a larger distribution of potential actions:

* Apply a random dropout filter to each weight layer of the neural network
* Evaluate the neural network on s to produce an action a-i
* Add a-i to the potential action distribution and go to step 1

After obtaining a Gaussian distribution of N actions via this dropout technique, the mean  is then computed. The method then considers the action that the expert would take in state s, denoted, . If the Euclidean distance between m and  falls below some threshold T, then the agent’s original action a is deemed safe and is taken . Otherwise,  is taken. This algorithm is called DropoutDAgger.

DropoutDAgger achieves the safety of behavioral cloning (which is safer than DAgger in real world environments) while achieving comparable rewards to DAgger (which achieves better rewards than behavioral cloning).

**Suitable Readers**: Those who are interested in accessible, powerful reinforcement learning techniques. This is relatively simple to implement with minimal background in the field.

### Feature Engineering for Predictive Modeling using Reinforcement Learning

**Authors**: Udayan Khurana, Horst Samulowitz, Deepak Turaga

**Arxiv Link**:  https://arxiv.org/pdf/1709.07150.pdf

**Published On**:  September 21, 2017

**Executive Summary**: The majority of ML algorithms deal well with feature selection, but for the most of predictive analytics simple feature selection isn’t sufficient. When data model is more complicated, feature engineering (FE) is required.  Feature engineering is a manual process that is done through trial and error. The process is not scalable nor efficient and is prone to bias and error, and this porcess should be automated.  Some challenges of this problem are: the number of possible features is unbounded (transformations can be applied repeatedly) and to perform exploration of the features automatically, there must be some estimation of effectiveness of a transformations based on prior experience.

**Notable Details**:  

* Transformation Graph: a DAG of all possible transformation where D0 is the initial dataset or a dataset derived from it through transformation path. Periodically combine nodes as union set of different transformations of the features.  
* Graph Exploration under a Budget Constraint: since exhaustive exploration is not an option, there is a need of limited time budget constraint.
* Traversal Policy Learning: Consider the graph exploration process as a Markov Decision Process. 

The transformations improved the final gain in performance by about 51%, measured on the 48 datasets. 

## Deep Reinforcement Learning that Matters

**Authors**: Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger

**Arxiv Link**:  https://arxiv.org/abs/1709.06560

**Published On**:  September 19,2017

**Executive Summary**: Recent results in deep reinforcement learning (RL) are difficult to replicate due to high sensitivity to details of implementation, hyperparameters, and benchmark environment. This paper provides empirical demonstration of the effects of several variables on the performance of several RL algorithms. With the goal of keeping RL relevant, it concludes by proposing methods by which new algorithms may be fairly evaluated.

**Notable Details**: The particulars of the environment, the RL algorithm, and specific interactions between the two create challenges when attempting to evaluate new algorithms: For a fixed learning algorithm on a particular benchmark task, a small modification such as using ReLU instead of Tanh activations can boost final reward by 5x. The difficulty of discovering the best hyperparameters for a particular algorithm and environment complicates comparison with baselines and motivates the need for hyperparameter-free learning algorithms. Moreover, several state-of-the-art algorithms demonstrate remarkably dissimilar results on different benchmarks for reasons ranging from finding local optima (read: cheats) to the scale of rewards. This finding necessitates comparing algorithms across a diverse set of benchmarks. It also raises the question of whether reward (as opposed to real-world performance) is truly a valid measure of success. However, even when the task, model, and algorithm are held constant, statistically significant changes in performance may be derived by simply varying the random seed.
While averaging over several runs and publishing the implementation with the paper, maintaining the rapid growth of RL will require reevaluating the reproducibility of results.

**Suitable Readers**: Those having basic familiarity with deep RL and a desire to understand how specific design choices affect experimental results.

## Arxiv Summaries 10/22

### Image similarity using Deep CNN and Curriculum Learning

**Authors**:  Srikar Appalaraju, Vineet Chaoji

**Arxiv Link**:  https://arxiv.org/pdf/1707.00683.pdf

**Published On**:  September 26, 2017

**Executive Summary**:  In order to train an effective image similarity model, the authors were interested in generating a feature embedding which optimizes a distance metric such that similar images (of the same class) are within a margin and dissimilar images are outside a margin. The combine previously developed Siamese architecture with a curriculum learning framework. Curriculum learning is about ordering training examples such that “easier” examples are presented before “harder” examples. Testing on CIFAR-10, they were able to achieve an 18% improvement in similarity predictions with curriculum learning over random pairing. Curriculum learning improves training speed given that most random pairs of images are not useful for learning a good embedding (e.g most embeddings will have an easier time predicting between cars and humans as opposed to cars and trucks).

**Notable Details**: The Siamese net was built from 3 CNN. CNN1 was VGG-16 which was trained to capture the strong invariance associated with image classification. CNN2 and CNN3 are shallow networks (3 conv layer deep) that operate on downsampled images and is trained independently from CNN1. Since they are shallow, they will encode visual appearance features as opposed to class invariance that deep layers have. The output of these networks are used to construct a 4096-D feature embedding. 

The motivation for curriculum learning is based on child development psychology which shows that children and animals learn faster when presented easier images before showing harder examples. 

In order to achieve the curriculum learning aspect, the authors implemented pair constraints. At a given stage, the pair constraints filtered data such that hard positives had a maximum distance and hard negatives had a minimum distance. As training progresses, the constraint relaxes. The positive pairs distance increases (resulting in farther and farther similar images) and the minimum pairs distance decreases (images which are dissimilar look more similar). The authors found that constructing pair constraints with L2-distance over image pixels had the best performance.

If hard negatives that were presented too early in the training procedure the network would converge to a bad minima. Therefore picking a -descent rate for the pair constraint was crucial to observing the 18% accuracy gain. Accuracy was computed by asking the network to compute pairs of embeddings and threshold the L2-distance. Random pairing achieved 78% accuracy, but with curriculum learning, the network was able to achieve 92.6% in the same amount of time.

### DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks
**Authors**:  Sen Wang, Ronald Clark, Hongkai Wen and Niki Trigoni
**Arxiv Link**:  https://arxiv.org/pdf/1709.08429.pdf
**Published on Arxiv**: September 25, 2017

**Executive Summary**: This paper studies monocular visual odometry (VO) problem. Most VO algorithms are developed using feature extraction, matching, motion estimation, local optimisation, etc. They are thus also accompanied by the need for careful fine-tuning by an expert to work well in different environments. This paper presents an end-to-end framework called Recurrent Convolutional Neural Networks (RCNNs). Being end to end, this method doesn’t need any specific processing that are common in traditional VO algorithms.  Extensive experiments on the KITTI VO dataset show competitive performance to state-of-the-art methods, verifying that the end-to-end Deep Learning technique can be a viable complement to the traditional VO systems.
**Notable Details**: 
* They provide the first ever end-to-end solution to the problem of VO in a way that does not need any processing pipeline like feature extraction, motion estimation etc ( not even camera calibration is required)
* Propose a new deep architecture called Recurrent Convolutional Neural Networks that enable it to learn complex motion dynamics in a sequence of images which is hard even for humans.

Architecture: Instead of using popular CNNs  such as VGGNet, GoogLeNet they propose a new architecture. The claim is that the existing architectures are designed for classification and detection problems in mind, which means that they are trained to learn knowledge from appearance and image context. However VO which is rooted in geometry should not be closely coupled with appearance. Thus, a framework which can learn geometric feature representations is of importance to address the VO and other geometric problems. It takes a video clip or a monocular image sequence as input. At each time step, two consecutive images are stacked together to form a tensor for the deep RCNN to learn how to extract motion information and estimate poses. The image tensor is fed into the CNN to produce an effective feature for the monocular VO, which is then passed through a RNN for sequential learning. 
Loss Function: The proposed RCNN based VO system can be considered to compute the conditional probability of the poses Yt = (y1 , . . . , yt ) given a sequence of monocular RGB images Xt = (x1, . . . , xt) up to time t in the probabilistic perspective. To learn the hyperparameters θ of the DNNs, the Euclidean distance between the ground truth pose (pk , ϕk ) at time k and its estimated one (^pk , ^ϕk ) is minimised. 

Among all the 11 training dataset, only a single one has velocities that are higher than 60 km/h. Also, the images are captured at only 10 Hz, which makes the VO estimation more challenging during fast movement. The large open area around highway (lacking of features) and dynamic moving objects can degrade the accuracy as well. Conventional geometry based methods could increase feature matching and introduce outlier rejection, such as RANSAC. However, for the DL based method, it is unclear how to embed these techniques yet. A feasible solution is to train the network with more data which not only reflects these situations but also is deliberatively augmented with noise, outliers, etc., allowing the network itself to figure out how to deal with these problems.

### Machine Learning Models that Remember Too Much
**Authors**:  C. Song, T. Ristenpart, V. Shmatikov
**Arxiv Link**:  https://arxiv.org/pdf/1709.07886.pdf
**Published on Arxiv**: September 22, 2017

**Executive Summary**: As "machine learning as a service" (MLaaS) platforms gain more traction among non-experts, ML providers can take advantage of users, particularly by gaining access to sensitive training data. While ML providers are often restricted from direct access to training data, this paper demonstrates a method by which an adversarial provider can take advantage of overprovisioning to create a model which silently memorizes information about the training data without compromising testing accuracy, and allows the adversary to later extract the memorized information from the customer's model in both white-box and black-box settings. With this, the authors expose a large class of security hazards inherent to MLaaS platforms which hide the model's training algorithms from users, and demonstrate the need for a more rigorous treatment of privacy in this setting.

**Notable Details**: An adversarial provider A supplies the training algorithm to the data holder, and does not observe the execution of the algorithm. After training, A obtains white/black box access to the model.

White box setting: The LSB encoding attack calls for the training algorithm to directly post-process model parameters θ by setting the lower b bits of each parameter to a bit string s extracted from the training data; A may then access θ to extract the bits of s after model training. A second attack adds a malicious “regularization” term C =  -|Pearson correlation coefficient(θ, s)| to the loss function, driving the gradient direction towards a local minimum where s and θ are highly correlated. The last attack adds a term P to the loss function forcing the model to encode sensitive information in the signs of the parameters, penalizing each parameter with a sign that does not match the encoding of s. 

Black box setting: An algorithm resembling data augmentation is used to train model Mθ on two tasks: the original classification task, as well as a requirement that, for a particular synthetic input i and secret information d, Mθ(i) = d. This technique forces the model to become overfitted to the labels of a relatively small number of synthetic inputs; due to the large expressive capacity of neural networks, this overfitting can be achieved without significant impact on the model's performance on the rest of the input space. 

Attacks were highly effective on CIFAR10, Labeled Faces in the Wild, FaceScrub, 20 Newsgroups, and IMDB datasets, and with model architectures including CNNs and Residual Networks; white box attack on a text classifier can leak 70% of a training corpus without loss of accuracy, and black box attack on a classifier from FaceScrub data allows full recovery of 17 images through leakage of 1 bit per query. 
The findings are of interest to non-experts in ML who rely on MLaaS platforms to model sensitive data, as well as to researchers at the intersection of security and deep learning. While countermeasures (examining the distribution of parameters; adding noise to low bits of parameters) are effective against specific attacks, the work exposes a fundamental security hazard in the MLaaS paradigm; users with sensitive data should avoid any platform providing a training algorithm not thoroughly understood and examined by the user. Questions of how to define and test the requirement that a model stores only the information required for the task at hand, rather than encoding extraneous information, remain open.

