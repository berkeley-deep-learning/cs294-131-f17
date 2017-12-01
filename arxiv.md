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

### Deep Reinforcement Learning that Matters

**Authors**: Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger

**Arxiv Link**:  https://arxiv.org/abs/1709.06560

**Published On**:  September 19,2017

**Executive Summary**: Recent results in deep reinforcement learning (RL) are difficult to replicate due to high sensitivity to details of implementation, hyperparameters, and benchmark environment. This paper provides empirical demonstration of the effects of several variables on the performance of several RL algorithms. With the goal of keeping RL relevant, it concludes by proposing methods by which new algorithms may be fairly evaluated.

**Notable Details**: The particulars of the environment, the RL algorithm, and specific interactions between the two create challenges when attempting to evaluate new algorithms: For a fixed learning algorithm on a particular benchmark task, a small modification such as using ReLU instead of Tanh activations can boost final reward by 5x. The difficulty of discovering the best hyperparameters for a particular algorithm and environment complicates comparison with baselines and motivates the need for hyperparameter-free learning algorithms. Moreover, several state-of-the-art algorithms demonstrate remarkably dissimilar results on different benchmarks for reasons ranging from finding local optima (read: cheats) to the scale of rewards. This finding necessitates comparing algorithms across a diverse set of benchmarks. It also raises the question of whether reward (as opposed to real-world performance) is truly a valid measure of success. However, even when the task, model, and algorithm are held constant, statistically significant changes in performance may be derived by simply varying the random seed.
While averaging over several runs and publishing the implementation with the paper, maintaining the rapid growth of RL will require reevaluating the reproducibility of results.

**Suitable Readers**: Those having basic familiarity with deep RL and a desire to understand how specific design choices affect experimental results.

## Arxiv Summaries Week 10/02

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

### Comparison of Batch Normalization and Weight Normalization Algorithms for the Large Scale Image Classification
**Authors**:  Igor Gitman, Boris Ginsburg
**Arxiv Link**:  https://arxiv.org/abs/1709.08145
**Published on Arxiv**: September 24, 2017

**Executive Summary**: Batch normalization (BN) and weight normalization (WN) are two effective approaches to normalize the propagation of deep neural networks. BN is reveal to take a significant portion of computation time in this paper. With the launch of the next generation of GPUs, BN will be even less efficient compared to the optimization on convolutions. Although the BN is becoming a standard for very deep network, author try to find an alternate of BN with WN methods which are more efficient. Surprisingly, the  normalization ability of BN cannot be replaced for very deep networks in the large dataset like ImageNet. The training loss and accuracy curve of WN is slightly better than BN but the test accuracy of using BN is much higher.

**Notable Details**: Batch normalization takes 24% computation time while convolution takes 60% with a ResNet model in Pascal GPU. In the next generation of Volta GPU, convolution will be optimized while BN are not able to be optimized.

Three state of the art weight normalization methods are tested, namely the vanilla weight normalization, Normalization Propagation and Normalization with ReLU. The idea is to match the norm between the input and output of a layer. It is reveal that this idea may be violated during training.

Comparison are conducted on both small and large datasets. The smaller CIFAR-10 dataset is tested with a shallow network structure. Other hyperparameters are tuned for each BN and WN method so that they reach the ‘maximum’ performance. They all achieve 100% training accuracy with similar speeds on the smaller CIFAR-10 dataset. The state of the art WN technic bypass BN by 1-2% on the CIFAR-10.

The comparison on the larger ImageNet dataset shows the opposite. The ResNet-50 is used for training. Although the training accuracy is similar between two methods, the BN has 6% higher accuracy on the test set. Even adding dropout layer and increasing the weight decay does not help the overfitting problem in BN. The assumption of the orthogonality of weights is observed to be violated in the very deep network. The norm of outputs stays similar for the first few epochs. It blows up in the last layer and vanishes in the first layer of ResNet-50. The small fraction of change in the norm is exponentially magnified through the training process.

The surprising effect makes the attempt fail to replace the inefficient BN method. The WN are limited to shallow network and small datasets for classification. The call for an efficient substitute of BN still remains. 

### Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping

**Authors**:  Konstantinos Bousmalis, Alex Irpan, Paul Wohlhart, Yunfei Bai, Matthew Kelcey, Mrinal Kalakrishnan, Laura Downs, Julian Ibarz, Peter Pastor, Kurt Konolige, Sergey Levine, Vincent Vanhoucke

**Arxiv Link**:  https://arxiv.org/pdf/1709.07857.pdf

**Published on Arxiv**: 22 Sep 2017

**Executive Summary**: This paper proposes a method for reducing the number of real-world trials needed to acquire vision-based robotic grasping skills by combining real-world data with large amounts of simulated data using domain adaptation techniques. Using the proposed method the amount of real-world training data can be reduced by a factor of 50. The method jointly applies pixel-level and feature-level domain adaptation. First simulated images of the robot arm performing grasp trials are fed into a generator transforming them into realistic looking images. These in turn are fed into a grasp prediction network which is used to plan grasps with a real robot. An additional discriminator loss is added to the optimization enforcing high-level features of the grasp prediction network to be invariant among the two domains.

**Notable Details**: A crucial requirement for the proposed GraspGAN architecture is that the domain-adapted images may not change semantics and physical properties of the scene. (The objects and the robot must be drawn at the exact same locations as in the source image to avoid errors in the grasp prediction). This is ensured by training the generator to output semantic segmentation masks of the scene in addition to the domain-adapted image (ground-truth for semantic segmentation is available through the simulator).  

In addition to the adverserial feature-level domain adaptation (using domain adverserial neural networks DANN) a feature matching loss between the activations of the last convolutional layer of the grasp prediction network is used since it provides more gradient information than the binary labels.
A variant of batch normalization is introduced accounting for the fact that batch statistics differ significantly among simulated and real-world data. 
The domain adaptation techniques are compared with scene randomization (changing camera pose, bin location, backgrounds and dynamical properties of the simulated objects) and it is shown that the proposed domain adaptation scheme significantly outperform naiive randomizatoin approaches.

## Arxiv Summaries Week 10/09

### Efficient K-Shot Learning with Regularized Deep Networks

**Authors**: Donghyun Yoo , Haoqi Fan , Vishnu Naresh Boddeti , Kris M. Kitani
**Arxiv Link**: https://arxiv.org/pdf/1710.02277.pdf
**Published Date**: October 6th, 2017. 

K-Shot Learning often suffers from severe overfitting on the training set due to the large number of parameters in deep networks. The standard procedure for K-shot learning is to train a regular convolutional neural network on a large dataset such as ImageNet so the convolutional layers can learn filters. Then, the last layer or last few layers are re-initialized and trained on the smaller dataset. Efficient K-Shot Learning with Regularized Deep Networks aims to solve the problem of overfitting by regularizing back-propagated gradients.  
	For each convolutional filter that is learned, there are often several filters per layer that share a similar structure. The authors of this paper propose that these subtle differences in filters are responsible for a large part of overfitting. Their work groups similar filters by their outputs on the training set, and apply the average of all the group’s gradients to each filter in the group. This is the “regularization” technique they propose.
	To find the optimal hyperparameters for their algorithm (threshold to group filters, number of groups, etc), they use a recurrent RL network. This approach is similar to Quoc Le and Barret Zoph’s work from 9/25’s talk. The loss for grouping is the L2 norm for minimizing inter-group filter output distance and maximizing intra-group filter output distance.
	Experiments were run on the Office data set introduced by Saenko et al. 2010. This dataset has 31 classes and 20 examples per class of a common office setting. Grouping Neurons by Activations (GNA) improve performance from fine-tuning by almost 10% (from 70.07% to 79.94%). Further optimizing with their RL hyperparameter search sees another 5% increase to 85.04%.

### SE3-Pose-Nets: Structured Deep Dynamics Models for Visuomotor Planning and Control

**Authors**: Arunkumar Byravan, Felix Leeb, Franziska Meier and Dieter Fox
**Arxiv Link**: https://arxiv.org/pdf/1710.00489.pdf 
**Published Date**: Oct.2 2017

**Executive Summary**:  This paper is an extension of the SE3-Net paper from the same lab. While the old SE3-Net only allows learning of robot kinematics from point clouds, this paper extends the previous model by using a pre-trained network for visuomotor control. Deep learning models typically strives to find a low-dimensional, noise-free representation of the high-dimensional data (e.g. picture/voice recognition), and this paper makes use of the advantage that in Robotics, the low-dimensional representation is actually fixed: it’s just the kinematics of all joints defined as SE3 rigid body transforms. Therefore, the learned low-dimensional representation is human-interpretable and the training process is a lot easier. The experiment proved that the network is capable of moving from any initial joint position to the final state (given by a 3D point cloud) with state-of-the-art precision, even with minimal supervision. 

**Notable Details**:  Unlike most deep-learning models, the SE3-Pose-Net learns the point mask and SE3 rigid body kinematics of the robot as the intermediate result, and the last part of the pipeline is simply applying the rigid body transform using the known dynamics. The encoder and decoder layers are also simple conv-pool and fully connected layers with no fancy yet computationally-heavy transforms, and the whole neural net is no larger than 20 layers in total, making it relatively efficient to train. Different from most “deep” learning methods that make heavy use of computation, the SE3-Pose-Net is simple both to train and to evaluate.

Another notable detail about the paper is that it uses a pre-trained SE3-Pose-Net for actual control of the robotics. Given a desired joint position, it uses a gradient-based approach to calculate the action achieved to move from the current to final state. It then uses an iterative method to update its calculations. As claimed by the author, even when the neural net is trained with minimal supervision, it still performs only slightly worse than other approaches with full knowledge about the robot. As a limitation though, the experiment on the real robot only applies to the first 4 joints, and it’s yet unknown what would happen if we use all joints.

This paper is using the “modern” deep-learning based approach to solve a traditional robotics problem, and might shed light on some unsolved problem in reinforcement learning / robot manipulation. It’s more like a “fusion” between deep learning and robotics and reading it does require some background in basic robotics. On the other hand, in its effort to minimize calculation, it does not spend a lot of effort in using complicated architectures. Therefore, it’s more suitable for the robotics/reinforcement learning community.

### Strengths and Weaknesses of Deep Learning Models for Face Recognition Against Image Degradations

**Authors**: Klemen Grm, Vitomir Struc, Anais Artiges, Matthieu Caron, Hazim Kemal Ekenel
**Arxiv Link**: https://arxiv.org/pdf/1710.01494.pdf
**Published Date**: October 4th, 2017. 

In this paper, the authors performed tested changes in image processing network performance across a wide survey of image degradation methods. In particular, they tested AlexNet, GoogLeNet, VGG-Face and SqueezeNet on 6 different methods: gaussian blur, gaussian noise, salt-pepper noise, contrast, brightness and JPG compression. 
	Their methodology was as follows. The networks were trained on the VGG-Face dataset in order to perform facial recognition. Each subject had 1000 copies in the dataset. Afterward, the models were tested on faces from the Labeled Faces in the Wild dataset (LFW). These pictures had the image degradations performed on them. In addition to these image covariates, the team also performed comparisons in which they modified the models. They did this in 2 ways: changing the number of image samples fed to the model and comparing performance depending on color or grayscale images were used. 
	Overall, the team found results that are quite plausible and realistic. The performance of all models dropped reliably when the image degradation became more pronounced. However, depending on which image covariate was being analyzed, different models would respond more sensitively to the changes, creating a noticeable gap in performance between models at certain thresholds. In addition, for some image covariates, the drop in performance is more dramatic and then plateaus, compared to others in which it seems to steadily decline. The researchers found that the model covariates did not seem to affect performance in a noticeable way. 
	This research, while not presenting a new model or framework for image processing, plays an important part in the overall science of deep learning research. In order to build a rigorous framework for why deep learning has its successes and pitfalls, we need to gather data like that of this paper. It is only from this can we ask why certain techniques are more vulnerable or robust to noise and perhaps find underlying truths in deep learning. 

### Learning to Segment Human by Watching YouTube

**Authors**: Xiaodan Liang, Yunchao Wei, Liang Lin, Yunpeng Chen, Xiaohui Shen, Jianchao Yang, Shuicheng Yan
**Arxiv Link**: https://arxiv.org/pdf/1710.01457.pdf
**Published Date**: Oct. 4 2017

**Summary**:  The goal of this paper is to train a CNN model that generates a segmentation mask of the human in an input image. This paper provides supervision to this model by first generating a crude segmentation mask for frames of Youtube videos with a pretrained “imperfect” human detector. Unsupervised methods are used to decompose the videos into superpixels and supervoxels (superpixel over time). The crude mask from the pretrained detector is used along with the segmentation mask predicted by the currently trained CNN to assign unary energies to superpixels, so that graph optimization techniques could be used to generate a refined segmentation mask from the noisy input segmentation masks. The CNN is then uses the refined segmentation mask for supervision, and improvement in the CNN model will eventually make unary energies more accurate, which results in improvement in the supervision mask, and etc.

**Notable Details**:  This paper’s very weakly supervised method, which trains on Youtube data and uses a pretrained human detector for mask initialization, is comparable to or outperforms many previously fully supervised methods that trains on pixel-level labels of segmentation. It is also notable that with some supervision, the model’s performance improves significantly. i.e. training with labeled segmentation masks as semi-supervision performs better than using the human detector (which was trained on annotated bounding boxes).

A notable detail is that this model is a binary model and only segments into two classes (i.e. human and non-human). This may explain in part the strong performance of the method. It might be interesting to see the authors extend their technique to create a multi-class method.

The authors mentioned that their model takes two days to train and 2 seconds to test. Since this method is iterative and may take somewhere around 10 alternations between graph and CNN optimization, it would have been informative for the author to compare the runtime performance of previous models with their proposed method. 


### Dense RGB-D semantic mapping with Pixel-Voxel neural network


**Authors**: Cheng Zhao, Li Sun, Pulak Purkait, and Rustam Stolkin
**Arxiv Link**: https://arxiv.org/pdf/1710.00132.pdf
**Published Date**: October 4th, 2017. 

Previous approaches to 3D semantic segmentation typically use models to construct the 3D map and perform scene understanding separately and then afterwards fusing the results with equal weights.
The goal of this paper is to use both RGB images and point cloud data together to produce a 3D semantic mapping. The authors propose a Pixel-Voxel network which will take both types of input and allows for utilizing the global context information from PixelNet in conjunction with the local shape information from VoxelNet. Combined with a softmax weighted fusion stack that can learn how much confidence to have either model, this model achieves state-of-the-art results.
The softmax weighted fusion stack is a key part of the success of the model and is also highly flexible. It can take any number of input models to be fused and can be used into any network for fusion style learning, always learning how much contribution to allocate to each model for every combination of situation and categories.
The runtime performance for this model is 5-6 Hz when using QuadHD data, which is almost enough to satisfy real time dense 3D semantic mapping. This can be boosted to up to 12 Hz by using half scale data. 


### Early Turn-taking Prediction with Spiking Neural Networks for Human Robot Collaboration

**Authors**: Tian Zhou and Juan P. Wachs
**Arxiv Link**: https://arxiv.org/pdf/1709.09276.pdf
**Published Date**: September 28th, 2017. 

Communication is a requirement for efficient teamwork. Often times when humans coordinate every human is aware of the others’  intentions and context of the others and will use this information when deciding when performing tasks. This paper proposes a model called the Cognitive Turn-taking Model (CTTM) to try and guess the human’s intentions while he is making them and make accurate predictions before the gesture is complete. In this manner we can build much more responsive robots as they can start their actions earlier.
In their CCTM, they incorporate Spiking Neural Networks (SNN). To try to better simulate how human brains work. SNNs are a more realistic simulation of neurons than traditional neural networks. Unlike traditional neural nets, where each neuron in a given activation layer activates at the same time, SNNs activate whenever their “membrane potential” (essentially the activation threshold) reaches a certain potential. Additionally, unlike traditional neural networks where the activations are modeled as discrete time steps with well defined functions, the activations of SNNs are typically described by differential equations (typically 2nd order diff eqs.).
In their experiments they took a human doing a mock surgery and at various points the human did turn-taking cues (i.e., pass the scalpel). For testing, they used the F1 score at various stages of the human’s motions. Performs better than previous ways of modeling this (dynamic time warping, ishii), but still performs worse than humans.

###  Detecting Adversarial Attacks on Neural Network Policies with Visual Foresight
**Authors**:  Yen-Chen Lin, Ming-Yu Liu, Min Sun, Jia-Bin Huang

**Arxiv Link**: https://arxiv.org/abs/1710.00814v1 
**Published Date**: Mon, 2 Oct 2017

Deep reinforcement learning is an effective way to learn complex policies in robotics, video games and many other tasks. However, just like modern deep neural networks, these algorithms are susceptible to adversarial attacks. That is to say: malicious actors can perturb inputs to the algorithm in order to alter the outcome, and can do so intelligently. In deep learning, there are several defenses against this sort of attack (such as model distillation). However, there do not exist model agnostic defenses in deep reinforcement learning until now. Lin et al introduce a method to detect adversarial attacks in environments whose visual dynamics are well-known and predictable. 
To do so, they introduce a “visual foresight” module (a one-step look ahead image-sequence prediction module), whose job is to predict the next reasonable observation, given state x_{1:t-1} and a_{1:t-1}. The sequence of states up to t-1, concatenated with the visual foresight frame is used to predict an action. At each step, this visual foresight module action is compared with the true action. If the difference is above some threshold, then we can be reasonably certain that an adversarial attack has taken place, and we are better off taking the action predicted by the visual foresight module, whose input has not been perturbed, and thus would allegedly act more reasonably than the true agent. 
This defense is both model and attack agnostic, so can be used against any attack, from the “Fast Gradient Sign Method” to “Carlini Wagner.” The visual foresight module consists of an encoder, an actio-conditional transformation, and a decoder. The main benefit to this method is that instead of rejecting adversarially perturbed inputs, it can actually continue to act despite their presence. However, the method does require accurate modeling of the visual dynamics of the environment. Thus it works best on simple visual data, wherein we can easily predict the next frame given the current.

### Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory

**Authors**: Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, Bing Liu
**Arxiv Link**: https://arxiv.org/pdf/1710.02277.pdf
**Published Date**: October 3th, 2017. 

Current chatbots don’t consider emotion in their responses. This paper is about an emotional chatting machine (ECM) that tries to make this happen by keeping track of emotion embeddings and feeding it into the decoder along with the encoder outputs. They get on average a .4/2 rating emotionally when tested manually when compared to the baseline that gets only .152/2. Since there is no baseline currently for this task, they used a generic seq2seq model that is the current state-of-the-art in chat generation. In terms of content, they matched the baseline which proved that they could keep the previous state of the art chat results standard while adding emotion to it.

Since there was no large-scale emotionally labeled dataset available, they trained a bi-lstm classifier on the NLPCC dataset and then ran it on millions of Weibo social network chats. By doing this, they got a noisy but annotated dataset that they could use for the task.

Their model is an GRU encoder-decoder system with attention. To incorporate emotion, they keep emotion embeddings that they pass through to the decoder with the encoder outputs and the encoder attention.

## Arxiv Summaries 10/16

###How Much Chemistry Does a Deep Neural Network Need to Know to Make Accurate Predictions

**Authors**: Garrett B. Goh, Charles Siegel, Abhinav Vishnu, Nathan O. Hodas, Nathan Baker

**Arxiv Link**: https://arxiv.org/abs/1710.02238

**Arxiv Published Date**: 5 Oct 2017

**Executive Summary**: Goh et al. described “Chemception,” a CNN based system that predicts chemical properties from molecular drawings. This work improves on the system by adding basic chemical information like atomic number, valence electrons, and partial charge. This work also describes experiments in which the authors tested Chemception with reduced or augmented inputs, uncovering information on learning patterns in predicting toxicity/activity and compares them to current knowledge in chemistry. The authors found that the additional basic chemical information improved Chemception’s predictions and that the system was learning to perform tasks in line with the current established practices.

**Notable Details**: Chemistry data sets are expensive and time-consuming to build, leading to the idea that providing chemistry basics to Chemception would allow it to spend its learning capacity on more sophisticated representations. In toxicity and activity prediction, Chemception does not outperform other systems that require substantial pre-requisite chemical knowledge, but it does very well considering it doesn’t need anything beyond basic chemistry concepts. In free energy of solvation, Chemception achieves state of the art performance and comes very close to the gold standard of accuracy. 
Suitable Readers: Those who are interested in chemistry, toxicology, and applied computer vision.

### Face Sketch Matching via Coupled Deep Transform Learning

**Authors**: Shruti Nagpal, Maneet Singh, Richa Singh, Mayank Vatsa, Afzel Noore, Angshul Majumdar

**Arxiv Link**: https://arxiv.org/abs/1710.02914

**Arxiv Published Date**: 9 Oct 2017


**Executive Summary**: Nagal et al. introduced the first deep transform learning framework for extracting domain invariant representation of inputs. Their learning framework, DeepTransformer consists of two layers. The first layer learns a low-level feature representation of the input, these low-level features are then passed onto the second layer that learns a high-level feature representation and a mapping from the input domain to a desire domain. They also presented a second variant of the framework that learns two mappings from and to the desired domain instead. The authors used neural-network based architecture for the layers in their framework, but hand-crafted feature extractors and classifiers can also be used in place of the neural-network based architecture.

They evaluated their approach on sketch-to-digital-photo matching and sketch-to-sketch matching for law enforcement applications. Their model outperforms state-of-the-art sketch matching algorithms and a commercial face recognition system.

**Notable Details**: DeepTransformer can use existing (both neural-based or hand-crafted) feature extractors and classifiers in place their proposed learning networks. The overall performance on both sketch-to-sketch and sketch-to-image matching tasks depends on the feature extractor, but using their learning framework outperforms other learning frameworks on the same learning tasks using the same feature extractor. 

They also presented the first publicly available dataset containing multiple age-separated digital photos for a given sketch image. They train models on this dataset that contains matching photos that were taken from younger, same and older image than the time the sketch was produced. Unsuprisingly, their model achieves the best performance when sketches produced at the same age as the digital photos of the same subjects.


**Suitable Readers**: Machine Learning Researchers interested in Domain Adaptation 

### Efficient K-Shot Learning w/ Regularized Deep Networks

**Authors**: Donghyun Yoo, Haoqi Fan, Vishnu Naresh Boddeti, Kris M. Kitani

**Arxiv Link**: https://arxiv.org/abs/1710.02277

**Arxiv Published Date**: 6 Oct 2017

**Executive Summary**: Fine-tuning pretrained neural networks for small datasets is a common practice used today. However, the complexity of networks make it easy to overfit to the given data. One indication of overfitting is the occurrence of correlated filter responses from neurons in the network. The network proposed a way to group similar neurons together to act as regularization for the network.

**Notable Details**:  As a form of regularization, the paper proposes to group correlated neurons together and apply a single backpropagation gradient through the entire group of neurons rather than having individual gradients per neuron. The paper trains the network to group neurons such that the difference between the activation response of neurons within a group are minimized and the resulting outputs of neurons of a group are orthogonal to other groups in the layer.

To determine the optimal number of groups per layer in the network, an RL agent is used to search this space over an LSTM network that outputs the proposed groups per layer. The implementation of this RL agent on an LSTM network is almost identical to Zoph and Le’s architecture search.

**Suitable Readers**: Those interested in k-shot learning and a unique method of regularization.

### Natural-Gradient Stochastic Variational Inference for Non-Conjugate Structured Variational Autoencoder

**Authors**: Wu Lin,  Mohammad Emtiyaz Khan, Nicolas Hubacher, Didrik Nielsen 

**Arxiv Link**: https://arxiv.org/pdf/1603.06277.pdf

**Arxiv Published Date**: 7 Jul 2017

**Executive Summary**: This paper proposed a new method for amortized inference in graphical models that contain deep generative models. The method generalizes existing approaches to a larger class of models where the graphical model can contain non-conjugate components. Their main contribution was the proposal of structured recognition models that preserve the correlations between all local variables. For this general class of models, they derived a scalable inference method that employs natural-gradient updates and can be implemented by reusing existing software for graphical models and deep models.

**Notable Details**:  In this paper, they proposed a method that generalizes the method of previous work to a larger class of models where the graphical model can contain non-conjugate components. Their main contribution is the proposal of structured recognition models that preserve the correlations between all local variables. For this general class of model, they also derived a scalable inference method that employs natural-gradient updates and can be implemented by reusing existing software for graphical models and deep models.

**Suitable Readers**: Those who want to know further optimization of variational inference beyond the reading material.

## Arxiv Summaries Week 10/23

### Fishing for Clickbaits in Social Images and Texts with Linguistically-Infused Neural Network Models

**Authors**: Maria Glenski, Ellyn Ayton, Dustin Arendt, Svitlana Volkova

**Arxiv Link**: https://arxiv.org/abs/1710.06390

**Arxiv Published Date**: 17 Oct 2017

**Executive Summary**: Glenski et al. describes two methods in analyzing the amount of clickbait in link-based tweets on Twitter. This work was submitted for the Clickbait Challenge 2017. This work uses a CNN approach as well as a LSTM approach, both of which use the tweet and information at the link destination as features. Of the two, the LSTM performed marginally worse(within 0.01 MSE), but was more computationally preferred for the challenge. Besides text based input features, pictures from the tweet and the linked page were also used as features. This work borrowed from the Faster R-CNN architecture, but also used Inception Resnet for feature extraction. They found certain types of objects in the image lead to a preference in classification as clickbait or not. Images containing food and dishes were more likely to be clickbait, while vehicles in the image would decrease the clickbait score. In the end, the result was 5th out of 13 in the competition.

**Notable Details**: While using the CNN architecture would likely have yielded better end results than the LSTM, the restriction of the competition led to the compromise of slightly lower score for better computation performance. The use of object recognition in the images allows there to be a somewhat human interpretable conclusion on the decision making process of the model. The resulting observations on certain objects affecting the clickbait score is possibly due to data selection and the nature of labelling certain articles and tweets as clickbait or not.

**Suitable Readers**: Those who are interested in Language Analysis, Social Media, Image Recognition.

### Map-based Multi-Policy Reinforcement Learning: Enhancing Adaptability of Robots by Deep Reinforcement Learning

**Arxiv Link**: https://arxiv.org/pdf/1710.06117.pdf

**Authors**: Ayaka Kume, Eiichi Matsumoto, Kuniyuki Takahashi, Wilson Ko and Jethro Tan

**Published on Arxiv**: Oct 17

**Summary**: A desirable quality in robots is having the ability of adapt to new situations without having to redo costly training.  New situations could be injuries in the robot or environmental obstacles.  Instead of using only one policy to deal with all situations, this paper details MMPRL, which searches and stores multiple policies.  The robots have no prior knowledge of what situations could arise.  Policies stored in the map are organized based on user-designed behavioral descriptors.  Bayesian optimization is used to select the best policy during test-time.

**Notable Details**: The behavioral descriptors by which the stored policies are organized are human-designed and are specific to each task and robot model.  Also, the authors chose not to have the robot detect when adaptation is appropriate, and simply have the robot begin adapting at the start of test time.

**Suitable readers**: Those interested in robustness and adaptation in reinforcement learning.

### Recent Advances in Zero-shot Recognition

**Authors**: Yanwei Fu, Tao Xiang, Yu-Gang Jiang, Xiangyang Xue, Leonid Sigal, and Shaogang Gong

**Arxiv Link**: https://arxiv.org/abs/1710.04837

**Arxiv Publish Date**: Fri, 13 Oct 2017

**Executive Summary**: Fu et al. is a literature summary and review on the topic of zero-shot learning, training a recognition model for a new class without any training data of this class. A similar concept, few-shot learning, or learning with only a few labels of an unseen class, is also explored. To achieve zero-shot learning, we draw inspiration from the fact that human can learn novel concepts with few labeled data via powerful semantic featurization of these new concepts. In machine learning, this can be modeled through semantic attributes. The paper gives examples of four popular types of such semantic attributes. After this, we can train models that convert data points into an embedding space in order to correspond seen and unseen classes based on their low-level features. The paper lists a few options for this embedding model as well. With embedding model translating data into embedding space, recognition models can then be trained in this embedding space. Finally, the paper lists some popular datasets for zero-shot learning and recognition, as well as future research paths for this concept.

**Notable Details**: Today’s machine learning community is making attempts to steer away from “big data” craze. Granted, supervised learning performs well with massive amounts of training data, however, it is more economical and realistic for us to develop methods that efficiently utilizes labeled data. This is a statistic question as well as a machine learning question. This paper is a good starting point for researchers to get started with the concept of few-shot learning, especially for fields such as robotics where labeled data can be expensive or time consuming to obtain.

**Suitable Readers**: People who are interested in image or video object recognition, and semantic understanding in natural language processing. Researchers in fields where collecting labeled data is expensive. Statisticians interested in latent space modeling of visual or natural language data.

### Unsupervised Object Discovery and Segmentation of RGBD-Images

**Authors**: Johan Ekekrantz∗ , Nils Bore∗ , Rares Ambrus∗ , John Folkesson∗ and Patric Jensfelt∗

**Arxiv link**: https://arxiv.org/pdf/1710.06929.pdf 

**Arxiv date**: Fri, 17 Oct 2017

**Executive Summary**: Sensor noise is modeled directly from data instead of hand-tuning. Using probabilistic inference, we are able to segment in previously challenging scenarios. The previous most common solution to object detection is the supporting plane assumption, which says that most objects are found on flat supporting surfaces. The new method presented in this paper is using the Statistical Inlier Estimation (SIE) algorithm which is modified and applied to change detection and probabilistic image edge detection. The algorithm is used to model noise distribution and perform probabilistic occlusion detection between RGBD pairs, reducing time spent on hand-tuning parameters. MLE is used to estimate which regions change in the RGBD pairs. A classic supervised learning detector is used to detect people, and the moving segments detector detects moving segments. Each complements the other, and the result is we can model the stationary rigid body objects in the environment.

**Notable Details**: Separating the dynamic and static objects in an environment is an important problem in computer vision and robotics. A typical use case is a surveillance camera with automatic detection. Otherwise, this system can also help improve the quality of SLAM systems.

**Suitable Readers**: Roboticists who are interested in object discovery and attention, such as for an autonomous discovery system.

### Natural-Gradient Stochastic Variational Inference for Non-Conjugate      Structured Variational Autoencoder

**Authors**: Wu Lin,  Mohammad Emtiyaz Khan, Nicolas Hubacher, Didrik Nielsen 

**Arxiv Link**: https://arxiv.org/pdf/1603.06277.pdf

**Arxiv Published Date**: 7 Jul 2017

**Executive Summary**: This paper proposed a new method for amortized inference in graphical models that contain deep generative models. The method generalizes existing approaches to a larger class of models where the graphical model can contain non-conjugate components. Their main contribution was the proposal of structured recognition models that preserve the correlations between all local variables. For this general class of models, they derived a scalable inference method that employs natural-gradient updates and can be implemented by reusing existing software for graphical models and deep models.

**Notable Details**:  In this paper, they proposed a method that generalizes the method of previous work to a larger class of models where the graphical model can contain non-conjugate components. Their main contribution is the proposal of structured recognition models that preserve the correlations between all local variables. For this general class of model, they also derived a scalable inference method that employs natural-gradient updates and can be implemented by reusing existing software for graphical models and deep models.

**Suitable Readers**: Those who want to know further optimization of variational inference beyond the reading material.

## Arxiv Summaries Week 10/30

### Unified Backpropagation for Multi-Objective Learning

**Authors**: Arash Shahriari

**Arxiv Link**: https://arxiv.org/pdf/1710.07438.pdf

**Published on Arxiv**: October 20th, 2017

**Executive Summary**: A common practice in deep learning (especially in classification) is to train a network on multiple different loss functions and then build a stronger classifier via some ensembling method. For example, one might use LDA, SVM, and softmax loss functions to train three different networks (with these loss functions as one of their only distinctions) separately, and then ensemble the results. However, this paper introduces this notion of unified backpropagation; where one can use a specific objective combining multiple different loss functions and then compute the gradients to do backpropagation on that specific objective. This provides a few key advantages. The clear advantage is that all of the objective functions are optimized together, and each is contributed by how much of an effect it has (which is determined by BPA). Further, instead of training multiple different networks using different loss functions, one network can be trained using this scheme, which is less computationally expensive. 

**Notable Details**: The algorithm computes for each loss function the basic probability assignment (BPA), an assignment of probabilities to each label found by building a confusion matrix for each loss function, using the training set. This is done by calculating from the training set’s confusion matrix quantities known as the recall and precision for each class/label pair. Using the Dempster-Schafer rule of combination, each precision/recall is translated into a basic probability assignment for each class. Then, a vector is constructed where each element corresponds to the 2-norm of the BPA for each loss function, which corresponds to the "effect" this loss function has on the result. During backpropagation, each gradient is then weighted by its associated element in this vector, with a learning rate hyperparameter used for gradients of all loss functions. To evaluate the model, this “unified” approach using loss functions SVM, softmax, and LDA was trained on the MNIST, CIFAR-10, CIFAR-100, and SVHN datasets. The training error and test error were then compared to both single-objective learning (using only one loss function) and multi-objective learning (using multiple loss functions and ensembling). In the single-objective scenario, the unified approach significantly outperformed all single-objective cases on each dataset for test errors and most training errors, particularly for LDA where the testing improvement was by 0.04%, 0.44%, 1.17%, and 0.71% for MNIST, CIFAR-10, CIFAR-100, and SVHN respectively. For multi-objective learning, the test errors were mostly lower for the unified approach (except for the CIFAR-100 dataset) and the training errors were lower for the majority of datasets/combinations. Particularly, against the baseline, the unified approach had the biggest testing improvement on the CIFAR-10 dataset with a 1.95% improvement, and it had worse performance on the CIFAR-100 dataset with a 0.19% higher test error than the baseline. This paper is particularly useful for multi-class classification applications in deep learning when multiple different objective functions are considered.

### Lip2AudSpec: Speech Reconstruction from Silent Lip Movements Video

**Authors**:  Hassan Akbari, Himani Arora, Liangliang Cao, Nima Mesgarani

**Arxiv Link**:  https://arxiv.org/pdf/1710.09798.pdf

**Published on Arxiv**: October 26, 2017

**Executive Summary**:  When it comes to the task of lip reading, previous works generally fall into two categories: those that aim to transcribe videos of speakers to text, and those that aim to reconstruct speech from the videos of speakers. Works from the second category oftentimes face a problem in which the models do not capture pitch information, resulting in unintelligible speech.  As as result, Akbari et al. propose Lip2AudSpec, a deep neural network that aims to reconstruct easily understandable speech (with pitch information) from silent videos of lip movements. This deep neural network consists of CNN, LSTM, and fully connected layers, followed by the decoder portion of an autoencoder network. The input of this network is a series of frames from the silent video, and the output is a spectrogram representing the reconstructed speech. Lip2AudSpec reconstructs spectrograms with 98% correlation with the original spectrogram, and has also shown to improve the realism of the reconstructed speech compared to other methods. 

**Notable Details**: The input of the neural network is a series of frames from the silent video, which is passed into a 7-layer 3D convolutional neural network. The output of the 3D CNN is then provided as input to an LSTM with 512 units to model time dependencies, and the resulting output is then flattened and passed through a dense layer, and then into the decoder part of a pre-trained autoencoder. The activation function for the 3D CNN is the LeakyReLU, while the LSTM and dense layers use the ELU activation function. 
The autoencoder is trained separately, and consists of an input layer of size 128, a dense layer of size 64, and a final bottleneck of size 32, before increasing back to 64 and then 128. To train the autoencoder, the audio waveforms were downsampled to 8kHz and converted to a spectrogram representation with 128 frequency bins. Gaussian noise was also added to the output of the bottleneck during training and was found to create a more robust representation. After training, the decoder portion of the autoencoder was used as the final portion of the deep neural network, effectively transforming the output of the LSTM to a spectrogram representation. The dataset used to train the network was the GRID audio-visual corpus, and the loss function used was a combination of the mean squared error and correlation. Several experiments were ran, including a comparison between reconstructed speech from Lip2AudSpec and a previous model proposed by Ephrat et al., Vid2Speech. Reconstructed samples from the two methods were provided in an Amazon Turk survey, and Lip2AudSpec had a higher accuracy and natural sound rating than Vid2Speech as reported by the users. Notably, users who listened to the samples from Lip2AudSpec were able to correctly guess the gender of the speaker up to 92 percent of the time, which demonstrates a significant improvement in pitch information retainment. Overall, Lip2AudSpec demonstrates that networks involving a deep autoencoder are promising ways to reconstruct speech. 

### How to Fool Radiologists with Generative Adversarial Networks? A Visual Turing Test for Lung Cancer Diagnosis 

**Authors**: Maria J. M. Chuquicusma, Sarfaraz Hussein, Jeremy Burt, Ulas Bagci

**Date released**: 26 Oct 2017

**Arxiv link**: https://arxiv.org/abs/1710.09762

**Executive summary**:  This paper is an application of deep learning to an important real world setting: the improvement of computer-aided diagnosis in lung cancer (specifically, the categorization of lung nodules as benign or malignant). The aim of the paper is twofold: to to learn highly discriminating features for benign, malignant, real, and generated nodules, and to improve the training of radiologists through generating large databases of cases. The authors use a deep-convolutional general adversarial network (DC-GAN) to learn how to do this, based on CT images of these nodules from the LIDC-IDRI dataset. This delivers a system in which images of benign and malignant lung nodules can be generated, for training purposes, and categorized, for diagnostic purposes. The generator function is proven useful in a visual Turing test, in which two senior radiologists are mostly unable to disambiguate real and generated lung nodules. 

**Notable details**:  The LIDC-IDRI dataset consists of diagnostic and lung cancer screening CT scans with marked-up annotated lesions, taken over 1018 cases. Their model architecture is simple: a generator function with three convolutional layers, which learns a distribution $p_z$ from random noise input that eventually allows it to sample a realistic image, and a discriminator with two convolutional and one fully connected layers, which learns to categorize generated images as real or generated, and benign or malignant. 

### One pixel attack for fooling deep neural networks 

**Authors**: Jiawei Su, Danilo Vasconcellos Vargas, Sakurai Kouichi 

**Date released**: 24 Oct 2017

**Arxiv link**: https://arxiv.org/pdf/1710.08864.pdf

**Executive Summary**: In the domain of image recognition, DNN-based approach has overcome traditional image classification techniques and achieved human-competitive results. However, several recent studies have revealed that by injecting artificial perturbations on natural images to generate samples called “adversarial images”, one can easily make DNN misclassify images that it previously classify with high confidence. A main way of creating adversarial images is adding a tiny amount of well-tuned additive perturbation to a correctly classified natural image that is expected to be imperceptible to human eyes. However, most of these methods do not consider the case that only a limited amount of adversarial information is provided. To resolve the problem, the author proposed a novel method for optically calculating extremely small adversarial perturbation (few-pixels attack), based on differential evolution. It requires less adversarial information and works with a broader classes of DNN models. With modification on one pixel, it is able to convert 73.8% of testing images to adversarial images, 82.0% with three pixels, and 87.3% on five pixels. The attack method also provided new perspective on understanding how DNN interpret features in high input dimension.

**Notable Details**: The differential evolution they proposed does not use the gradient information for optimizing therefore does not require for the optimization problem to be differentiable. The information of each pixel is packed in a 5 elements DNA structure (x, y, RGB channel value). DE optimizes problems by keeping improving the quality of a candidate population according to a given fitness function through a procedure which can be summarized as: in each iteration, a new group of pixels(children) are examined based on the pixels(parents) examined from last iteration. Then the new group is only compared to the direct parent of every pixel in order to maintain diversity. 
The Structure of the target DNN consists of 5 conv layers followed by 0.3 dropout, 4 more conv layers, an fc layer and a softmax output layer. 

### Learning compressed representations of blood samples time series with missing data

**Authors**: Filippo Maria Bianchi, Karl Øyvind Mikalsen and Robert Jenssen 

**Date released**: 20 Oct 2017

**Arxiv link**: https://arxiv.org/pdf/1710.07547.pdf

**Executive Summary**:  An important tool in assessing the general health of medical patients are clinical measurements taken over a duration of time, which can be viewed as a multivariate time series (MTS). A major goal in deep learning has been to create efficient compressed representations of these MTS. On first glance, a standard autoencoder should work well. However, these measurements can be often be infrequent and therefore filled with missing measurements. Bianchi et al propose a new framework, combining deep kernalized autoencoders with Time Series Cluster Kernels (TSCKs), in order to create accurate low dimensional representations of the clinical MTS data. This custom deep kernelized autoencoder(dkAE) is compared to a regular autoencoder, with various imputation schemas, and is shown to perform significantly better, when compared on reconstruction error and kNN classification. 

**Notable Details**: Expanding on the model, we note that in practice autoencoders are trained on a loss function which is the reconstruction loss of the original sample data. However, deep kernelized autoencoders work by combining two loss functions with weight hyperparameter: the original autoencoder loss function, as well as the Frobenius Norm of the inner product matrix of latent samples with the kernel matrix, given as a prior. As noted, the kernel matrix is a Time Series Cluster Kernel, which is formed via joining the clustering results of many Gaussian Mixture Models (GMMs). TSCK’s have been shown to exploit missing patterns to compute similarity, rather than impute missing values.

The data that this custom dkAE was tested on were blood samples collected from patients within 20 days after gastrointestinal surgery. The data had 10 features, with a label of whether the patient had developed surgical infections. The total dataset consisted of 883 samples, 232 with infections. Three metrics were used to compare seven models: MSE of the reconstruction, F1 score of kNN prediction (k=3), and AUC of the kNN prediction. The seven models were as follows:  the missing data was imputed in three ways, and run on a regular autoencoder, and the custom dkAE (Six models). The data was imputed as follows: zero for missing data, mean for missing data, carry-forward for missing data. The final model was created using the original input space with TCK similarity. When compared across all 7 models, dkAE with zero imputed data performs the best by far when compared on mean AUC of 81.3% and F1 score of .748. As a whole, the custom dkAEs outperform regular autoencoders with higher F1 scores and AUC, and lower mean squared reconstruction loss, which is surprising as regular autoencoders loss functions are built only on reconstruction error. As a whole, by using a TCK kernel matrix in the dkAE, data is better embedded in lower dimensions, and classification results improve as a whole. 

## Arxiv Summaries Week 11/06

### Transfer Learning to Learn with Multitask Neural Model Search

**Authors**: Catherine Wong, Andrea Gesmundo

**Arxiv Link**: https://arxiv.org/abs/1710.10776

**Published on Arxiv**: October, 30 2017

**Executive Summary**: Neural Architecture Search (NAS) utilizes a controller RNN and reinforcement learning in order to produce a child network with trained hyperparameters. This technique has proven to be very effective, producing novel architectures that in some cases surpass the performance of human designed models, however the computational cost in constructing these networks is immense. One conceivable way to reduce the computation time of NAS is to leverage transfer learning for similar tasks. Wong and Gesmundo propose Multitask Neural Model Search (MNMS), a formulation aimed at effectively applying transfer learning to NAS. MNMS relies on three components: learned task representation and task conditioning, off-policy training using multitask replay, per-task baseline and reward distribution normalization. Wong and Gesmundo apply MNMS to 2 pairs of NLP tasks and demonstrate that training the network on the first pair of tasks followed by training on the second pair of tasks indeed yields faster training and better final accuracies than training on the second task from a randomly initialized network.

**Notable Details**:  Previous works have attempted to leverage transfer learning to NAS; however, none have produced a model that, without some level of human design intervention, can demonstrate successful transfer learning. MNMS attempts to produce a formulation that can allow for effective transfer learning in the domain of NAS. This method relies on three components:
Learned task representation and task conditioning
Off-policy training using multitask replay
Per-task baseline and reward distributions normalization
In learned task representation and task conditioning, tasks are randomly mapped to an embedding vector. When training on N tasks, a random task is sampled, the embedding vector for this task is then concatenated with every input fed to the controller RNN; this conditions the output of the RNN on the task. It was found that on-policy training reduced the ability for the controller to learn a differentiated model for each task. To remedy this, off-policy PPO is used to train the controller in which an actor controller generates sampled models and a critic controller trains on a replay bank of sampled models and rewards. Since each task has it’s own reward metrics they must be normalized so the amplitudes of each tasks gradients are similar. To do so, the gradients are scaled by advantages instead of rewards, since the advantage is zero meaned. A baseline (b(t)) for each task is set to an exponential moving average of rewards for each tasks. This advantage is then divided by this baseline yielding the normalized advantage (A’) which is used to scale the gradients. The full formulation is thus A’(a, t) = (R(a, t) - b(t))/b(t). Wong and Gesmundo applied MNMS to 2 pairs of NLP tasks where each pair contained tasks similar to a corresponding task in the other pair. The first pair was trained from a randomly initialized network and the second was trained with transfer learning. They then trained the second pair of tasks on a randomly initialized network and compared the accuracy of training with and without transfer learning. They had the controller find hyperparameters over a discrete set of values. The first two tasks were binary sentiment classification of the Stanford Sentiment Treebank (SST) dataset and binary Spanish language identification on a dataset consisting of each of the 5,000 highest frequency Wikipedia tokens in English, Spanish, German, and Japanese. The example label is a binary label denoting whether the token is SPanish or not. The next pair of tasks were binary sentiment classification on the IMBD Large Movie Review dataset and binary sentiment classification on the CorpusCine dataset, which consists of 3,878 Spanish movie reviews. These pairs of tasks were selected due to their potential to utilize transfer learning. The network trained with transfer learning both trained faster, and converged to a higher accuracy value that trained without transfer learning demonstrating that MNMS successfully enables transfer learning in the domain of NAS.

### PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION

**Authors**: T. Karras, T. Aila, S. Laine, J. Lehtinen

**Arxiv Link**: https://arxiv.org/pdf/1710.10196.pdf

**Publised on Arxiv**: November 3, 2017 

**Executive Summary**: Current pitfalls of the widely popular GANs include that they are often restricted to low-resolution images, do not produce a lot of variation in the samples, and training is unstable. Generating higher resolution images is difficult for a couple reasons. First, at higher resolution the generated samples are easier to tell apart from the original samples. Second, due to memory constraints, training is restricted to small mini-batches, which compromises stability. This paper suggests a novel approach: start with a simpler model, producing lower resolution images. Then, as training progresses, add new layers to capture more and more details. This allows the model to first learn overall structure, and then focus on finer details, which speeds up training, and stabilizes at higher resolution. Fade these new layers in smoothly to avoid sudden shocks to the already trained low res images.

**Notable Details**:  The model “fades” from lower resolution to higher resolution images by using a weighted average of the next level and the previous one, where the weights are α and 1-α respectively. This weight α grows linearly from 0 to 1. The results of this growing GAN model improved best published inception score on CIFAR10. Furthermore, they created a high quality version of the CELEBA dataset, with output resolution up to 1024 x 1024, something that hasn’t been feasible before. The lab intends to publish this dataset. I think anyone involved in vision machine learning, particularly in generation, and anyone who is excited about GANs should read this paper.

### DYNAMIC ROUTING BETWEEN CAPSULES

**Authors** S. Sabour, N. Frosst, G. E. Hinton

**Arxiv Link**:  https://arxiv.org/pdf/1710.09829.pdf

**Published on Arxiv**: October 26, 2017

**Executive Summary**: Traditional CNNs have struggled at tasks where there may be variations in the data such as rotations or translations. Furthermore, they tend to take a significant amount of data in order to generalize well. To combat some of these issues, the idea of capsules was proposed. Capsules represent small groups of neurons whose goal is to detect the likelihood of a feature as well as different parameters of the same feature. Capsules output a vector which represent the existence of a feature via its length and the properties of the feature via its orientation. These capsules are then structured in a parse tree such that each node corresponds to an active capsule. Active capsules in lower layers will choose a capsule in the layer above to be its parent in the tree. These active capsules make predictions about the instantiation parameters of higher level capsules. If sufficient predictions align, the higher level capsule becomes active. Through the use of a routing algorithm, weights (called coupling coefficients) are calculated which determine how the capsules will receive information from previous capsules. These coupling coefficients are then adjusted according to the level of agreement in a prediction between capsules through the use of routing. 

**Notable Details**: Using a simple 3 layer network with capsules provides similar results to deep CNNs on many tasks. Furthermore, it seems that capsule networks perform well on tasks which traditional CNNs may not have performed as well on, such as identifying overlapping digits as seen through increased performance on MNIST. Capsules networks also did well on CIFAR-10, but it seems that capsule networks in their current state are not great when there is a lot of background variation (MNIST has constant black backgrounds whereas CIFAR-10 does not). Overall, capsule networks are able to generalize well and withstand variations in the data with less training data.

**Suitable Readers**: Anyone with familiarity with CNNs and interested in an exciting new architecture structure for machine learning tasks.

### SYNTHESIZING ROBUST ADVERSARIAL EXAMPLES

**Authors**: Anish Athalye, Logan Engstrom, Andrew Ilyas, Kevin Kwok

**Arxiv Link**:  https://arxiv.org/pdf/1707.07397.pdf

**Published on Arxiv**: October 30, 2017

**Executive Summary**: Adversarial examples can currently be created relatively easily by perturbing an image by a particular amount specific to the classifier. However, prior work has shown that the adversarial example generated using this technique lose their adversarial nature once transformed or altered in any way, which often happens in the real world. The authors introduce a new approach, the Expectation Over Transformation (EOT) algorithm, that is able to produce adversarial examples that remain adversarial after being transformed under any transformation T, facilitating the creation of 3D objects that are misclassified from all angles. This implies that these methods are, in fact, a concrete threat to real-world systems. 

**Notable Details**: The authors’ new approach, Expectation Over Transformation (EOT), is different because rather than optimizing the log-likelihood of a single example, EOT uses a chosen distribution T of transformation functions t taking an input x generated by the adversary to the “true” input t(x) perceived by the classifier.  In practice, we can have T model random rotation, translation, or addition of noise.

### Interpretation of Neural Networks is Fragile

**Authors**: Amirata Ghorbani, Abubakar Abid, James Y. Zou

**Arxiv Link**: https://arxiv.org/abs/1710.10547

**Published on Arxiv**: October 29, 2017

**Executive Summary**: Researchers have proposed various methods for interpreting convolutional neural networks (CNN), but have not studied whether these methods are stable with respect to trivial perturbations. In this paper, the authors show that several widely-used interpretation methods are unstable with respect to even random perturbations. Moreover, they develop an approach for imperceptibly modifying a picture without altering the prediction, but drastically altering the interpretation. More succinctly, this paper shows that adversarial examples can be constructed for interpretation methods, in addition to the existing results on various supervised learning tasks.

**Notable Details**: For their introduced perturbation techniques, they perform experiments on Imagenet and CIFAR-10. The evaluation criteria are the spearman correlation between the original interpretation scores and the interpretation scores on the perturbed image, and the intersection of the top-k features chosen before and after the perturbation. With relatively small perturbations to the image, they are able to almost completely change the interpretation of 512 correctly classified Imagenet images. They also applied their approach to Koh and Liang’s recent ICML paper on influence functions, yielding similar results

**Suitable readers**: This work is of interest to researchers in two fields. First, those interested in adversarial examples now have an interesting new use case. Second, researchers working in interpretability have a new challenge to deal with. 


## Arxiv Summaries Week 11/13

### Non-Autoregressive Neural Machine Translation

**Authors**: Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, Richard Socher

**Arxiv Link**: https://arxiv.org/pdf/1711.02281.pdf

**Published on Arxiv**: November, 7 2017

**Executive Summary**: Current state-of-the-art neural machine translation models are autoregressive in nature. In other words, they condition on previous output to generate next output. This makes the decoding steps impossible to take advantage of parallelism and computationally expensive. The paper introduces a fertility measure to make each decoding step independent. This non-autoregressive approach parallelizes decoding steps while maintaining competitive BLEU scores.

**Notable Details**: The non-autoregressive model initialize decoding process using copied input source of the encoder. In order to learn how long the target sentence will be, the authors suggest copying each encoder input as a decoder input zero or more times, which they term, “fertility.” They used another one-layer neural network with softmax classifier to model this fertility. For decoding process, they used noisy parallel decoding method to draw samples from the fertility space and compute the best translation for each fertility sequence. Also, to address the nondeterminism in the training data by approximating the multimodal target distribution, they trained the non-autoregressive model using output from an autoregressive model. The training loss then seeks to minimize the sum of translation loss and fertility loss.

**Suitable readers**: Those interested in neural machine translation or seeking RNN alternatives for sequential input.  

### Composing Meta-Policies for Autonomous Driving Using Hierarchical Deep Reinforcement Learning

**Authors**: Richard Liaw , Sanjay Krishnan , Animesh Garg , Daniel Crankshaw, Joseph E. Gonzalez, Ken Goldberg

**Link**: https://arxiv.org/pdf/1711.01503.pdf

**Executive Summary**The problem statement of the paper “Composing Meta-Policies for Autonomous Driving Using Hierarchical Deep Reinforcement Learning” is that after a change in the task at hand relearning a policy from scratch using RL takes a long time, so it might be a good idea to use a meta-policy that chooses from a finite set of policies learned for other related tasks. The paper uses the example of a new car with a policy_new and an old car with a policy_old, noting that maybe a medium-aged car would work well with a policy that is a combination of policy_old and policy_new. The paper uses a GRU (Gated Recurrent Units)-RNN to model the meta-policy from a set of other policies. It uses a neural network to account for the shared-state space for different policies and RNN to account for noise due to partial observation. 

**Notable Details** Below are some interesting results:
* In the partially observed setting, meta policy learning converges in ~50 iterations while traditional RL converges ~500.
* Meta-policy learning handles noise very well. Traditional RL doesn’t achieve the same performance as meta-policy learning after 4x the training.
* Meta-policy converges well in terms of finding the subspace of policy which are optimal from a larger space of policies.

Below are some areas of future research it leaves unexplored:

* Meta-policy may not find the optimal policy, since it only finds the best composite policy.
* Meta-policy doesn’t seem to handle environments where rewards are not sparse as well.
* Meta-policy may fall short of the best policy initially, thus “falling short” of a lower bound.

### Structured Generative Adversarial Networks

**Authors**: Zhijie Deng, Hao Zhang, Xiaodan Liang, Luona Yang, Shizhen Xu, Jun Zhu, Eric P. Xing

**Link**: https://arxiv.org/pdf/1711.00889.pdf

**Published on arXiv**: Nov 2

**Executive Summary**: In the paper “Structured Generative Adversarial Networks”, the authors tackle the problem of generating conditional distribution given some designated semantics or structures. The authors propose the structured generative adversarial networks (SGANs) to achieve this goal in the semi-supervised setting, where there are only a small amount of training data are labeled.

**Notable Details**: SGAN considers two latent variables: y (the designated structure) and z (Gaussian noise) and the goal is to learn a function x=G(y, z) to approximate the conditional distribution p(x|y). The SGAN framework consists of 4 adversarial games:

* L_{xz} which learns the posterior distribution of z given x,
* L_{xy} which learns the joint distribution p(x, y) using the labeled data,
* Collaborative game R_y: use the generated (x, y) pairs as the true pairs, in order to improve controllability, also introduce the network C: x-> y in approximate the posterior 
* Collaborative game R_y:  enforce any other unstructured information that is not of our interest to be fully captured in z, without being entangled with y

The experimental results show that SGAN is a more controllable generator than the baseline algorithms, and it also establishes start-of-the-art results across multiple datasets when applied for semi-supervised image classification.

### DeepRain: ConvLSTM Network for Precipitation Prediction Using Multichannel Radar Data

**Author**: Seongchan Kim, Seungkyun Hong, Minsu Joh, Sa-kwang Song

**Link**: https://arxiv.org/pdf/1711.02316.pdf

**Published on arXiv**: Nov 7

**Executive Summary** The paper “DeepRain: ConvLSTM Network for Precipitation Prediction Using Multichannel Radar Data” by Kim et. al focuses on the problem of accurate rainfall prediction for multidimensional spatial data. This research area is relevant because rainfall forecasting has a pervasive impact on people’s socioeconomic behaviors. The current state of the art techniques have adapted deep learning and high performance computing techniques on spatiotemporal data (namely CNNs and RNNs) for weather related tasks. However, none of these tasks have optimized for multi-dimensional data input or approach the problem with convolutional cells LSTMs. Kim et. al seek to adopt the ConvLSTMs for three-dimensional and four-channel radar data, stack the ConvLSTM cells for performance enhancement, and confirm the ConvLSTM method is more effective for predicting rainfall than linear regression and the last paper on fully connected LSTMs (FC-LSTM).

The methodology of the ConvLSTM is outlined in section 4 of the paper with four equations for input, forget, and output for multi-dimensional data as an adaptation of the Shi et. al. work in 2015. The W matrix represents the learned weights, x represents the current input data, and c represents the cell state. They implement the ConvLSTM and the FC-LSTM on the data using NVIDIA Dual GPUs and a Tensor Flow framework, using the Adam and Gradient Descent Optimizer. 

Through their experiments (section 5), we see that the two-stacked ConvLSTM outperforms the one-stacked ConvLSTM and the FC-LSTM, reducing RMSE by 23% comparing to linear regression, and 21.8% compared to the FC-LSTM.

## Arxiv Summaries Week 11/20

### Breaking the Softmax Bottleneck: A High-Rank Language Model (Allen Guo)
**Authors**: Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen

**Arxiv Link**:  https://arxiv.org/pdf/1711.03953.pdf 

**Published on Arxiv**: Nov. 13, 2017

**Executive Summary**: In this paper, Yang et al. show that the standard approach to language modeling is essentially equivalent to computing a matrix factorization of the true data distribution. The authors then argue that the true data distribution in the real world is high-rank -- which means the standard approach may not be expressive enough to factorize it. The paper then proposes a new model called Mixture of Softmaxes (MoS) with significantly improved expressiveness: it is capable of factorizing an arbitrary-rank approximation of the true data distribution. Experiments show SOTA results for MoS on Penn Treebank and WikiText-2.

**Notable Details**: Language modeling involves predicting the next word in a sequence given its context. The current standard SOTA approach to language modeling involves 1) encoding the context using an RNN, 2) computing the dot product of the context with word embeddings to find the logits, and 3) feeding the logits into a softmax function. The result is a probability distribution over the next word. This is essentially equivalent to the matrix factorization
HW^T = A’
where H (N-by-d) is the matrix of context vectors, W (M-by-d) is the matrix of word embeddings, and A’ (N-by-M) is a matrix where Softmax(A’) is the true data distribution. Since it is the job of the model to learn H and W, the theoretical expressiveness of the model is limited by d, the dimensionality of the space of context vectors; if d is less than the rank of A’, the model cannot fully capture A’. Yang et al. argue that A’ is indeed high-rank; this argument is based on qualitative judgements about human language, as well as empirical results. The MoS model they propose is essentially the weighted average of K individual softmax models. This model is nonlinear, and is thus not affected by the expressiveness limitations that regular softmax models face.



### Hindsight policy gradients (Jeffrey O Zhang)
**Authors**: Paulo Rauber, Filipe Mutz, Juergen Schmidhuber

**Arxiv Link**:  https://arxiv.org/pdf/1711.06006.pdf

**Published on Arxiv**: Nov. 16, 2017

**Executive Summary**: Goal-conditioned policies for agents can be useful for developing hierarchical models or for generalization to unseen goals. However, incorporating goal in the problem can be different. This paper introduces a variant of policy gradient that let us use trajectories from another goal to improve our policy. The experiments show that this variant of policy gradient has better sample efficiency.

**Notable Details**: They start with the regular policy gradient equations and use importance sampling to come up with a goal-conditioned version of policy gradient. They introduce variants to the hindsight policy gradient formula by replacing the expectation on goals with the an assumption that the goal distribution is uniform for some experiments. The goal space is defined to be any goal such that r(s, g) != 0 for some state. Environment setup is simple and the network used is not very large (single hidden layer).



### Simple and Efficient Architecture Search for Convolutional Neural Networks (Melih Elibol)
**Authors**:  Thomas Elsken, Jan-Hendrik Metzen, Frank Hutter

**Arxiv Link**: https://arxiv.org/pdf/1711.04528.pdf

**Published on Arxiv**: November 13, 2017

**Executive Summary**: The paper presents a new way to perform architecture search for convolutional neural networks. They call their method Neural Architecture Search by Hillclimbing (NASH). This carries out the usual architecture iterative search procedure: At each iteration, select architectures to try based on previous results, then try those architectures. They select architectures by applying a set of alternative network morphisms to the current network, then perform short optimization runs of cosine annealing on the resulting child networks. It is worth noting that the weights from the parent network are inherited by the child network, thus decreasing training time. The procedure is repeated on the child architecture that performed best.

**Notable Details**: Most notably, this optimization algorithm seems like an evolutionary algorithm in disguise. At each step, a single candidate is mutated. The performance metric can be thought of as the fitness function used to evaluate “offspring.” One good question they address is whether the inheritance of weights leads to poor optima. They compared results of the networks with inherited weights to networks trained from scratch, and there appears to be no significant difference in error.


### Open-World Knowledge Graph Completion

**Authors**: Baoxu Shi, Tim Weninger, @ UND

**Arxiv Link**: https://arxiv.org/pdf/1711.03438.pdf 

**Published on Arxiv**: Nov. 9, 2017

**Executive Summary**: A knowledge graph is a triple <h, r, t> that signifies the relationship between two objects or entities h, t with relationship r. It follows that a popular task for KG’s is Knowledge Graph Completion where r is estimated. The authors of this paper propose a new method of open world (where entities can be unobserved - such as h, t) knowledge graph completion where a fully convolutional neural network is trained to to perform the following three steps: 1. Identify important parts of the text, 2. Embed the important parts 3. Entity resolution between similar object embeddings in order to estimate correct target entities. The purpose of this model is to try to learn implicit relationships between entities that may not be given or known. The results suggest this model is able to successfully infer target entities.

**Notable Details**: Typically, it is very difficult to model text in a knowledge graph that does not have an explicitly given structure with a neural network (such as having excess words that are not fundamentally important to a sentence structure like the, a, etc.). Although the usage of CNNs on language is not new, the way the model identifies relevant portions of text is particularly interesting. The model defines this method as Maximal World Relationship Weights where entities that are relevant to each other such as the entity Michelle Obama and Harvard University would have high weights and Michelle Obama and Apple would not. 

**Suitable Readers**: Those that are interested in graphical representations of language tasks. 



### Less-forgetful Learning for Domain Expansion in Deep Neural Networks (Yujia Luo)

**Authors**:  Heechul Jung, Jeongwoo Ju, Minju Jung, Junmo Kim

**Arxiv Link**:  https://arxiv.org/pdf/1711.05959.pdf	

**Published on Arxiv**: Nov, 16 2017

**Executive Summary**: The paper proposes a less-forgetful learning method for deep neural network to expand domains. The approach works well with both old and new domains without the need to discriminate between the two.

**Notable Details**: The new approach is based on two important properties:
1. The decision boundaries should be unchanged
2. The features extracted by the new network from old domain data should be close to those extracted by the old network from old domain data

The algorithm is as following: reuse the weights of the old network as those of the new network → freeze the weights of the softmax classifier layer to preserve boundaries of the classifier → train the network to minimize a composite loss function 
The new network trained by new domain data only still performs as well on old domain. The final network works well without knowing which domain the input data come from
The paper conducts two experiments on image classification:

1. Tiny images (MNIST + CIFAR-10 + SVHN)
The method was significantly more effective than the traditional fine-tuning method when the old-domain data were partially accessible
2. Large images (ImageNet)
The method improved the recognition rate by about 1.8% compared to the existing fine-tuning method

## Arxiv Summaries Week 11/27

### A Double Joint Bayesian Approach for J-Vector Based Text-Dependent Speaker Verification

**Authors**: Ziqiang Shi, Mengjiao Wang, Liu Liu, Huibin Lin, Rujie Liu

**Date Released**: 17 Nov 2017

**Arxiv link**: https://arxiv.org/pdf/1711.06434.pdf

**Executive summary**: This paper introduces a new model--Double Joint Bayesian (DoJoBa) which effectively make uses of J-Vector with classic joint Bayesian model to perform text-dependent speaker verification with short-duration speech. J-Vector is a deep feature that is proved to be very effective to capture both speaker and phrase information that is used for speech verification, obtained by training DNN with cross-entropy losses related to speaker and text label. Joint Bayesian model is the current back-end state-of-art classifier. This model could not properly deal with J-vector as the equation used for data generation contains a term that need to reflect both speaker and text info. The proposed DoJoBa model separate the term into two independent label variable.

**Notable details**: This paper is related to the work of joint PLDA proposed for J-vector verification. The advantage of DoJoBa over jPLDA is that DoJoBa does not require user tuning in order to learn appropriate dimensionality of low-rank user subspace. Two dataset - public RSR2015 English corpus and internal Huiting202 Chinese Mandarin database collected by the Huiting Techonogly - were used to evaluate the performance of 4 different model: J-vector, joint Bayesian, jPLDA, and DoJoBa. DoJoBa outperforms all the other 3 models. The error rate of DoJoBa is significantly lower than J-vecor and joint Bayesian, close to jPLDA.
Suitable Readers: For those who are interest in speaker verification. The idea is simple for common reader but there are extensive math that require some math background.

### Deep Long Short-Term Memory Adaptive Beamforming Networks For Multichannel Robust Speech Recognition

**Authors**: Zhong Meng, Shinji Watanabe, John R. Hershey, Hakan Erdogan

**Link**: https://arxiv.org/abs/1711.08016 

**Date**: 21 November 2017

**Topics**: Beamforming, Speech recognition (ASR) in noisy and reverberant environments, moving sources and multiple microphones

**Executive Summary**: The authors propose an approach, in which they use an LSTM to adaptively estimate the real-time beamforming filter coefficients, “to cope with non-stationary environmental noise and dynamic nature of source and microphones positions which results in a set of time-varying room impulse responses”.
 
Although substantial performance increases were achieved in Automatic Speech Recognition (ASR) through the application of deep neural networks in the last years, robust speech recognition in reverberant environments with multiple microphones remains a challenge. A frequently used approach is the so-called beamforming, where beams are steered towards the position of a target source. In the course of this, a spatial suppression is inserted, trying to negate noise, reverb, etc. In real world applications, Delay-and-sum (DAS) is widely used. With this approach signals are delayed, such that they overlap in time, and then get summed up to one signal.
 
An approach by X. Xiao (2016) uses DNNs to estimate parameters of a frequency-domain beamformer. However, in real world scenarios the estimated coefficients fail to robustly enhance the target signal. Therefore, this paper proposes to “adaptively estimate the beamforming filter coefficients at each time frame using an LSTM to deal with any possible changes of the source, noise or channel conditions”. The enhanced signal is then generated by applying Short-time Fourier transformation. The obtained features are then passed into a LSTM acoustic model, using truncated back-propagation through time.
 
The performance improvement (8% over baseline) suggests that the adaptive LSTM is able to estimate the filter coefficients for moving target sources.  


### Relating Input Concepts to Convolutional Neural Network Decisions

**Authors**: Ning Xie, Md Kamruzzaman Sarker, Derek Doran, Pascal Hitzler, Michael Raymer

**Arxiv Link**: https://arxiv.org/pdf/1711.08006.pdf

**Arxiv Published Date**: 21 Nov 2017

**Executive Summary**: The existing methods to interpret convolutional neural networks all made an assumption that the concepts of the input has instrumental effects in the decision making of CNN, but the nature of this influence was not well explored. This paper addresses this problem by examining the quality of a concept's recognition by a CNN and the degree of how the recognition is related with the CNN decisions. Then, the paper uses a novel method to score the strength of minimally distributed representations of input concepts across late stage feature maps. The result shows that concept recognition does influence CNN's decision: Strong recognition of concepts frequently-occurring in few scenes are indicative of correct decisions, but recognizing concepts common to many scenes can mislead the network.

**Notable Details**: In this paper, it uses image object annotations in the ADE20k scene dataset. During the analysis, it found a weak relationship between the average recognition of image concepts in a scene and classification accuracy. They discovered that the relationship is impeded by recognized concepts that are “sparse” and by “misleading” concepts that appear in many images across many different scenes. Recognizing “unique” concepts is moderately positively correlated with the CNN’s classification accuracy. It means the kind of concepts that appear often but in a limited set of scenes.
Suitable Readers: Those who want to know how the recognition of concepts of input will affect the decision making of convolutional neural networks.

### DeepSign: Deep Learning for Automatic Malware Signature Generation and Classification

**Authors**: Eli (Omid) David, Nathan S. Netanyahu

**Link**: https://arxiv.org/pdf/1711.08336.pdf

**Published Date**: 21 Nov 2017

**Executive Summary**: This paper relies on training deep belief network(DBN) - a deep unsupervised neural network, to generate an invariant compact representation of the malware behavior. Compared with conventional signature and token-based methods, this method can detect a majority of new variants for existing malware. The first step is to run malware programs in a sandbox to generate a sandbox log, which is a text file that contains the behavior of the programs. Then, the text file is parsed and converted into a binary bit-string as the input of the neural network. Next, it trains a deep stack of denoising auto-encoders of eight layers and finally produces a 30-sized vector as the “signature” of the program.

**Notable Details**: An auto-encoder sets the number of neurons at the input and the number of neurons at the output to be equal. And the number of neurons at the hidden layer is set to be fewer. A denoising auto-encoder instead of a common auto-encoder is used to diminish the risk of overfitting. To get a denoising auto-encoder, a small portion of the given input is corrupted by adding some noise x ̃. The trained denoising autoencoder is consisted of eight layer and during each step, the weight of one layer (only trains one layer each step) is frozen and the subsequent layer is trained. The output is a vector that contains 30 floating-point values, which represent as the signature of the program.
Suitable Readers: For those who interested in deep learning and computer security area.

### Deep Learning for Physical Processes: Incorporating Prior Scientific Knowledge

**Authors**: Emmanuel de Bézenac, Arthur Pajot, Patrick Gallinari

**Date Released**: 21 Nov 2017

**Arxiv link**: https://arxiv.org/pdf/1711.07970.pdf

**Executive summary**: The paper “deep learning for physical processes: incorporating prior scientific knowledge” considers the use of deep learning methods for complex natural processes. They show how general background knowledge gained from physics could be used as a guideline for designing efficient Deep Learning models, using an example application - Sea Surface Temperature (SST) Prediction. They raised two issues: one is whether ML techniques are ready to be used to model complex model complex physical phenomena, and the other is how general knowledge gained from the physical modeling paradigm could help designing efficient ML models. Their goal is to use the solution to this problem as an illustration for advancing on these challenges. 

**Notable details**: Rather than directly estimating the temperature, the motion is estimated from the input images with a convolutional neural network. A warping scheme then displaces the last input image along this motion estimate to produce the future image. They evaluate the discrepancy between the warped image and the target image using the Charbonnier penalty function , which is known to reduce the influence of outliers compared to an l2 norm. Their model gives the best MSE score than any of the baselines.
