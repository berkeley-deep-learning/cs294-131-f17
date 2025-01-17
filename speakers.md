## Anima Anandkumar: Role of Tensors in Machine Learning

### Abstract

Tensors are higher order extensions of matrices that can incorporate multiple modalities and encode higher order relationships in data. Tensors play a significant role in machine learning through (1) tensor contractions, (2) tensor sketches, and (3) tensor decompositions. Tensor contractions are extensions of matrix products to higher dimensions. Tensor sketches efficiently compress tensors while preserving information. Tensor decompositions compute low rank components that constitute a tensor.  I will show that tensor contractions are an effective replacement for fully connected layers in deep learning architectures and result in significant space savings. This is because tensor contractions retain the multi-dimensional dependencies of activation tensors while the fully connected layers flatten them and lose this information. Tensor contractions present rich opportunities for hardware optimizations through extended BLAS kernels. We propose a new primitive known as StridedBatchedGEMM in CuBLAS 8.0 that significantly speeds up tensor contractions, and avoids explicit copy and transpositions. Tensor sketches are extensions of the popular count sketches for vectors and provide succinct representations in multi-modal tasks such as visual question and answering. Here, the image and text embeddings are pooled together for better performance through tensor sketches. Lastly, I will present analysis on how tensor decompositions can efficiently learn latent variable models with guarantees.  These functionalities will be demonstrated on Jupyter notebooks on AWS using Tensorly package and has the Apache Mxnet backend for efficient multi-GPU scaling.  It uses the new gluon interface with both declarative and imperative programming to combine ease of programming with high efficiency.  AWS credits will be provided to run these notebooks.

### Bio

Anima Anandkumar is a Bren professor at Caltech CMS department and a principal scientist at Amazon Web Services.   Her research interests are in the areas of large-scale machine learning, non-convex optimization and high-dimensional statistics. In particular, she has been spearheading the development and analysis of tensor algorithms.  She is the recipient of several awards such as the Alfred. P. Sloan Fellowship, Microsoft Faculty Fellowship, Google research award, ARO and AFOSR Young Investigator Awards, NSF Career Award, Early Career Excellence in Research Award at UCI, Best Thesis Award from the ACM Sigmetrics society, IBM Fran Allen PhD fellowship, and several best paper awards. She has been featured in a number of forums such as the yourstory, Quora ML session, Huffington post, Forbes, O’Reilly media, and so on.  She received her B.Tech in Electrical Engineering from IIT Madras in 2004 and her PhD from Cornell University in 2009. She was a postdoctoral researcher at MIT from 2009 to 2010, an assistant professor at U.C. Irvine between 2010 and 2016, and a visiting researcher at Microsoft Research New England in 2012 and 2014.

[Slides.]("slides/berkeley-aug2017.pdf")

## Vladlen Koltun: Learning to Act with Natural Supervision

### Abstract

I will begin by discussing the common organization of machine learning into supervised, unsupervised, and reinforcement learning. I will argue that these distinctions have largely outlived their usefulness, in part because the terms are heavily overloaded and mean different things to different people (or even the same people at different times of day). I will discuss natural supervision as a regime that supports robust algorithmic machinery, scalable datasets, and the coupling of perception and action. This will be illustrated via recent work that uses natural supervision to train intelligent systems that act in complex and dynamic three-dimensional environments based on raw sensory input.

### Bio

Vladlen Koltun is the Director of the Intel Visual Computing Lab. He received a PhD in 2002 for new results in theoretical computational geometry, spent three years at UC Berkeley as a postdoc in the theory group, and joined the Stanford Computer Science faculty in 2005 as a theoretician. He switched to research in visual computing in 2007 and joined Intel as a Principal Researcher in 2015 to establish the Visual Computing Lab.

## Jianfeng Gao: Neural approaches to Machine Reading Comprehension and Dialogue

### Abstract

In this talk, I start with a brief introduction to the history of symbolic approaches to natural language processing (NLP), and why we move to neural approaches recently. Then I describes in detail the deep learning technologies that are recently developed for two areas of NLP tasks. First is a set of neural attention and inference models developed for machine reading comprehension and question answering. Second is the use of deep learning for various of dialogue agents, including task-completion bots and social chat bots.

### Bio

Jianfeng Gao is Partner Research Manager at Microsoft AI and Research, Redmond. He works on deep learning for text and image processing and leads the development of AI systems for machine reading comprehension (MRC), question answering (QA), dialogue, and business applications. From 2006 to 2014, he was Principal Researcher at Natural Language Processing Group at Microsoft Research, Redmond, where he worked on Web search, query understanding and reformulation, ads prediction, and statistical machine translation. From 2005 to 2006, he was a Research Lead in Natural Interactive Services Division at Microsoft, where he worked on Project X, an effort of developing natural user interface for Windows. From 2000 to 2005, he was Research Lead in Natural Language Computing Group at Microsoft Research Asia, where he and his colleagues developed the first Chinese speech recognition system released with Microsoft Office, the Chinese/Japanese Input Method Editors (IME) which were the leading products in the market, and the natural language platform for Microsoft Windows.

[Slides](slides/MSR%20and%20Dialogue%20-%20UCBerkeley%2009182017%20-%20JianfengGao.pdf)

## Barret Zoph and Quoc Le: Learning Transferable Architectures for ImageNet

### Abstract

In this talk, we will discuss the use of Reinforcement Learning to automate the process of designing neural architectures (Neural Architecture Search).
We will also report our recent results on applying this method to the ImageNet classification dataset.

### Bio

Quoc is a research scientist at Google Brain. Prior to Google Brain, Quoc did his PhD at  Stanford. He was recognized at one of top innovators in 2014 by MIT Technology Review for his work on large scale deep learning. 

Barret is a research scientist at Google Brain. Prior to Google Brain, Barret did his undergraduate at USC and worked on Natural Language Understanding.

[Slides.]("slides/le-zoph-slides.pdf")

## Ross Girshick: The Past, Present, and Future of Object Detection

### Abstract

Object detection has rapidly transformed from a research area in which not much worked, to one that some people now consider solved. If it is solved, what's next? In this talk, I will cover the nuts and bolts of current state-of-the-art object detection systems (the “solution”), relate these ideas to classical methods, and share thoughts on the future of research in this and related areas. Perhaps unsurprisingly, I will argue against the view that object detection is a solved problem and instead provide evidence for how its challenges have shifted over time. This talk will cover recent work from Facebook AI Research (FAIR) on Feature Pyramid Networks and Mask R-CNN, which are state-of-the-art on the COCO Challenge Dataset.


### Bio

Ross Girshick is a research scientist at Facebook AI Research (FAIR), working on computer vision and machine learning. He received a PhD in computer science from the University of Chicago under the supervision of Pedro Felzenszwalb in 2012. Prior to joining FAIR, he completed a postdoc at the University of California, Berkeley, where he was advised by Jitendra Malik and Trevor Darrell, and he was a researcher at Microsoft Research, Redmond. His interests include instance-level object understanding and visual reasoning challenges that combine natural language processing with computer vision. He received the 2017 PAMI Young Researcher Award and is well-known for developing the R-CNN (Region-based Convolution Neural Network) approach to object detection.

Ross has many nice images in his slides, which means the powerpoint was too large for me to put on git, so it is in parts.  If you want to combine the files and compress into a single powerpoint, please do!

[Intro.]("slides/past_present_future_obj_det_intro.pptx")
[Mask-RCNN-1.]("slides/past_present_future_obj_det_mask_rcnn1.pptx")
[Mask-RCNN-2.]("slides/past_present_future_obj_det_mask_rcnn2.pptx")
[Future.]("slides/past_present_future_obj_det_future.pptx")
[Questions.]("slides/past_present_future_obj_det_questions.pptx")


## Igor Mordatch: Emergence of Grounded Compositional Language in Multi-Agent Populations

### Abstract

By capturing statistical patterns in large corpora, machine learning has enabled significant advances in natural language processing, including in machine translation, question answering, and sentiment analysis. However, for agents to intelligently interact with humans, simply capturing the statistical patterns is insufficient. In this paper we investigate if, and how, grounded compositional language can emerge as a means to achieve goals in multi-agent populations. Towards this end, we propose a multi-agent learning environment and learning methods that bring about emergence of a basic compositional language. This language is represented as streams of abstract discrete symbols uttered by agents over time, but nonetheless has a coherent structure that possesses a defined vocabulary and syntax. We also observe emergence of non-verbal communication such as pointing and guiding when language communication is unavailable.

### Bio

Igor Mordatch is a research scientist at OpenAI and faculty at Carnegie Mellon University Robotics Institute. Previously he was a post-doctoral fellow working with professor Pieter Abbeel at University of California, Berkeley and received his PhD at University of Washington and undergraduate degree in University of Toronto. He worked as a visiting researcher at Stanford University and Pixar Research. His research interests lie in the development and use of optimal control and reinforcement learning techniques for robotics.

## David Patterson: Evaluation of a Domain-Specific Architecture for Deep Neural Networks in the Datacenter: The Google TPU

### Abstract

With the ending of Moore's Law, many computer architects believe that major improvements in cost-energy-performance must now come from domain-specific hardware. The Tensor Processing Unit (TPU), deployed in Google datacenters since 2015, is a custom chip that accelerates deep neural networks (DNNs).  We compare the TPU to contemporary server-class CPUs and GPUs deployed in the same datacenters. Our benchmark workload, written using the high-level TensorFlow framework, uses production DNN applications that represent 95% of our datacenters’ DNN demand. The TPU is  an order of magnitude faster than contemporary CPUs and GPUs and its relative performance per Watt is even larger.

### Bio

David Patterson is likely best-known for the book Computer Architecture: A Quantitative Approach written with John Hennessy or for Berkeley research projects Reduced Instruction Set Computers (RISC), Redundant Arrays of Inexpensive Disks (RAID), and Network of Workstations. He also served as Berkeley’s CS Division Chair, the Computing Research Association Chair, and President of the Association for Computing Machinery. This led to election to the National Academy of Engineering, the National Academy of Sciences, and the Silicon Valley Engineering Hall of Fame. He received the Berkeley Citation in 2016 after 40 years as a CS professor. He then joined the Google Brain as a distinguished engineer and serves as Vice-Chair of the Board of Directors of the RISC-V Foundation, an open architecture organization.

## Matthew Johnson: Composing graphical models and neural networks for structured representations and fast inference

### Abstract

I'll discuss some recent ideas in generative modeling and approximate inference that combine the flexibility of deep learning with the efficient and interpretable representations of probabilistic graphical models. In particular, I'll focus on structured variational autoencoder (SVAE) methods, which augment latent exponential family graphical model structure, like Gaussian mixture models or switching linear dynamical systems, with neural network observation likelihoods. To enable fast inference, SVAEs leverage graph-structured approximating distributions and, building on variational autoencoders, fit recognition networks that learn to approximate difficult graph potentials with conjugate ones. I'll show how SVAE methods can be applied to learn how to parse mouse behavior from depth video.

## Percy Liang: Fighting Black Boxes, Adversaries, and Bugs in Deep Learning

### Abstract

While deep learning has been hugely successful in producing highly accurate models, the resulting models are sometimes (i) difficult to interpret, (ii) susceptible to adversaries, and (iii) suffer from subtle implementation bugs due to their stochastic nature.  In this talk, I will take some initial steps towards addressing these problems of interpretability, robustness, and correctness using some classic mathematical tools.  First, influence functions from robust statistics can help us understand the predictions of deep networks by answering the question: which training examples are most influential on a particular prediction?  Second, semidefinite relaxations can be used to provide guaranteed upper bounds on the amount of damage an adversary can do for restricted models.  Third, we use the Lean proof assistant to produce a working implementation of stochastic computation graphs which is guaranteed to be bug-free.

## Li Deng: From Supervised to Unsupervised Deep Learning: Successes and Challenges

### Abstract

Deep learning has been the main driving force in the recent resurgence of AI. The first large-scale success of deep learning in modern industry was on large vocabulary speech recognition around 2010-2011, soon followed by its successes in computer vision (2012) and then in machine translation (2014-2015) and so on. While reflecting on the historical path to these early successes of deep learning, it is important to note that the useful learning algorithms used thus far have been strongly supervised, requiring a large amount of paired input-output data to define the objective function and to then train the parameters of the deep neural nets. Such paired data are often very expensive to acquire in practical applications, and this has prevented the use of naturally-found big data several orders of magnitudes greater than can be exploited in training current deep learning systems in industry.
One most direct benefit of unsupervised learning is to drastically reduce the cost of pairing input-output data for training the many dense parameters in deep learning systems. This is expected to boost the system prediction accuracy via the use of much greater amounts of training data than supervised deep learning can afford but without suffering from prohibitively high labeling cost. A key to successful unsupervised learning is to judiciously exploit rich sources of world knowledge and prior information, including inherent statistical or geometric structures of input and output data, as well as “generative” relations from output to input. In this lecture, recent experiments will be presented on unsupervised learning in two sequential classification tasks. The unsupervised learning algorithm developed in this work makes use of statistical N-gram structure in the output sequences that do not pair with the input sequences, and is shown to achieve classification accuracy comparable to the fully supervised system with the same amount of input-output data which are closely paired. The challenges of how to design the unsupervised objective function and of how to efficiently optimize it will be presented in detail, while contrasting them with the much easier corresponding problems in supervised learning. (Joint work with Yu Liu and Jianshu Chen).

### Bio

Li Deng joined Citadel, one of the most successful investment firms in the world, as its Chief AI Officer, in May 2017. Prior to Citadel, he was Chief Scientist of AI and Partner Research Manager at Microsoft. Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play: Prior to Microsoft, he was a tenured Full Professor at the University of Waterloo in Ontario, Canada as well as teaching/research at MIT (Cambridge), ATR (Kyoto, Japan) and HKUST (Hong Kong). He is a Fellow of the IEEE, a Fellow of the Acoustical Society of America, and a Fellow of the ISCA. He has also been an Affiliate Professor at University of Washington since 2000.

He was an elected member of Board of Governors of the IEEE Signal Processing Society, and was Editors-in-Chief of IEEE Signal Processing Magazine and of IEEE/ACM Transactions on Audio, Speech, and Language Processing (2008-2014), for which he received the IEEE SPS Meritorious Service Award. In recognition of the pioneering work on disrupting speech recognition industry using large-scale deep learning, he received the 2015 IEEE SPS Technical Achievement Award for “Outstanding Contributions to Automatic Speech Recognition and Deep Learning”. He also received numerous best paper and patent awards for the contributions to artificial intelligence, machine learning, information retrieval, multimedia signal processing, speech processing, and human language technology. He is an author or co-author of six technical books on deep learning, speech processing, discriminative machine learning, and natural language processing. 


## Rob Fergus: Intrinsic Motivation and Unsupervised Learning

## Abstract

Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play: We describe a simple scheme that allows an agent to learn about its environment in an unsupervised manner. Our scheme pits two versions of the same agent, Alice and Bob, against one another. Alice proposes a task for Bob to complete; and then Bob attempts to complete the task. In this work we will focus on two kinds of environments: (nearly) reversible environments and environments that can be reset. Alice will "propose" the task by doing a sequence of actions and then Bob must undo or repeat them, respectively. Via an appropriate reward structure, Alice and Bob automatically generate a curriculum of exploration, enabling unsupervised training of the agent. When Bob is deployed on an RL task within the environment, this unsupervised training reduces the number of supervised episodes needed to learn, and in some cases converges to a higher reward. (work by Sainbayar Sukhbaatar, Arthur Szlam and colleagues at FAIR)

Unsupervised Learning of Disentangled Representations from Video: We present a new model DRNET that learns disentangled image representations from video. Our approach leverages the temporal coherence of video and a novel adversarial loss to learn a representation that factorizes each frame into a stationary part and a temporally varying component. The disentangled representation can be used for a range of tasks. For example, applying a standard LSTM to the time-vary components enables prediction of future frames. We evaluate our approach on a range of synthetic and real videos, demonstrating the ability to coherently generate hundreds of steps into the future. (work by Emily Denton and Vighnesh Birodkar)

## Bio

Rob Fergus is an Associate Professor of Computer Science at the Courant Institute of Mathematical Sciences, New York University. He is also a Research Scientist at Facebook, working in their AI Research Group. He received a Masters in Electrical Engineering with Prof. Pietro Perona at Caltech, before completing a PhD with Prof. Andrew Zisserman at the University of Oxford in 2005. Before coming to NYU, he spent two years as a post-doc in the Computer Science and Artificial Intelligence Lab (CSAIL) at MIT, working with Prof. William Freeman. He has received several awards including a CVPR best paper prize, a Sloan Fellowship & NSF Career award and the IEEE Longuet-Higgins prize.

## Rishabh Singh: Neural Program Synthesis

### Abstract

The key to attaining general artificial intelligence is to develop architectures that are capable of
learning complex algorithmic behaviors modeled as programs. The ability to learn programs allows these
architectures to learn to compose high-level abstractions with complex control-flow, which can lead to many
potential benefits: i) enable neural architectures to perform more complex tasks, ii) learn interpretable
representations (programs which can be analyzed, debugged, or modified), and iii) better generalization
to new inputs (like algorithms). In this talk, I will present some of our recent work in developing neural
architectures for learning complex regular-expression based data transformation programs from input-output examples,
and will also briefly discuss some other applications such as program repair and optimization that can benefit
from learning neural program representations.

### Bio

Rishabh Singh is a researcher in the Cognition group at Microsoft Research, Redmond. His research interests span the areas of programming languages, formal methods, and deep learning. His recent work has focused on developing new neural architectures for learning programs. He obtained his PhD in Computer Science from MIT in 2014, where he was a Microsoft Research PhD fellow and was awarded the MIT’s George M. Sprowls Award for Best PhD Dissertation. He obtained his BTech in Computer Science from IIT Kharagpur in 2008, where he was awarded the Institute Silver Medal and Bigyan Sinha Memorial Award.

## Danny Tarlow: Differentiable Interpreters

### Abstract

 I'll give an overview of synthesizing programs from input-output examples using differentiable interpreters. The main idea of differentiable interpreters is to extend the logic for how a program gets executed so that it is differentiable. Given a differentiable interpreter, we can backpropagate from input-output examples to the program representation and search over program space using gradient descent. I'll focus on our system, TerpreT, which is a framework for defining differentiable interpreters and comparing gradient-based program search to tradition discrete search-based methods. Time permitting, I'll discuss an extension of TerpreT that enables jointly synthesizing programs and learning neural network "subroutines" so that programs can operate on a mix of integer and tensor types.
 
 ### Bio
 
 Danny is a Research Scientist at Google Brain in Montreal. His main research interests are in the application of machine learning to problems involving structured data, with a particular interest in the intersection of machine learning and programming languages. He holds a Ph.D. from the Machine Learning group at the University of Toronto (2013) and previously was a Researcher at Microsoft Research Cambridge UK with a Research Fellowship at Darwin College, University of Cambridge. He was a co-editor of the recent MIT Press book on Perturbations, Optimization, and Statistics (2017). His work has won awards at UAI (Best Student Paper, Runner Up), ICML and NIPS Workshops (Best Paper x 2), and NIPS (Best Paper). 
