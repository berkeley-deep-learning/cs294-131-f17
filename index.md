## Instructors
<div class="instructor">
  <a href="https://people.eecs.berkeley.edu/~trevor/">
  <div class="instructorphoto"><img src="trevordarrell.jpg"></div>
  <div>Trevor Darrell</div>
  </a>
</div>
<div class="instructor">
  <a href="https://people.eecs.berkeley.edu/~dawnsong/">
  <div class="instructorphoto"><img src="dawnsong.jpg"></div>
  <div>Dawn Song 
  <a href="https://twitter.com/intent/follow?original_referer=https%3A%2F%2Fpeople.eecs.berkeley.edu%2F~dawnsong%2F&ref_src=twsrc%5Etfw&screen_name=dawnsongtweets&tw_p=followbutton"><img src="twitter.jpg"></a>
  </div>
  </a>
  
</div>

## Teaching Assistants
<div class="instructor">
  <a href="https://people.eecs.berkeley.edu/~lisa_anne/">
  <div class="instructorphoto"><img src="lisaannehendricks.jpg"></div>
  <div>Lisa Anne Hendricks</div>
  </a>
</div>

## Office Hours

Lisa Anne: Monday 5--6:00 pm, Soda-Alcove-283H

## Lectures
**Time**: Monday 1--2:30 pm

**Location**: 306 Soda

**Room Limit**:  Soda 306 is designed for smaller courses.  We increased course enrollment so more students could benefit from this course.  However, if the room becomes too full (and thus poses a fire hazard), students who arrive after the room has reached capacity will be directed to watch the lecture remotely.  The link for the live webcast (and recorded lectures) can be found on Piazza.

You may see the intro slides from the first day of class [here](https://github.com/berkeley-deep-learning/cs294-131-f17/blob/master/cs294-131-f17-overview-8-22.pdf).

## Mailing list and Piazza
To get announcements about information about the class including guest speakers, and more generally, deep learning talks at Berkeley, please sign up for the [talk announcement mailing list](https://groups.google.com/forum/#!forum/berkeley-deep-learning) for future announcements.

If you are in the class, you may sign up on [Piazza](https://piazza.com/class/j4ock30iz2t2xk).  Additionally, you should sign up for the [class slack channel](https://cs294-131-f17.slack.com) and the [class google group](https://groups.google.com/forum/#!forum/cs-294-131-f17) (this is different than the talk announcement mailing list).

## Arxiv Summaries

This semester we started summarizing interesting papers from Arxiv each week.  Check out the papers we have chosen and summarized [here](https://github.com/berkeley-deep-learning/cs294-131-f17/blob/master/arxiv.md)!

## Syllabus

<table style="table-layout: fixed; font-size: 88%;">
  <thead>
    <tr>
      <th style="width: 5%;">Date</th>
      <th style="width: 17%;">Speaker</th>
      <th style="width: 50%;">Readings</th>
      <th style="width: 20%;">Talk</th>
      <th style="width: 8%;">Deadlines</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>08/28</td>
      <td>Anima Anandkumar</td>
      <td>
      Main Readings:
      <ul>
      <li><a href="https://arxiv.org/abs/1707.08308">Tensor Regression Networks</a> by Jean Kossaifi, Zachary Lipton, Aran Khanna, Tommaso Furlanello and Anima Anandkumar</li>
      <li><a href="https://arxiv.org/pdf/1606.05696.pdf">Tensor Contractions with Extended BLAS Kernels on CPU and GPU</a> by Yang Shi, U.N. Niranjan, Anima Anandkumar, Cris Cecka.  Also see a <a href="https://devblogs.nvidia.com/parallelforall/cublas-strided-batched-matrix-multiply/">blog post</a> and <a href="http://newport.eecs.uci.edu/anandkumar/pubs/tensorcontraction_poster.pdf">poster</a>.</li>
      </ul>
      Background Reading:
      <ul>
      <li><a href="https://arxiv.org/pdf/1210.7559.pdf">Tensor Decompositions for Learning Latent Variable Models</a> by A. Anandkumar, R. Ge, D. Hsu, S.M. Kakade and M. Telgarsky. Also see a <a href="http://www.offconvex.org/2015/12/17/tensor-decompositions">blog post</a>. </li>
       <li><a href="https://arxiv.org/pdf/1506.04448.pdf">Fast and Guaranteed Tensor Decomposition via Sketching</a> by A. Anandkumar, R. Ge, D. Hsu, S.M. Kakade and M. Telgarsky. Also see a <a href="http://newport.eecs.uci.edu/anandkumar/pubs/poster_fftlda.pdf">poster</a>. </li>
       </ul>
       Jupyter notebooks (credits will be provided on AWS to run them):
       <ul>
       <li> <a href="https://github.com/JeanKossaifi/tensorly_notebooks/">Tensors on tensorly package (with mxnet backend) </a></li>
       <li> <a href="http://thestraightdope.mxnet.io/">Gluon tutorials for deep learning</a> </li>
       <li> <a href="https://github.com/shiyangdaisy23/vqa-mxnet-gluon/blob/master/VQA-gluon.ipynb">Visual question & answering using sketches</a></li>
       </ul>
       </td>
      <td><a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#anima-anandkumar-role-of-tensors-in-machine-learning">Role of Tensors in Machine Learning</a>       </td>
      <td></td>
    </tr>
        <tr>
      <td>09/05</td>
      <td>Labor Day - No Class</td>
      <td>
      </td>
      <td></td>
      <td></td>
    </tr>    
    <tr>
      <td>09/11</td>
      <td>Vladlen Koltun</td>
      <td>
      Main Readings:
      <ul>
      <li><a href="http://vladlen.info/publications/learning-act-predicting-future/">Learning to Act by Predicting the Future</a> by A. Dosovitskiy and V. Koltun</li>
      <li><a href="http://vladlen.info/publications/playing-for-benchmarks/">Playing for Benchmarks</a> by S. Richter, Z, Hayder, and V. Koltun</li>
      </ul>
      Background Reading:
      <ul>
      <li>
      <a href="https://papers.cnl.salk.edu/PDFs/A%20Critique%20of%20Pure%20Vision%201994-2933.pdf">A Critique of Pure Vision</a> by P. Churchland, V.S. Ramachandran, and T. Sejnowski. </li>
      <li><a href="http://vladlen.info/publications/playing-data-ground-truth-computer-games/">Playing for Data: Ground Truth from Computer Games</a> by S.Richter, V. Vineet, S. Roth, and V. Kolton.
      </li>
      </ul>
      </td>
      <td><a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#vladlen-koltun-learning-to-act-with-natural-supervision">Learning to Act with Natural Supervision</a> </td>
      <td></td>
    </tr>  
    <tr>
      <td>09/18</td>
      <td>Jianfeng Gao</td>
      <td>Main Readings:
      <ul>
      <li><a href="https://arxiv.org/pdf/1609.05284.pdf">ReasoNet: Learning to Stop Reading in Machine Comprehension</a> by Y. Shen, P. Huang, J. Gao, and W. Chen</li>
      <li><a href="https://arxiv.org/abs/1609.00777">Towards End-to-End Reinforcement Learning of Dialogue Agents for Information Access</a> by B. Dhingra, L.Li, X. Li, J. Gao, Y.Chen, F. Ahmed, and L. Deng</li>
      </ul>
      Background Reading:
      <ul>
      <li><a href="https://arxiv.org/pdf/1606.05250.pdf">SQuAD: 100,000+ Questions for Machine Comprehension of Text</a> by P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang</li>
      <li><a href="http://cs.brown.edu/courses/csci2951-k/papers/young13.pdf">POMDP-Based Statistical
Spoken Dialog Systems:
A Review</a> by S. Young, M. Gasic, B. Thomson, and J.Williams</li>
      </ul>
      </td>
      <td><a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#jianfeng-gao-neural-appraches-to-machine-reading-comprehension-and-dialogue">Neural approaches to Machine Reading Comprehension and Dialogue</a></td>
      <td>Project Proposal Due</td>
    </tr>
    <tr>
      <td>09/25</td>
      <td>Quoc Le and Barret Zoph</td>
      <td>
      Main Reading:
      <ul>
      <li><a href="https://www.openreview.net/pdf?id=r1Ue8Hcxg">Neural Architecture Search with Reinforcement Learning</a> by B. Zoph and Q. Le
      </li>
      <li><a href="https://arxiv.org/abs/1707.07012">Learning Transferable Architectures for Scalable Image Recognition</a> by B. Zoph, V. Vasudevan, J. Schlens, and Q. Le
      </ul>
      </td>
      <td><a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#barret-zoph-and-quoc-lelearning-transferable-architectures-for-imageNet">Learning Transferable Architectures for ImageNet</a></td>
      <td></td>
    </tr>
    <tr>
      <td>10/02</td>
      <td>Ross Girshik</td>
      <td>TBA</td>
      <td>TBA</td>
      <td></td>
    </tr>
    <tr>
      <td>10/09</td>
      <td>Igor Mordatch</td>
      <td>TBA</td>
      <td>TBA</td>
      <td></td>
    </tr>
    <tr>
      <td>10/16</td>
      <td>David Patterson</td>
      <td>TBA</td>
      <td>TBA</td>
      <td></td>
    </tr>
    <tr>
      <td>10/23</td>
      <td>Matthew Johnson</td>
      <td>TBA</td>
      <td>TBA</td>
      <td></td>
    </tr>
    <tr>
      <td>10/30</td>
      <td>Percy Liang</td>
      <td>
      Main reading:
      <ul>
      <li>
      <a href="https://arxiv.org/pdf/1703.04730.pdf">Understanding Black-box Predictions via Influence Functions</a> by P.W. Koh and P. Liang</li>
      <li>
      <a href="https://arxiv.org/pdf/1706.08605.pdf">Developing Bug-Free Machine Learning Systems With Formal Mathematics</a> by D. Selsam, P. Liang, and D. Dill.</li>
      </ul>
      Background Reading:
      <ul>
      <li>
      <a href="http://arxiv.org/pdf/1702.08608.pdf">A Roadmap for a Rigorous Science of Interpretability</a> by F. Doshi-Velez and B. Kim.</li>
      <li>
      <a href="http://arxiv.org/pdf/1705.07263.pdf">Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods</a> by N. Carlini and D. Wagner.</li>
      <li>
      <a href="https://arxiv.org/abs/1506.05254">Gradient Estimation Using Stochastic Computation Graphs </a> by J. Schulman, N. Heess, T. Weber, and P. Abbeel</li>
      </ul>
      </td>
      <td>
      <a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#percy-liang-fighting-black-boxes-adversaries-and-bugs-in-deep-learning">Fighting Black Boxes, Adversaries, and Bugs in Deep Learning</a>
      </td>
      <td>Project Milestone Due</td>
    </tr>
    <tr>
      <td>11/06</td>
      <td>Li Deng</td>
      <td>
      Main Reading:
      <ul>
      <li><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/HintonDengYuEtAl-SPM2012.pdf">Deep neural networks for acoustic modeling in speech recognition</a> by G. Hinton, L. Deng, D. Yu, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhouke, P. Nguyen, T. Sainath, and B. Kingsbury.</li>
      <li><a href="https://arxiv.org/pdf/1702.07817.pdf">An Unsupervised Learning Method Exploiting Sequential Output Statistics</a> by Y. Liu, J. Chen, and L.Deng.
      </li>
      </ul>
      Background Reading:
      <ul>
      <li><a href="http://www.aclweb.org/anthology/P13-1021">Unsupervised transcription of historical documents</a> by T. Berk-Kirkpatrick, G. Durrett, and D. Klein.
      </li>
      <li><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=GQWTo4MAAAAJ&citation_for_view=GQWTo4MAAAAJ:t13vp6B-Zw8C">Deep Learning: Methods and Applications (Ch. 2, 7, 11)</a> by L. Deng and D. Yu.
      </li>
      <li>
      <a href="http://www.deeplearningbook.org/">Deep Learning (Ch. 8, 10, 20)</a> by I. Goodfellow, Y. Bengio, and A. Coourville.
      </li>
      </ul>
      </td>
      <td><a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#li-deng-from-supervised-to-unsupervised-deep-learning-successes-and-challenges">From Supervised to Unsupervised Deep Learning: Successes and Challenges</a></td>
      <td></td>
    </tr>
    <tr>
      <td>11/13</td>
      <td>Rob Fergus</td>
      <td>TBA</td>
      <td>TBA</td>
      <td></td>
    </tr>
    <tr>
      <td>11/20</td>
      <td>Rishabh Singh</td>
      <td>Main Reading:
      <ul>
      <li><a href="https://rishabhmit.bitbucket.io/papers/icml17.pdf">RobustFill: Neural Program Learning under Noisy I/O </a> by J. Devlin, J. Uesato, S. Bhupatiraju, R. Singh, A. Mohamed, and P. Kohli.</li>
      <li><a href="https://arxiv.org/abs/1611.01855">Neuro-symbolic Program Synthesis</a> by E. Parisotto, A. Mohamed, R. Singh, L. Li, D. Zhou, and P. Kohli</li>
      </ul>
      Background Reading:
      <ul>
      <li><a href="https://rishabhmit.bitbucket.io/papers/ap_snapl.pdf">AP: Artificial Programming</a> by R. Singh and P. Kohli</li>
      <li><a href="https://arxiv.org/abs/1410.5401" >Neural Turing Machines</a> by A. Graves, G. Wayne, I. Danihelka</li>
      </ul>
      </td>
      <td><a href="https://berkeley-deep-learning.github.io/cs294-131-f17/speakers.html#rishabh-singh-neural-program-synthesis">Neural Program Synthesis</a></td>
      <td></td>
    </tr>
    <tr>
      <td>11/27</td>
      <td>Danny Tarlow</td>
      <td>TBA</td>
      <td>TBA</td>
      <td></td>
    </tr>
    <tr>
      <td>11/27 DATE CHANGED!</td>
      <td>Poster Session</td>
      <td>3:00-5:00 (tentative)</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>12/09</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Final Report Due</td>
    </tr>
  </tbody>
</table>

## Course description
In recent years, deep learning has enabled huge progress in many domains including computer vision, speech, NLP, and robotics. It has become the leading solution for many tasks, from winning the ImageNet competition to winning at Go against a world champion. This class is designed to help students develop a deeper understanding of deep learning and explore new research directions and applications of deep learning. It assumes that students already have a basic understanding of deep learning. In particular, we will explore a selected list of new, cutting-edge topics in deep learning, including new techniques and architectures in deep learning, security and privacy issues in deep learning, recent advances in the theoretical and systems aspects of deep learning, and new application domains of deep learning such as autonomous driving.

## Class format and project
This is a lecture, discussion, and project oriented class. Each lecture will focus on one of the topics, including a survey of the state-of-the-art in the area and an in-depth discussion of the topic. Each week, students are expected to complete reading assignments before class and participate actively in class discussion.

Students will also form project groups (two to three people per group) and complete a research-quality class project.

## Enrollment information
**For undergraduates**: Please note that this is a graduate-level class. However, with instructors' permission, we do allow qualified undergraduate students to be in the class. If you are an undergraduate student and would like to enroll in the class, please fill out **[this form](https://docs.google.com/forms/d/e/1FAIpQLSdQT0hPZQ0UjjTarXel3f5ZvQV2XmeMf70MoB7CStaihrNtTA/viewform)** and come to the first lecture of the class. Qualified undergraduates will be given instructor codes to be allowed to register for the class after the first lecture of the class, subject to space availability.

Students may enroll in this class for variable units.

* 1 unit: Participate in reading assignments (including serving as discussion lead once and Arxiv lead once).
* 2 units: Complete a project.  Projects may fall into one of four categories:
  * Traditional Literature Review of a deep learning topic (e.g., literature review of deep dialogue systems)
  * Distill-like Literature Review of a deep learning topic (e.g., a Distill-like blog post illustrating different optimization techniques used in deep learning)
  * Reimplement research code and open source it
  * Conference level research project
* 3 units: Both reading assignments and a project.
* You **may not** take this class for 4 units.

## Deadlines
* Reading assignment deadlines:
  * For students,
    * Submit questions by Friday noon
    * Vote on the poll of discussion questions by Saturday 11:59 pm
  * For discussion leads,
    * Send form to collect questions from students by Wednesday 11:59 pm
    * Summarize questions proposed by students to form the poll and send it by Friday 11:59 pm
    * Summarize the poll to generate a ranked & categorized discussion question list and send the list to teaching staff by Sunday 7pm
 * Arxiv leads (new this semester!),
    * Throughout the week discuss papers which have appeared on Arxiv during the prior week on Slack.  All Arxiv leads are expected to be involved in the Slack discussion.  Other students may participate as well, but they are not required to.  
      * [Arxiv](https://arxiv.org/) is an archive of scientific papers covering a broad set of fields.  When researchers want to share their results, they frequently place the paper on Arxiv.  There are a few Arxiv pages Arxiv leads should follow: [Computation and Language](https://arxiv.org/list/cs.CL/recent), [Artificial Intelligence](https://arxiv.org/list/cs.AI/recent), [Computer Vision and Pattern Recognition](https://arxiv.org/list/cs.CL/recent), [Learning](https://arxiv.org/list/cs.CV/recent), [Robotics](https://arxiv.org/list/cs.RO/recent), and [Neural and Evolutionary Computing](https://arxiv.org/list/cs.NE/recent).  Not all papers are deep learning papers, but most (if not all) deep learning papers will fall into one of these categories.
    * Choose (approximately) five exciting papers and write a short summary (0.5 pages) for each paper and send to the TA by Monday morning.
    * Give a five minute presentation at the beginning of the next class (five minutes total, not per paper).
 

## Grading
* 20% class participation
* 25% weekly reading assignment
  * 7.5% discussion leads
  * 7.5% Arxiv leads (new this semester!)
  * 10% individual reading assignments
* 55% project

## Additional Notes
* For students who need computing resources for the class project, we recommend you to look into AWS educate program for students. You'll get 100 dollar's worth of sign up credit. Here's the <a href="https://aws.amazon.com/education/awseducate/apply/"> link </a>. 
