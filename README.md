Download Link: https://assignmentchef.com/product/solved-cs224n-assignment-2
<br>
<strong>Note: In this assignment, the inputs to neural network layers will be row vectors because this is standard practice for TensorFlow (some built-in TensorFlow functions assume the inputs are row vectors). This means the weight matrix of a hidden layer will right-multiply instead of left-multiply its input (i.e., </strong><em>xW </em>+ <em>b </em><strong>instead of </strong><em>Wx </em>+ <em>b</em><strong>).</strong>

<h1>1           Tensorflow Softmax</h1>

In this question, we will implement a linear classifier with loss function

<em>J</em>(<em>W</em>) = <em>CE</em>(<em>y,</em>softmax(<em>xW</em>))

Where <em>x </em>is a row vector of features and <em>W </em>is the weight matrix for the model. We will use TensorFlow’s automatic differentiation capability to fit this model to provided data.

<ul>

 <li>(5 points, coding) Implement the softmax function using TensorFlow in py. Remember that</li>

</ul>

softmax(<em>x</em>)

Note that you may <strong>not </strong>use tf.nn.softmax or related built-in functions. You can run basic (nonexhaustive tests) by running python q1softmax.py.

<ul>

 <li>(5 points, coding) Implement the cross-entropy loss using TensorFlow in py. Remember that</li>

</ul>

<em>N<sub>c</sub></em>

<em>CE</em>(<em>y,</em><em>y</em>ˆ) = −<sup>X</sup><em>y<sub>i </sub></em>log(ˆ<em>y<sub>i</sub></em>)

<em>i</em>=1

where <em>y </em>∈ R<em><sup>N</sup></em><em><sup>c </sup></em>is a one-hot label vector and <em>N<sub>c </sub></em>is the number of classes. This loss is summed over all examples (rows) of a minibatch. Note that you may <strong>not </strong>use TensorFlow’s built-in cross-entropy functions for this question. You can run basic (non-exhaustive tests) by running python q1softmax.py.

<ul>

 <li>(5 points, coding/written) Carefully study the Model class in py. Briefly explain the purpose of placeholder variables and feed dictionaries in TensorFlow computations. Fill in the implementations for addplaceholders and createfeeddict in q1classifier.py.</li>

</ul>

1

<strong>Hint: </strong>Note that configuration variables are stored in the Config class. You will need to use these configuration variables in the code.

<ul>

 <li>(5 points, coding) Implement the transformation for a softmax classifier in the function addpredictionop in py. Add cross-entropy loss in the function addlossop in the same file. Use the implementations from the earlier parts of the problem, <strong>not </strong>TensorFlow built-ins.</li>

 <li>(5 points, coding/written) Fill in the implementation for addtrainingop in py. Explain how TensorFlow’s automatic differentiation removes the need for us to define gradients explicitly. Verify that your model is able to fit to synthetic data by running python q1classifier.py and making sure that the tests pass.</li>

</ul>

<strong>Hint: </strong>Make sure to use the learning rate specified in Config.

<h1>2           Neural Transition-Based Dependency Parsing</h1>

In this section, you’ll be implementing a neural-network based dependency parser. A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between “head” words and words which modify those heads. Your implementation will be a <em>transition-based </em>parser, which incrementally builds up a parse one step at a time. At every step it maintains a partial parse, which is represented as follows:

<ul>

 <li>A <em>stack </em>of words that are currently being processed.</li>

 <li>A <em>buffer </em>of words yet to be processed.</li>

 <li>A list of <em>dependencies </em>predicted by the parser.</li>

</ul>

Initially, the stack only contains ROOT, the dependencies lists is empty, and the buffer contains all words of the sentence in order. At each step, the parse applies a <em>transition </em>to the partial parse until its buffer is empty and the stack is of size 1. The following transitions can be applied:

<ul>

 <li>SHIFT: removes the first word from the buffer and pushes it onto the stack.</li>

 <li>LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack.</li>

 <li>RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack.</li>

</ul>

Your parser will decide among transitions at each state using a neural network classifier. First, you will implement the partial parse representation and transition functions.

<ul>

 <li>(6 points, written) Go through the sequence of transitions needed for parsing the sentence <em>“I parsed this sentence correctly”</em>. The dependency tree for the sentence is shown below. At each step, give the configuration of the stack and buffer, as well as what transition was applied this step and what new dependency was added (if any). The first three steps are provided below as an example.</li>

</ul>

<table width="587">

 <tbody>

  <tr>

   <td width="122">stack</td>

   <td width="219">buffer</td>

   <td width="111">new dependency</td>

   <td width="135">transition</td>

  </tr>

  <tr>

   <td width="122">[ROOT]</td>

   <td width="219">[I, parsed, this, sentence, correctly]</td>

   <td width="111"> </td>

   <td width="135">Initial Configuration</td>

  </tr>

  <tr>

   <td width="122">[ROOT, I]</td>

   <td width="219">[parsed, this, sentence, correctly]</td>

   <td width="111"> </td>

   <td width="135">SHIFT</td>

  </tr>

  <tr>

   <td width="122">[ROOT, I, parsed]</td>

   <td width="219">[this, sentence, correctly]</td>

   <td width="111"> </td>

   <td width="135">SHIFT</td>

  </tr>

  <tr>

   <td width="122">[ROOT, parsed]</td>

   <td width="219">[this, sentence, correctly]</td>

   <td width="111">parsed→I</td>

   <td width="135">LEFT-ARC</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>(2 points, written) A sentence containing <em>n </em>words will be parsed in how many steps (in terms of <em>n</em>)?</li>

</ul>

Briefly explain why.

<ul>

 <li>(6 points, coding) Implement the init and parsestep functions in the PartialParse class in py. This implements the transition mechanics your parser will use. You can run basic (not-exhaustive) tests by running python q2parsertransitions.py.</li>

 <li>(6 points, coding) Our network will predict which transition should be applied next to a partial parse. We could use it to parse a single sentence by applying predicted transitions until the parse is complete. However, neural networks run much more efficiently when making predictions about batches of data at a time (i.e., predicting the next transition for a many different partial parses simultaneously). We can parse sentences in minibatches with the following algorithm.</li>

</ul>

<strong>Algorithm 1 </strong>Minibatch Dependency Parsing

<strong>Input: </strong>sentences, a list of sentences to be parsed and model, our model that makes parse decisions

Initialize partialparses as a list of partial parses, one for each sentence in sentences Initialize unfinishedparses as a shallow copy of partialparses <strong>while </strong>unfinishedparses is not empty <strong>do</strong>

Take the first batchsize parses in unfinishedparses as a minibatch

Use the model to predict the next transition for each partial parse in the minibatch

Perform a parse step on each partial parse in the minibatch with its predicted transition

Remove the completed parses from unfinishedparses <strong>end while</strong>

<strong>Return: </strong>The dependencies for each (now completed) parse in partialparses.

Implement this algorithm in the minibatchparse function in q2parsertransitions.py. You can run basic (not-exhaustive) tests by running python q2parsertransitions.py.

<em>Note: You will need </em><em>minibatchparse to be correctly implemented to evaluate the model you will build in part (h). However, you do not need it to train the model, so you should be able to complete most of part (h) even if </em><em>minibatch</em><em>parse is not implemented yet.</em>

We are now going to train a neural network to predict, given the state of the stack, buffer, and dependencies, which transition should be applied next. First, the model extracts a feature vector representing the current state. We will be using the feature set presented in the original neural dependency parsing paper: <em>A Fast and Accurate Dependency Parser using Neural Networks</em><a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. The function extracting these features has been implemented for you in parserutils. This feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.). They can be represented as a list of integers

[<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>m</sub></em>]

where <em>m </em>is the number of features and each 0 ≤ <em>w<sub>i </sub>&lt; </em>|<em>V </em>| is the index of a token in the vocabulary (|<em>V </em>| is the vocabulary size). First our network looks up an embedding for each word and concatenates them into a single input vector:

<em>x </em>= [<em>L</em><em>w</em>0<em>,</em><em>L</em><em>w</em>1<em>,…,</em><em>L</em><em>w</em><em>m</em>] ∈ R<em>dm</em>

where <em>L </em>∈ R<sup>|<em>V </em>|×<em>d </em></sup>is an embedding matrix with each row <em>L<sub>i </sub></em>as the vector for a particular word <em>i</em>. We then compute our prediction as:

<em>h </em>= ReLU(<em>xW </em>+ <em>b</em><sub>1</sub>)

<em>y</em>ˆ = softmax(<em>hU </em>+ <em>b</em><sub>2</sub>)

(recall that ReLU(<em>z</em>) = max(<em>z,</em>0)). We evaluate using cross-entropy loss:

<em>N<sub>c</sub></em>

<em>J</em>(<em>θ</em>) = <em>CE</em>(<em>y,</em><em>y</em>ˆ) = −<sup>X</sup><em>y<sub>i </sub></em>log ˆ<em>y<sub>i</sub></em>

<em>i</em>=1

To compute the loss for the training set, we average this <em>J</em>(<em>θ</em>) across all training examples.

<ul>

 <li>(4 points, coding) In order to avoid neurons becoming too correlated and ending up in poor local minimina, it is often helpful to randomly initialize parameters. One of the most frequent initializations used is called Xavier initialization<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>.</li>

</ul>

Given a matrix <em>A </em>of dimension <em>m </em>× <em>n</em>, Xavier initialization selects values <em>A<sub>ij </sub></em>uniformly from [], where

Implement the initialization in xavierweightinit in q2initialization.py. You can run basic (nonexhaustive tests) by running python q2initialization.py. This function will be used to initialize <em>W </em>and <em>U</em>.

<ul>

 <li>(2 points, written) We will regularize our network by applying Dropout<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>. During training this randomly sets units in the hidden layer <em>h </em>to zero with probability <em>p<sub>drop </sub></em>and then multiplies <em>h </em>by a constant <em>γ </em>(dropping different units each minibatch). We can write this as</li>

</ul>

<em>h<sub>drop </sub></em>= <em>γ</em><em>d </em>◦ <em>h</em>

where <em>d </em>∈ {0<em>,</em>1}<em><sup>D</sup></em><em><sup>h </sup></em>(<em>D<sub>h </sub></em>is the size of <em>h</em>) is a mask vector where each entry is 0 with probability <em>p<sub>drop </sub></em>and 1 with probability (1 − <em>p<sub>drop</sub></em>). <em>γ </em>is chosen such that the value of <em>h<sub>drop </sub></em>in expectation equals <em>h</em>:

E<em>p<sub>drop</sub></em>[<em>h</em><em>drop</em>]<em>i </em>= <em>h</em><em>i</em>

for all 0 <em>&lt; i &lt; D<sub>h</sub></em>. What must <em>γ </em>equal in terms of <em>p<sub>drop</sub></em>? Briefly justify your answer.

<ul>

 <li>(4 points, written) We will train our model using the Adam<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> Recall that standard SGD uses the update rule</li>

</ul>

<em>θ </em>← <em>θ </em>− <em>α</em>∇<em>θJ</em><em>minibatch</em>(<em>θ</em>)

where <em>θ </em>is a vector containing all of the model parameters, <em>J </em>is the loss function, ∇<em><sub>θ</sub>J<sub>minibatch</sub></em>(<em>θ</em>) is the gradient of the loss function with respect to the parameters on a minibatch of data, and <em>α </em>is the learning rate. Adam uses a more sophisticated update rule with two additional steps<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>.

<ul>

 <li>First, Adam uses a trick called <em>momentum </em>by keeping track of <em>m</em>, a rolling average of the gradients:</li>

</ul>

<em>m </em>← <em>β</em>1<em>m </em>+ (1 − <em>β</em>1)∇<em>θJ</em><em>minibatch</em>(<em>θ</em>) <em>θ </em>← <em>θ </em>− <em>α</em><em>m</em>

where <em>β</em><sub>1 </sub>is a hyperparameter between 0 and 1 (often set to 0.9). Briefly explain (you don’t need to prove mathematically, just give an intuition) how using <em>m </em>stops the updates from varying as much. Why might this help with learning?

<ul>

 <li>Adam also uses <em>adaptive learning rates </em>by keeping track of <em>v</em>, a rolling average of the magnitudes of the gradients:</li>

</ul>

<em>m </em>← <em>β</em>1<em>m </em>+ (1 − <em>β</em>1)∇<em>θJ</em><em>minibatch</em>(<em>θ</em>) <em>v </em>← <em>β</em>2<em>v </em>+ (1 − <em>β</em>2)(∇<em>θJ</em><em>minibatch</em>(<em>θ</em>) ◦ ∇<em>θJ</em><em>minibatch</em>(<em>θ</em>))

√ <em>θ </em>← <em>θ </em>− <em>α </em>◦ <em>m/ </em><em>v</em>

where ◦ and <em>/ </em>denote elementwise multiplication and division (so <em>z </em>◦ <em>z </em>is elementwise squaring) and <em>β</em><sub>2 </sub>is a hyperparameter between 0 and 1 (often set to 0.99). Since Adam divides the update by

√

<em>v</em>, which of the model parameters will get larger updates? Why might this help with learning?

<ul>

 <li>(20 points, coding/written) In py implement the neural network classifier governing the dependency parser by filling in the appropriate sections. We will train and evaluate our model on the Penn Treebank (annotated with Universal Dependencies). Run python q2parsermodel.py to train your model and compute predictions on the test data (make sure to turn off debug settings when doing final evaluation).</li>

</ul>

<strong>Hints:</strong>

<ul>

 <li>When debugging, pass the keyword argument debug=True to the main method (it is set to true by default). This will cause the code to run over a small subset of the data, so the training the model won’t take as long.</li>

 <li>This code should run within 1 hour on a CPU.</li>

 <li>When running with debug=False, you should be able to get a loss smaller than 0.07 on the train set (by the end of the last epoch) and an Unlabeled Attachment Score larger than 88 on the dev set (with the best-performing model out of all the epochs). For comparison, the model in the original neural dependency parsing paper gets 92.5. If you want, you can tweak the hyperparameters for your model (hidden layer size, hyperparameters for Adam, number of epochs, etc.) to improve the performance (but you are not required to do so).</li>

</ul>

<strong>Deliverables:</strong>

<ul>

 <li>Working implementation of the neural dependency parser in py. (We’ll look at, and possibly run this code for grading).</li>

 <li>Report the best UAS your model achieves on the dev set and the UAS it achieves on the test set.</li>

 <li>List of predicted labels for the test set in the file predicted.</li>

</ul>

<ul>

 <li><strong>Bonus </strong>(1 point). Add an extension to your model (e.g., l2 regularization, an additional hidden layer) and report the change in UAS on the dev set. Briefly explain what your extension is and why it helps (or hurts!) the model. Some extensions may require tweaking the hyperparameters in Config to make them effective.</li>

</ul>

<h1>3           Recurrent Neural Networks: Language Modeling</h1>

In this section, you’ll compute the gradients of a recurrent neural network (RNN) for language modeling.

Language modeling is a central task in NLP, and language models can be found at the heart of speech recognition, machine translation, and many other systems. Given a sequence of words (represented as onehot row vectors) <em>x</em><sup>(1)</sup><em>,</em><em>x</em><sup>(2)</sup><em>,…,</em><em>x</em><sup>(<em>t</em>)</sup>, a language model predicts the next word <em>x</em><sup>(<em>t</em>+1) </sup>by modeling:

<em>P</em>(<em>x</em>(<em>t</em>+1) = <em>v</em><em>j </em>| <em>x</em>(<em>t</em>)<em>,…,</em><em>x</em>(1))

where <em>v<sub>j </sub></em>is a word in the vocabulary.

Your job is to compute the gradients of a recurrent neural network language model, which uses feedback information in the hidden layer to model the “history” <em>x</em><sup>(<em>t</em>)</sup><em>,</em><em>x</em><sup>(<em>t</em>−1)</sup><em>,…,</em><em>x</em><sup>(1)</sup>. Formally, the model<a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a> is, for <em>t </em>= 1<em>,…,n </em>− 1:

<em>e</em>(<em>t</em>) = <em>x</em>(<em>t</em>)<em>L h</em><sup>(<em>t</em>) </sup>= sigmoid <em>y</em>ˆ<sup>(<em>t</em>) </sup>= softmax

where <em>h</em><sup>(0) </sup>= <em>h</em><sub>0 </sub>∈ R<em><sup>D</sup></em><em><sup>h </sup></em>is some initialization vector for the hidden layer and <em>x</em><sup>(<em>t</em>)</sup><em>L </em>is the product of <em>L </em>with the one-hot row vector <em>x</em><sup>(<em>t</em>) </sup>representing the current word. The parameters are:

<em>L </em><sup>∈ </sup>R|<em>V </em>|×<em>d </em><em>H </em><sup>∈ </sup>R<em>D</em><em>h</em>×<em>D</em><em>h </em><em>I </em><sup>∈ </sup>R<em>d</em>×<em>D</em><em>h </em><em>b</em><sub>1 </sub><sup>∈ </sup>R<em>D</em><em>h </em><em>U </em><sup>∈ </sup>R<em>D</em><em>h</em>×|<em>V </em>| <em>b</em><sub>2 </sub><sup>∈ </sup>R|<em>V </em>|                                                                        (1)

where <em>L </em>is the embedding matrix, <em>I </em>the input word representation matrix, <em>H </em>the hidden transformation matrix, and <em>U </em>is the output word representation matrix. <em>b</em><sub>1 </sub>and <em>b</em><sub>2 </sub>are biases. <em>d </em>is the embedding dimension, |<em>V </em>| is the vocabulary size, and <em>D<sub>h </sub></em>is the hidden layer dimension.

The output vector <em>y</em>ˆ<sup>(<em>t</em>) </sup>∈ R<sup>|<em>V </em>| </sup>is a probability distribution over the vocabulary. The model is trained by minimizing the (un-regularized) cross-entropy loss:

|<em>V </em>|

<em>J</em>(<em>t</em>)(<em>θ</em>) = <em>CE</em>(<em>y</em>(<em>t</em>)<em>,</em><em>y</em>ˆ(<em>t</em>)) = −X<em>y</em><em>j</em>(<em>t</em>) log ˆ<em>y</em><em>j</em>(<em>t</em>)

<em>j</em>=1

where <em>y</em><sup>(<em>t</em>) </sup>is the one-hot vector corresponding to the target word (which here is equal to <em>x</em><sup>(<em>t</em>+1)</sup>). We average the cross-entropy loss across all examples (i.e., words) in a sequence to get the loss for a single sequence.

<ul>

 <li>(5 points, written) Conventionally, when reporting performance of a language model, we evaluate on <em>perplexity</em>, which is defined as:</li>

</ul>

PP

<em>P</em>¯(<em>x</em>(pred+1) = <em>x</em>(<em>t</em>+1) | <em>x</em>(<em>t</em>)<em>,…,</em><em>x</em>

i.e. the inverse probability of the correct word, according to the model distribution <em>P</em>¯. Show how you can derive perplexity from the cross-entropy loss (<em>Hint: remember that </em><em>y</em><sup>(<em>t</em>) </sup><em>is one-hot!</em>), and thus argue that minimizing the (arithmetic) mean cross-entropy loss will also minimize the (geometric) mean perplexity across the training set. <em>This should be a very short problem – not too perplexing!</em>

For a vocabulary of |<em>V </em>| words, what would you expect perplexity to be if your model predictions were completely random (chosen uniformly from the vocabulary)? Compute the corresponding cross-entropy loss for |<em>V </em>| = 10000.

<ul>

 <li>(8 points, written) Compute the gradients of the loss <em>J </em>with respect to the following model parameters at a single point in time <em>t </em>(to save a bit of time, you don’t have to compute the gradients with the respect to <em>U </em>and <em>b</em><sub>1</sub>):</li>

</ul>

where <em>L<sub>x</sub></em>(<em><sub>t</sub></em><sub>) </sub>is the row of <em>L </em>corresponding to the current word <em>x</em><sup>(<em>t</em>)</sup>, and  denotes the gradient for the appearance of that parameter at time <em>t </em>(equivalently, <em>h</em><sup>(<em>t</em>−1) </sup>is taken to be fixed, and you need not backpropagate to earlier timesteps just yet – you’ll do that in part (c)).

Additionally, compute the derivative with respect to the <em>previous </em>hidden layer value:

<ul>

 <li>(8 points, written) Below is a sketch of the network at a single timestep:</li>

</ul>

Draw the “unrolled” network for 3 timesteps, and compute the backpropagation-through-time gradients:

<em>∂J</em>(<em>t</em>)

<em>∂</em><em><sup>L</sup></em><em><sub>x</sub></em>(<em>t</em>−1)

where  denotes the gradient for the appearance of that parameter at time (<em>t </em>− 1). Because parameters are used multiple times in feed-forward computation, we need to compute the gradient for each time they appear.

You should use the backpropagation rules from Lecture 5<a href="#_ftn7" name="_ftnref7"><sup>[7]</sup></a> to express these derivatives in terms of error term  computed in the previous part. (Doing so will allow for re-use of expressions for <em>t </em>− 2, <em>t </em>− 3, and so on).

<em>Note that the true gradient with respect to a training example requires us to run backpropagation all the way back to t </em>= 0<em>. In practice, however, we generally truncate this and only backpropagate for a fixed number τ </em>≈ 5 − 10 <em>timesteps.</em>

(d) (4 points, written) Given <em>h</em><sup>(<em>t</em>−1)</sup>, how many operations are required to perform one step of forward propagation to compute <em>J</em><sup>(<em>t</em>)</sup>(<em>θ</em>)? How about backpropagation for a single step in time? For <em>τ </em>steps in time? Express your answer in big-O notation in terms of the dimensions <em>d</em>, <em>D<sub>h </sub></em>and |<em>V </em>| (Equation 1).

What is the slow step?

<strong>Bonus </strong>(1 point, written) Given your knowledge of similar models (i.e. word2vec), suggest a way to speed up this part of the computation. Your approach can be an approximation, but you should argue why it’s a good one. The paper “Extensions of recurrent neural network language model” (Mikolov, et al. 2013) may be of interest here.

<a href="#_ftnref1" name="_ftn1">[1]</a> Chen and Manning, 2014, http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf

<a href="#_ftnref2" name="_ftn2">[2]</a> This is also referred to as Glorot initialization and was initially described in <a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf">http://jmlr.org/proceedings/papers/ </a><a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf">v9/glorot10a/glorot10a.pdf</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> Srivastava et al., 2014, https://www.cs.toronto.edu/ hinton/absps/JMLRdropout.pdf

<a href="#_ftnref4" name="_ftn4">[4]</a> Kingma and Ma, 2015, https://arxiv.org/pdf/1412.6980.pdf

<a href="#_ftnref5" name="_ftn5">[5]</a> The actual Adam update uses a few additional tricks that are less important, but we won’t worry about them for this problem.

<a href="#_ftnref6" name="_ftn6">[6]</a> This model is adapted from a paper by Toma Mikolov, et al. from 2010: <a href="http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf">http://www.fit.vutbr.cz/research/groups/ </a><a href="http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf">speech/publi/2010/mikolov_interspeech2010_IS100722.pdf</a>

<a href="#_ftnref7" name="_ftn7">[7]</a> <a href="https://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture5.pdf">https://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture5.pdf</a>