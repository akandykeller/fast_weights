# Fast-Weights for NLP

Implementations of [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258) and [Gated Fast-Weights for On-The-Fly Neural Program Generation](http://metalearning.ml/papers/metalearn17_schlag.pdf) for Question & Answering over the 20 bAbI Q&A tasks. Implementations are in tensorflow and reuse existing framework of [End2End Memory Networks](https://github.com/akandykeller/memn2n) for [bAbI](http://arxiv.org/abs/1502.05698) dataset processing.

The novelty of this work lies in the application of fast-weights to NLP tasks as a replacement for Memory Network type architectures. It shows that associative memories as enabled by fast-weights are competitive with the explicit memory structures of End-to-End Memory Networks. Additionally, this work expands upon previous implementations by incorporating the fast-weights into multi-layered and bi-directional RNN models. Furthermore, I believe this to be the first publicly available implementation of the Gated Fast-Weights model.
#### Fast-Weights RNN
![Fast-Weights picture](http://i.imgur.com/DCznSf4.png)
#### Gated Fast-Weights
![Gated-Fast Weights](https://i.imgur.com/7i4Y59u.png)


### Get Started

```
pip install --upgrade tensorflow
pip install sklearn

wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz

# Running training & testing of basic FW-RNN on task 1
python single.py --data_dir tasks_1-20_v1-2/en/ --task 1

# To visualize attention (using the alternate equivalent model implementation)
python single_alt.py --data_dir tasks_1-20_v1-2/en/ --task 1 --show_attn True

# Running training & testing of multi-layer bi-directional FW-RNN on task 1
python single_multi_layer.py --data_dir tasks_1-20_v1-2/en/ --task 1

# Running training & testing of Gated FW-RNN on task 1
python single_gated_fw.py --data_dir tasks_1-20_v1-2/en/ --task 1
```

### Notes

Currently all models use the same sentence and question encoders as the End-to-End Memory Network (Bag-of-words w/ position encoding). These seem to work well enough for the simple babi sentences, so I'm not too worried about that being the performance bottleneck here. 

The FW-RNN currently predicts answer words by:
  * Encoding all story & question sentences with BoW encoder
  * Run Fast-Weights RNN over all memory sentences followed by question sentence
  * Project final hidden state of the network through final weight matrix and softmax over vocabulary

The main problem right now seems to be overfitting. The network easily achieves 100% training accuracy while validation accuracy is frequently left at 50% without sufficient regularization. Adding dropout and decreasing the fast-weights learning rate (eta) from 0.5 to 0.25 helped this slightly, but the network still takes 1000 epochs to reach 90% accuracy, when the E2EMemNN can reach 100% in as few as 10 epochs. 

Gated Fast-Weights don't seem to be performing significnatly better than orignal fast-weights, more expeiments to come. Results not yet shown here. 

Next steps involve application of these models to larger and more complex datasets / NLP tasks such as reading comprehension.

### Single Task Results On bAbI-1k

Below are the best results for each task individually selected from a significant grid search. Model hyper-parameters between tasks vary.

Published End-to-End Memory Network & LSTM reference accuracy is shown for comparison. The higest accuracy for each task is bolded. Within 1% both are bolded.

Task  |  LSTM Accuracy  |  MemN2N Accuracy PE-LS-RN | Fast-Weights Accuracy  |
------|-----------------|---------------------------|------------------------|
1     |  0.5            |  **1.0**                  |  **0.997**             |
2     |  0.2            |  **0.917**                |  0.333                 |   
3     |  0.2            |  **0.597**                |  0.359                 |   
4     |  0.61           |  0.972                    |  **1.0**               |
5     |  0.7            |  **0.869**                |  **0.856**             |   
6     |  0.48           |  **0.924**                |  0.826                 |   
7     |  0.49           |  **0.827**                |  0.783                 |   
8     |  0.45           |  0.900                    |  **0.922**             |   
9     |  0.64           |  **0.868**                |  0.759                 |   
10    |  0.44           |  **0.849**                |  0.771                 |   
11    |  0.62           |  **0.991**                |  **1.0**               |   
12    |  0.74           |  **0.998**                |  **0.999**             |   
13    |  0.94           |  **0.996**                |  0.953                 |   
14    |  0.27           |  **0.983**                |  0.794                 |   
15    |  0.21           |  **1.0**                  |  0.841                 |
16    |  0.23           |  **0.987**                |  0.492                 |   
17    |  0.51           |  0.4                      |  **0.592**             |
18    |  0.52           |  0.889                    |  **0.950**             |
19    |  0.08           |  **0.172**                |  0.116                 |   
20    |  0.91           |  **1.0**                  |  **0.989**             |



### Single Task Results On bAbI-10k

Below are the best results for each task individually selected from the grid search using 10k training examples. Model hyper-parameters between tasks vary.

Published End-to-End Memory Network & LSTM reference accuracy is shown for comparison. The higest accuracy for each task is bolded. Within 1% both are bolded.

Task  |  LSTM Accuracy  |  MemN2N Accuracy PE-LS-RN | Fast-Weights Accuracy  |
------|-----------------|---------------------------|------------------------|
1     |  1.0            |  **1.0**                  |  **1.0**               |
2     |  0.181          |  **0.997**                |  0.906                 |   
3     |  0.169          |  **0.979**                |  0.470                 |   
4     |  0.998          |  **1.0**                  |  0.749.                |
5     |  0.988          |  **0.992**                |  **0.996**             |   
6     |  0.482          |  **0.999**                |  0.969                 |   
7     |  0.751          |  **0.98**                 |  0.968                 |   
8     |  0.659          |  **0.991**                |  **1.0**               |   
9     |  0.798          |  **0.997**                |  0.993                 |   
10    |  0.699          |  **1.0**                  |  0.986                 |   
11    |  0.897          |  **0.999**                |  **1.0**               |   
12    |  0.766          |  **1.0**                  |  **1.0**               |   
13    |  0.939          |  **1.0**                  |  **1.0**               |   
14    |  0.19           |  **0.999**                |  0.962                 |   
15    |  0.213          |  **1.0**                  |  0.715                 |
16    |  0.481          |  **0.482**                |  0.455                 |   
17    |  0.499          |  **0.814**                |  0.652                 |
18    |  0.932          |  **0.947**                |  0.906                 |
19    |  0.097          |  **0.977**                |  0.464                 |   
20    |  0.979          |  **1.0**                  |  **1.0**               |
