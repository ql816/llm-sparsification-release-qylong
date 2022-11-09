## Choice of Models
I chose the following models:
Encoder Only: BERT
Decoder Only: GPT-2
Encoder-Decoder: BART

## Model Parameter Value Distribution

I generated three plots each represent the distribution of the plots for the three models. One can observe that a great portion of the parameters in the model is close to zero.

![Alt](/plots/bart-num-dist.png)
![Alt](/plots/bert-num-dist.png)
![Alt](/plots/gpt-2-num-dist.png)

The following plots show the parapmter value distribution of each layer. It shows that the layer-wise distributions of parameter values do not differ a lot, with many paramter's value close to zero.
![Alt](/plots/bart-layer-dist.png)
![Alt](/plots/bert-layer-dist.png)
![Alt](/plots/gpt-2-layer-dist.png)

## Model Sparsification

I pruned the model based using pytorch pruning with L1 normalization. PyTorch only support unstructed pruning, and therefore the number of parameters will not change even if we pruned the model. It is because what unstructed pruning does is setting a certain amount of weights of the paramters to 0, which means we will still have the same number of paramters. That being said, the size of the model will not change, and so does the speed. 


## Memory and Speed Analysis
The Speed and Memory of the models are evaluted by Huggieface benchmark tool for data of Sequence Length of 64 and batch size of 128. The results are shown in the following table:
| Model | Memory in MB | Time in S|
|--|--| --|
| Bart | 2388 | 0.4828|
| Bart - 0.1| 2388 |0.4962|
| Bart - 0.5| 2388 |0.5252|
| Bart - 0.9| 2388 |0.5252|
| Bart - 0.95|2388  |0.5399|
| Bart - 0.99| 2388 |0.5439|
| GPT-2 | 4584 |0.434|
| GPT-2 - 0.1| 4584|0.5484|
| GPT-2 - 0.5| 4584 |0.4636|
| GPT-2 - 0.9| 4584 |0.4636|
| GPT-2 - 0.95| 4584 |0.4744|
| GPT-2 - 0.99| 4584 |0.4722|
| Bert | 2624 |0.6581|
| Bert - 0.1| 2624 |0.6698|
| Bert - 0.5| 2624 |0.72|
| Bert - 0.9| 2624 |0.72|
| Bert - 0.95| 2624 |0.7134|
| Bert - 0.99| 2624 |0.7159|

![Alt](/plots/required_memory.png)

![Alt](/plots/required_time.png)

We can tell that the memory space required does not change with the sparsity of the model, as we are using unstructed pruning. The times required vary slight. However, with a model of the same size, there is no significant improvement on speed with unstructed pruning.
## Performance Analysis
First, I tested the models on Huggieface's [language modeling example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling%29). The perplexity appears to vary slight by sparisity, yet no apparent difference is observed. 

Secondly, I tested the models on Huggieface's [text-classification example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). 

It seems that the accuracy of text classification decreases with the model get more sparsed, as shown in the plot. Since as the models are sparsed, more weights are set to 0, it makes sense the accuracy drops. The trend is as shown below in the plots.

![Image](/plots/Text_Classification_Accuracy.png)

## Challenges of Sparsification
In this assignment, I used unstructed pruning, which does not help with the speed and size of the model. Yet, in structed pruning, this issue could be resolved. However, as we see in the performance analysis, pruning a model makes it less accurate. It seems that currently there has been no effective way to both reduce the size of a model and maintain its accuracy. 
