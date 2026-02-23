# tinyLM
A really really smoll and stoopid text generation model :3

# Installation
- You can clone this project and install the requirement since it is relatively light weight

```
git clone https://github.com/shionxva/tinyLM
cd tinyLM
pip install -r requirements.txt
```

- Or run it on google colab with the .jpynb file

# How it works?

## Tokenizer
- This is a reprocessing layer that simply split text by white space and assign it with an interger IDs

- `output_sequence_length=CONTEXT_WIN` : apply "0" padding so that every seqs are the same length (20 in this case)

## Data prep

slicing the last word for input and last word for output
-> this will then force the model to learn the connection of words and predict the next token :0

## Perplexity metric
*overall it shows how many choises is the model confused between*
- `crossentropy(true,pred)` cross entropy loss with true_labels predicted value
- then `perplexity` = e^`crossentropy`

## Token Position Embedding
Once the text is tokenized, each token needs to be represented in a way that captures not just the token but also its meaning and position (relation to other tokens)

*This is how it works visually:*

```
Input tokens:     ["the", "dog", "runs"]  →  IDs: [4, 27, 13]
Token embeddings:  [row_v4,  ...  ,row_v27, ...  ,row_v13  ]  
Position embeddings:[row_p0,  ...  ,row_p1,  ...  ,row_p2  ]  
Output:           [v4+p0, v27+p1, v13+p2]
```

## Transformer blocks
Overall structure:
```
Input -> Multi-head attention -> Add Residual -> LayerNorm 
-> Feed Forward ->Add Residual -> LayerNorm -> Output
```
- `multi-head attention`: each head works on an equal slice of the embedding.

- `feed forward network` is a small 2 layers network:

It first expands the dimension to feed_forward, applies ReLU, then squishes back down to embed_dim

- `layer norm and dropout`: norm stabilizes training and dropout randomly zeros out 10% of values during training to prevent overfitting.

*notes: self-attention is basically seeing how other words are relevant to the current word*

## miniLM

Now we put everything into this `miniLM` class + a text generation function

a little overview of our smoll brain:
- `embedding layer` converts token IDs into vectors
- `transformer blocks` to process the sequence
- `dense out` gives us the logits or "score" for each word in the vocab

text generation process:
- tokenize input (also remove paddings)
- ran the loop for a fixed length
  - `tokens[-CONTEXT_WIN:]` only have context of the previous word
  - run the model -> get logits and apply temperature
  - apply top_k to get the top k-th prediction
  - softmax to convert in into probability
  -> randomly pick a word

# Resources and references
I learned this from @lachinemearning.com on tiktok and also wanted to try to make this dumb AI.

Additional references to deepen my understanding:

 - [reddit - embedding/tokenization](https://www.reddit.com/r/learnmachinelearning/comments/1cs29kn/confused_about_embeddings_and_tokenization_in_llms/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)

- [statquest - neural network](https://www.youtube.com/@statquest)
