import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import pickle

MAX_VOCAB = 1000
CONTEXT_WIN = 20
EMBED_WIN = 64
HEADS = 2
FEED_FOWARD = 128 #4x embed_win
TRANSFORMER_BLOCKS = 2
BATCH_SIZE = 16
EPOCHS = 70

train_text = [
"Machine Learning is very hard",
"I use Arch btw",
"I need a job",
"GitHub is goated",
"Saber best waifu :3",
"Skibidi Toilet",
"This is a test"
]

tokenizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=CONTEXT_WIN,
    standardize=None,
    split="whitespace"
)
tokenizer.adapt(train_text) #build vocab
vocab = tokenizer.get_vocabulary() #convert to int IDs

print(vocab)

#next token prediction
def prep_data(train_text):
  seq = tokenizer(train_text)
  x_input = seq[:, :-1]
  y_target = seq[:, 1:]

  return x_input, y_target

x_train, y_train = prep_data(train_text)

def perplexity(true,pred):
  return tf.exp(tf.reduce_mean(losses.sparse_categorical_crossentropy(true,pred)))

class TokenPositionEmbedding(layers.Layer):
  def __init__(self, context_win, max_vocab, embed_dim, **kwargs) -> None:
    super().__init__()
    self.token_embed = layers.Embedding(input_dim = max_vocab, output_dim=embed_dim)
    self.position_embed = layers.Embedding(input_dim = context_win, output_dim=embed_dim)

  def call(self, x):
    context_win = tf.shape(x)[-1]
    positions = self.position_embed(tf.range(start=0, limit=context_win, delta=1))
    return self.token_embed(x) + positions

class TransformerBlock(layers.Layer):
  def __init__(self, embed_dim, heads, feed_forward, **kwargs) -> None:
    super().__init__()
    self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim // heads)
    self.feed_foward_net = models.Sequential([layers.Dense(feed_forward, activation="relu"), layers.Dense(embed_dim)])
    self.norm1 = layers.LayerNormalization(epsilon=1e-4)
    self.norm2 = layers.LayerNormalization(epsilon=1e-4)
    self.drop1 = layers.Dropout(0.1)
    self.drop2 = layers.Dropout(0.1)

  def call(self, inputs, training = False):
    #Self Attention                   query    key/value
    attention_output = self.attention(inputs, inputs, use_causal_mask=True)
    attention_output = self.drop1(attention_output, training=training)
    output = self.norm1(inputs + attention_output)

    #Feed Forward Network
    feed_forward_output = self.feed_foward_net(output)
    feed_forward_output = self.drop2(feed_forward_output, training=training)

    return self.norm2(output + feed_forward_output)

class miniLM(models.Model):
  def __init__(self, context_win, max_vocab, embed_dim, heads, feed_forward, blocks, **kwargs) -> None:
    super().__init__(**kwargs)
    self.embed_layer = TokenPositionEmbedding(context_win, max_vocab, embed_dim)
    self.transformer_blocks = [TransformerBlock(embed_dim, heads, feed_forward) for _ in range(blocks)]
    self.dense_out = layers.Dense(max_vocab)

  def call(self, inputs, training=False):
      x = self.embed_layer(inputs)            # (batch, seq_len, embed_dim)
      for block in self.transformer_blocks:
        x = block(x, training=training)       # (batch, seq_len, embed_dim)
      return self.dense_out(x)                # (batch, seq_len, max_vocab)


  def gen(model, prompt, length = 6, temperature= 1.0, top_k = 5):
    input_tensor = tokenizer([prompt])
    tokens = [token for token in input_tensor.numpy()[0] if token != 0]

    gen_text = prompt
    for _ in range(length):
      context_token = tokens[-CONTEXT_WIN:]
      input_data = tf.convert_to_tensor([context_token])

      preds = model(input_data, training=False)
      next_logits = preds[0, -1, :]
      next_logits /= (temperature + 1e-7)
      top_val,  top_idx = tf.math.top_k(next_logits, k=top_k)
      top_prob = tf.nn.softmax(top_val).numpy()

      next_idx = np.random.choice(top_idx.numpy(), p=top_prob)
      if next_idx == 0 and len(tokens) >0:
        next_idx = top_idx.numpy()[1]

      tokens.append(next_idx)
      gen_text += " " + vocab[next_idx]

    return gen_text

def main():
  skibidi_model = miniLM(CONTEXT_WIN, len(vocab), EMBED_WIN, HEADS, FEED_FOWARD, TRANSFORMER_BLOCKS)
  skibidi_model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[perplexity])
  skibidi_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

  while True:
    test_input = input("Enter a prompt (or 'quit'): ")
    if test_input.lower() == "quit":
        break
    print(skibidi_model.gen(test_input))
    
  save = input("Save model? (y/n): ")
  if save.lower() == "y":
    skibidi_model.save("skibidi_model.keras")
    skibidi_model.save_weights("skibidi_model.weights.h5")
    with open("vocab.pkl", "wb") as f:
      pickle.dump(vocab, f)

if __name__ == "__main__":
  main()