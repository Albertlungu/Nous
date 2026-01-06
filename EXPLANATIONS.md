# Explanations of pieces of my code for contributors or people who are simply curious.
This file serves users who want a more in-depth coverage of what my code does, and how it does it. It is most useful to contributors.

**Enjoy the ride!**

## TikToken tokenizer (recommended)
My program uses OpenAI's TikToken library, which has pre-trained tokenizers such as `r50k_base`, which tend to be much mode efficient than the custom ones.

```python
Load tokenizer - using TikToken
tokenizer = TikToken()
print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")
```

And comment out:
```python
with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    tokenizer._ensure_vocab()
```

## How to train tokenizer on your own dataset
I recommend using the custom tokenizer for experimentation and research use only, since it tends to be unstable and slow. Keep this in mind both during training and use.

To train the tokenizer on your own dataset, you first want to make sure it is cleaned of "Instructions", or "Responses", or anything of the sort. This is so that the model doesn't get confused by that noise.

Before training, load your dataset into a .txt file, and take a look at the format. If it has any of these labels, follow the code for the labels already covered in `clean_text` (lines 281 to 307):

```python
def clean_text(file_path):
    """
    Read dataset and strip out labels.
    Returns clean text with only the actual content.
    """
    clean_text = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Remove the labels
            if line.startswith("Instruction:"):
                line = line.replace("Instruction:", "").strip()
            elif line.startswith("Input:"):
                line = line.replace("Input:", "").strip()
            elif line.startswith("Output:"):
                line = line.replace("Output:", "").strip()
            elif line.startswith("Response:"):
                line = line.replace("Response:", "").strip()
            elif line.startswith("Context:"):
                line = line.replace("Context:", "").strip()

            # Keep the line if it has content
            if line:
                clean_text.append(line)

    return " ".join(clean_text)
```

To make your own "flag" simply copy paste this block:
```python
elif line.startswith("Input:"):
  line = line.replace("Input:", "").strip()
```
And replace `"Input"` with the label that you would like to flag, both in the `elif` statement and in the `.replace()` parameter. The rest stays the same.

**Next**, you run the `main()` function to actually tokenize.

**Finally**, you have to run the `tokenize_training_data(path)` function to pre-tokenize the your training data. This is so that the model doesn't have to do this itself during training, which will save some time.

>_Side note_, expect training data to take a while, especially with a large dataset. Currently, The device used is an M4 MacBook Air, and for a vocab size of 32k and a training sample of 145k lines, it is taking ~5hrs.

## Tokenizer Details
- Implements **Byte Pair Encoding (BPE)** algorithm to compress all words into subword tokens.
- Starts with a base vocab size of 256
- Iteratively merges the most frequent adjacent byte pairs (letter or character pairs) until max vocab size is reached

To learn more about BPE, I highly recommend [this video by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) 
- **MASSIVE** thanks to him for his amazing instructional videos.

#### Example Usage:
```python
with open("artifacts/tokenizer.pkl", "rb") as f:
  tokenizer = pickle.load(f)
  tokenizer._ensure_vocab()

text = "hello world"
token_ids = tokenizer.encode(text)
print(token_ids)  # e.g., [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

decoded_text = tokenizer.decode(token_ids)
print(decoded_text)  # "hello world"
```

#### Example of how it works (from [Wikipedia.org](https://en.wikipedia.org/wiki/Byte-pair_encoding#:~:text=The%20original%20BPE%20algorithm%20operates,the%20target%20text%20effectively%20compressed)):

Suppose the data to be encoded is:
```
aaabdaaabac
```
  The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:

```
ZabdZabac
Z=aa
```
  Then the process is repeated with byte pair "ab", replacing it with "Y":

```
ZYdZYac
Y=ab
Z=aa
```
  The only literal byte pair left occurs only once, and the encoding might stop here. Alternatively, the process could continue with recursive byte-pair encoding, replacing "ZY" with "X":

```
XdXac
X=ZY
Y=ab
Z=aa
```
  This data cannot be compressed further by byte-pair encoding because there are no pairs of bytes that occur more than once.

To decompress the data, simply perform the replacements in the reverse order.

**Source**: [Wikipedia](https://en.wikipedia.org/wiki/Byte-pair_encoding#:~:text=The%20original%20BPE%20algorithm%20operates,the%20target%20text%20effectively%20compressed)

## Embeddings
My file for the embeddings can be found in `src/embeddings/embeddings.py`. I have a more comprehensive markdown file on how my embeddings work, so you can check that out at `concepts/positional_encoding.md`. However, this section will give a *higher level* overview on embeddings in general and positional encoding.

### What are embeddings?
___
In simple terms, embeddings are vector quantities that are attributed to each token id, giving every single token a numerical representation. We have to represent tokens and words in the english language through numbers because that's what computers understand. 
>For example, if I tell a normal computer, in plain english: "Eating is the action of putting (food) into the mouth and chewing and swallowing it," the computer will just yell at me for using incorrect syntax. If I tell it: "`eating = [0.01, -1.02, -3.5e-4, ..., 6e7, -9e-2]`, it will understand!

### But how does my code do it?
___

In my `__init__` method of `embeddings.py`, the most relevant attribute there is the following:
```python
self.key = jax.random.PRNGKey(0)
self.embeddings = jax.random.normal(self.key, (self.vocab_size, self.embedding_dim)) * jnp.sqrt(1.0/self.vocab_size)
```

Here, `key` is effectively JAX's overly complicated way to do NumPy's `np.random.randn`. It creates a random floating point number.
In the next line, I declare embeddings as being a random selection of floating point numbers. The reason I do this is so that my model has a starting point. It does not start with the training and then make the embeddings from there, but the opposite.
- First, I generate a matrix (called embedding matrix) of shape (vocab_size, embedding_dim) full of embeddings. Each of my embeddings contain `embedding_dim` numbers inside of them.
> For example, if I have `embedding_dim = 256` and `vocab_size = 32000`, I would have 32000 embeddings, each with 256 random numbers inside of it.
>
For the next step, we have to understand batching, and how it works.

_To note:_ a batch is a collection of sequences, used to make training faster and more efficient
A single batch of token IDs has a shape of `(batch_size, sequence_length a.k.a max_seq_len)`
> For example, if I have a batch size of 8, and a maximum sequence length of 512, my batch would have 8 sequences of 512 token IDs.
- Next, I feed this into my embedding matrix, and I look for the specific embedding of a token ID. This works because the embedding of a token of number "x" is simply the embedding at index "x", meaning:
```python
embedded = embeddings[token_ids]
```
- After I do this, I replace each token ID with its vector from the embedding matrix. This means that that specific token ID is no longer a simple number, such as "6741", but is now a vector quantity, such as `[0.01, -1.02, -3.5e-4, ..., 6e7, -9e-2]`
  - This produces a shape of (batch_size, max_seq_len, embedding_dim)
  - > Meaning: if I have the same parameters as before, my batch after embedding lookup would have 8 sequences of 512 tokens, where each token is represented by 256 numbers.

- Once that is done, I move on to the paddings. A padding token is a specific token ID that I choose, which is most often simply token 0. `pad_token_id = 0`. Padding tokens appear when sequences are different lengths, so that the batch is nice and rectangular. These tokens are then ignored by the model through an attention mask.
- > For example, if I had a sequence saying "I like to eat food", and another that said, "But I should really look into slimming down", and a third saying "But I love food too much, I cannot commit such a crime", assume the third string to have `seq_len=256`, the second string to have `seq_len=232` (just a random number, it doesn't matter), and the first string is of `seq_len=167` (again, random number). My model likes it when these sequences are all the same length, so that it doesn't put more weight and emphasis onto the longer one. So, what you do, is you add a bunch of padding ids to the shorter ones (assume `max_seq_len=256`) to make them 256 numbers.
- Finally, we move on to the "End of Sequence" id (EOS for short). This, unlike the padding id, should not be ignored by my model. The EOS token, while masked as to not affect weights and actual output, tells my model that the sequence is over, so it knows when a line ends, or when a sentence ends. The EOS token is usually set to the vocab size
  *- This is applied later in the transformer*
>> To see details on positional encoding, see `concepts/positional_encoding.md`

## Transformer
This project is based on a multi-head attention transformer architecture.
### What is the transformer architecture?
A transformer-based model is made up of either an encoder or decoder, or both. This model is based on the decoder architecture, mimicking ChatGPT. In the following graph, the encoder architecture is on the left, and the decoding architecture is on the right

This description will be focusing on the decoder architecture, since that is what is used in this model.

---
### Is Attention all You Need?
Using the famous paper from Google Mind, [Attention Is All You Need](https://arxiv.org/pdf/1706.03762), I have created an attention model in the ways which are described in this documentation.

This mode, as it currently stands, has architecture implemented for both single head and multi head attention. These can be explored in the files `src/transformer/single_head_attention.py` and `src/transformer/multi_head_attention.py` respectively. The single head attention is a remnant from the branch using NumPy, and does not use JAX or GPU-based processing.

Multi head attention, in the way I implemented it here, uses a set of four learnable parameters called **weights**. These include:
```python
self.W_Q = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
self.W_K = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
self.W_V = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
self.W_O = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
```
- `W_Q` represents **query weights**, which turns the user input into a "query" vector that represents what the current token wants to find. i.e., understands what the user is asking through learnable weights.
- `W_K` represents **key weights**, which represent the information that every token offers. They pull and transform the meaning from embeddings into a vector quantity.
  - You may be asking yourself *Wait, aren't embeddings already vectors? If so, then why do I need to transform them further into something the computer can understand?*
  - Well, they are transformed because, in multi-head attention, each head deals with a different part of the token's meaning, while the full embedding represents **everything** about the token itself.
- `W_V` represents **value weights**, transform token embeddings into value vectors. Values contain the information that will be used when a token is looked at. 
  - The **embedding** contains the complete definition of a token. **W_V** learns to extract aspects most useful for downstream processing. **Query and key weights** determine which tokens to attend to (attention scores), while value weights determine what information to retrieve from those tokens.
- Finally, `W_O` represents **output weights**, which are the final projection that comes after attention, where it's already combined information.
  - Output weights combine the information from all heads in order to pass this to the transformer block. 

All weights except for `W_O` each have shape `(embedding_dim, embedding_dim)`, which is more efficient than `(head_dim, head_dim)`
>> Again, just like any other learnable parameters, these keys are declared as random vector values, and modified later.

#### The forward method:
```python
Q = x @ params['W_Q']
K = x @ params['W_K']
V = x @ params["W_V"]
```

Here, I perform the linear transformation of the input into queries, keys, and values, here is the breakdown of the code and what it does:

- `x` is the input tensor with shape `(batch_size, seq_len, embedding_dim)` from the first transformer block.
- `params['W_Q']` is the query weight matrix, which is explained above (shape `(embedding_dim, embedding_dim`)
  - Params is a dictionary containing the weights declared in the `get_params_and_grads` function
- `@` is the matrix multiplication symbol
- `Q` is the output query tensor (jnp.jnparray) of shape `(batch_size, seq_len, embedding_dim)`

The same logic is true for the rest of the tensors (K and V)

**What is matrix mutliplication?**
Take 2 matrices, A and B:
- A has shape `(m x n)`
- B has shape `(n x p)`
- If `A[1]` ≠ `B[0]`, the matrix multiplication does not work

Each entry of the new matrix, C, is built by lining up the `i`-th row of A with the `j`-th column of B, mutliplying those numbers, and then summing them up. If you're a math person, the formula is below:

$$
C[i,j] = \sum_k{A[i, k] \cdot B[k, j]}
$$

Where:
- $i$ is the row index of A (therefore the row index of C), and $0 \leq i \leq m-1$
- $k$ is the column index of A and the row index of B, and $0 \leq k \leq n-1$
  - This dimension must match across both arrays (inner dimension)
- $j$ is the column index of B (therefore the column index of C), and $0 \leq j \leq p-1$

**For example:**

```math
A = \begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12
\end{bmatrix}

B = \begin{bmatrix}
10 & 11 \\
20 & 21 \\
30 & 31 \\
40 & 41
\end{bmatrix}
```

These are denoted with:

```math
A \in \mathbb{Z}^{3x4}, \quad
B \in \mathbb{Z}^{4x2}
```

What happens under the hood here for `C[0,0]` ($k=4$):

$C=AB$ will have shape $3x2$, so the entry $C[0,0]$ is built by taking:
- row $0$ of $A:[1, 2, 3, 4]$
- column $0$ of $B:[10, 20, 30, 40]$ 
  
And applying dot product:

```math
C_{0,0} = \sum_{k=0}^3 A_{0,k} \cdot B_{k,0}
```

Expanding it term by term gives:

```math
C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0} + A_{0,2}B_{0,2} + A_{0,3}B_{3,0} \\

\text{Plugging in the values: } C_{0,0} = (1)(10) + (2)(20) + (3)(30) + (4)(40) \\

C_{0,0} = 10 + 40 + 90 + 160 \\

C_{0,0} = 300
```

This pattern continues for all of the other indices of $C$

**Afterwards**, the weights are reshaped to be in accordance with the number of attention heads, using the `.reshape` method in Python, which simply rearranges the same elements into a different structure. 

When you reshape from `[batch, seq_len, embedding_dim]` to `[batch, num_heads, seq_len, embedding_dim // num_heads]`, you effectively create `num_heads` separate vectors for each token, and each vector is given to each head, where instead of for example, head 1, having to deal with all the floating point numbers inside of `embedding_dim`, it only has to deal with `embedding_dim // num_heads`, allowing for "specialization" of each head.


**For example**:
```python
# Original array: 1D with 12 elements
arr = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# Shape: (12,)

# Reshape to 2D: 3 rows, 4 columns
arr_2d = arr.reshape(3, 4)
# Shape: (3, 4)
# [[1,  2,  3,  4],
#  [5,  6,  7,  8],
#  [9, 10, 11, 12]]

# Reshape to 3D: 2 × 2 × 3
arr_3d = arr.reshape(2, 2, 3)
# Shape: (2, 2, 3)
# [[[1,  2,  3],
#   [4,  5,  6]],
#  [[7,  8,  9],
#   [10, 11, 12]]]
```

**Next**, we transpose matrices to `[batch, num_heads, seq_len, head_dim]` to allow for the computation of scores later in:
```python
scores = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)
```
Which works because of matrix multiplication, which requires the last two dimensions of the first vector to be the same as the first two dimensions of the second vector:
- Last two dimensions: `[seq_len, head_dim]` - this is what each head operates on
- First two dimensions: `[batch, num_heads]` - these are just batched


#### The backward pass:
The backward pass is made with JAX's automatic gradient function, `jax.vjp`:

```python
params = self.get_params()
output, vjp_fn = jax.vjp(
    lambda p, x_: self.fwd(p, x_, self.num_heads, self.head_dim, self.embedding_dim), params, x)
grads_params, d_input = vjp_fn(d_output)
return grads_params, d_input
```

Here, we first get the parameters calculated from the forward pass using the `get_params()` method, which returns a dictionary:
```python
return {
    "W_Q": self.W_Q,
    "W_K": self.W_K,
    "W_V": self.W_V,
    "W_O": self.W_O
}
```

VJP stands for Vector-Jacobian Product, which is passed the fwd method of the MHA class, and automatically calculates gradients for backpropagation.

### Transformer Block and Stack:

The transformer block is like the glue that ties everything together. It is where the forward passes of the FFN and MHA layer are put together, and are used to get normalized and stable outputs.

#### What is layer normalization?
Layer normalization is used to ensure stable training by flattening every feature and taking away any excessively large number ranges. It makes it so that each of these features have mean 0 and variance 1.

> [!NOTE]
> ***Mean***: the average of all the numbers in the embedding:
>
> For a vector
>
> $$x = [x_1, x_2, ..., x_n]$$
>
> The formula becomes:
>
> $$\mu = \frac{1}{n}\sum_{i=1}^n x_i$$
>
> ***Variance***: measure of how spread out the numbers are from that same mean (range):
>
>$$\sigma^2=\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2$$
> Where:
> - $x_i$ is the *i-th* number in the vector $x$, also called the **feature**.
> - $n$ is the number of features in the vector, or in this case, `n = embedding_dim`
> - $\mu$ is the mean that was calculated using the earlier mathematical model.
>
> $(x_i - \mu)$ tells us how far each feature is from the mean, and it is squared so that positive and negative distances don't cancel out, but also so that big deviations become more important.
>
> Finally, it is divided by the $n$ to get the average of all the deviations

The next step in layer norm is the actual calculation using mean and variance:

$$
\hat{x} = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

Here, we subtract the mean, $\mu$ from the original features, $x_i$, centering each feature around zero.

Then, we divide by the standard deviation, being $\sqrt{\sigma^2-\epsilon}$, where $\epsilon$ is a very small positive number added for numerical stability. This is so that if the variance $\sigma^2$ is very close to zero, we avoid division by zero error.

Finally, we apply $\gamma$ and $\beta$, being learnable parameters. `gamma`, $\gamma$, rescales the normalized value, while `beta`, $\beta$, shifts the normalized values. Think of them like the weights and biases in the FFN.

#### The Forward Pass:

Firstly, the fwd pass of the transformer block consists of two sublayers:
- LayerNorm → MultiHeadAttention
- LayerNorm → FeedForward Network

In the first sublayer, it sets a residual variable, defined by `residual_1 = x`, where `x` are the input embeddings, of shape [batch, seq_len, embedding_dim].

Then, `ln1_out` is declared, being defined as the first layer normalization, taking the parameters of `gamma1` and `beta1`.
