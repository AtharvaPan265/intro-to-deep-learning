# LSTM Cell

$\begin{array}{ll} \\
i_t = \sigma(W_{x_i} x_t + b_{x_i} + W_{h_i} h_{t-1} + b_{h_i}) \\
f_t = \sigma(W_{x_f} x_t + b_{x_f} + W_{h_f} h_{t-1} + b_{h_f}) \\
g_t = \tanh(W_{x_g} x_t + b_{x_g} + W_{h_g} h_{t-1} + b_{h_g}) \\
o_t = \sigma(W_{x_o} x_t + b_{x_o} + W_{h_o} h_{t-1} + b_{h_o}) \\
c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
h_t = o_t \odot \tanh(c_t) \\
\end{array}$

## Deriving Gradients
### Input Gate($i_t$)

$$i_t = \sigma(W_{x_i}x_t+W_{h_i}h_{t-1}+b_i)$$
- Partials
	- $i_t = \sigma (p)$
		- $\nabla_pi_t=(1-\sigma(p))\sigma(p)$
	- $p = q + v + b_i$
		- $\nabla_qp=1$
		- $\nabla_vp=1$
		- $\nabla_{b_i}p=1$
	- $q = W_{x_i}x_t$
		- $\nabla_{W_{x_i}}q = x_t^T$
		- $\nabla_{x_t}q = W_{x_i}^T$
	- $v = W_{h_i}h_{t-1}$
		- $\nabla_{W_{h_i}}q = h_{t-1}^T$
		- $\nabla_{h_{t-1}}q = W_{h_i}^T$
- $\nabla_{W_{x_i}}{i_t}=\nabla_pi_t \nabla_qp \nabla_{W_{x_i}}q = x_t^T\nabla_pi_t$
- $\nabla_{W_{h_i}}{i_t}=\nabla_pi_t \nabla_vp \nabla_{W_{h_i}}v = h_{t-1}^T\nabla_pi_t$
- $\nabla_{b_i}i_t = \nabla_pi_t \nabla_{b_i}p = \nabla_pi_t$

### Forget gate($f_t$)
$$f_t=\sigma({W_{x_f}x_t}+{W_{h_f}h_{t-1}}+{b_f})$$
- Partials
	- $f_t = \sigma (p)$
		- $\nabla_pf_t=(1-\sigma(p))\sigma(p)$
	- $p = q + v + b_f$
		- $\nabla_qp=1$
		- $\nabla_vp=1$
		- $\nabla_{b_f}p=1$
	- $q = W_{x_f}x_t$
		- $\nabla_{W_{x_f}}q = x_t^T$
		- $\nabla_{x_t}q = W_{x_f}^T$
	- $v = W_{h_f}h_{t-1}$
		- $\nabla_{W_{h_f}}q = h_{t-1}^T$
		- $\nabla_{h_{t-1}}q = W_{h_f}^T$
- $\nabla_{W_{x_f}}{f_t}=\nabla_pf_t \nabla_qp \nabla_{W_{x_f}}q = x_t^T\nabla_pf_t$
- $\nabla_{W_{h_f}}{f_t}=\nabla_pf_t \nabla_vp \nabla_{W_{h_f}}v = h_{t-1}^T\nabla_pf_t$
- $\nabla_{b_f}f_t = \nabla_pf_t \nabla_{b_f}p = \nabla_pf_t$
### Output gate($o_t$)
$$o_t=\sigma({W_{x_o}x_t}+{W_{h_o}h_{t-1}}+{b_o})$$
- Partials
	- $o_t = \sigma (p)$
		- $\nabla_po_t=(1-\sigma(p))\sigma(p)$
	- $p = q + v + b_o$
		- $\nabla_qp=1$
		- $\nabla_vp=1$
		- $\nabla_{b_o}p=1$
	- $q = W_{x_o}x_t$
		- $\nabla_{W_{x_o}}q = x_t^T$
		- $\nabla_{x_t}q = W_{x_o}^T$
	- $v = W_{h_o}h_{t-1}$
		- $\nabla_{W_{h_o}}q = h_{t-1}^T$
		- $\nabla_{h_{t-1}}q = W_{h_o}^T$
- $\nabla_{W_{x_o}}{o_t}=\nabla_po_t \nabla_qp \nabla_{W_{x_o}}q = x_t^T\nabla_po_t$
- $\nabla_{W_{h_o}}{o_t}=\nabla_po_t \nabla_vp \nabla_{W_{h_o}}v = h_{t-1}^T\nabla_po_t$
- $\nabla_{b_o}o_t = \nabla_po_t \nabla_{b_o}p = \nabla_po_t$
### Candidate gate($g_t$)
$$g_t=\tanh({W_{x_g}x_t}+{W_{h_g}h_{t-1}}+{b_g})$$
- Partials
	- $g_t = \sigma (p)$
		- $\nabla_pg_t=1-\tanh^2(p)$
	- $p = q + v + b_g$
		- $\nabla_qp=1$
		- $\nabla_vp=1$
		- $\nabla_{b_g}p=1$
	- $q = W_{x_g}x_t$
		- $\nabla_{W_{x_g}}q = x_t^T$
		- $\nabla_{x_t}q = W_{x_g}^T$
	- $v = W_{h_g}h_{t-1}$
		- $\nabla_{W_{h_g}}q = h_{t-1}^T$
		- $\nabla_{h_{t-1}}q = W_{h_g}^T$
- $\nabla_{W_{x_g}}{g_t}=\nabla_pg_t \nabla_qp \nabla_{W_{x_g}}q = x_t^T\nabla_pg_t$
- $\nabla_{W_{h_g}}{g_t}=\nabla_pg_t \nabla_vp \nabla_{W_{h_g}}v = h_{t-1}^T\nabla_pg_t$
- $\nabla_{b_g}g_t = \nabla_pg_t \nabla_{b_g}p = \nabla_pg_t$
### Cell State Update($c_t$)
$$c_t = {f_t \odot c_{t-1}}+{i_t \odot g_t}$$
- Partials
	- $c_t = q + p$
		- $\nabla_qc_t=1$
		- $\nabla_pc_t=1$
	- $q = f_t \odot c_{t-1}$
		- $\nabla_{f_t}q=c_{t-1}$
		- $\nabla_{c_{t-1}}q=f_t$
	- $p = i_t \odot g_t$
		- $\nabla_{i_t}p=g_t$
		- $\nabla_{g_t}p=i_t$
- $\nabla_{f_t}c_t=\nabla_qc_t\nabla_{f_t}q=c_{t-1}$
- $\nabla_{i_t}c_t= \nabla_{p}c_t\nabla_{i_t}p=g_t$
- $\nabla_{g_t}c_t= \nabla_{p}c_t\nabla_{g_t}p=i_t$

### Hidden State Update($h_t$)
$$h_t =o_t \odot \tanh(c_t)$$

- Partials
	- $h_t = o_t \odot q$
		- $\nabla_{o_t}h_t=q$
		- $\nabla_{q}h_t=o_t$
	- $q=\tanh(c_t)$
		- $\nabla_{c_t}q=1-\tanh^2(c_t)$
- $\nabla_{o_t}h_t=q$
- $\nabla_{c_t}h_t=\nabla_{q}h_t\nabla_{c_t}q=o_t\nabla_{c_t}q$


## All Gradients

- $i_t$
	- $\nabla_{W_{x_i}}{i_t}=\nabla_pi_t \nabla_qp \nabla_{W_{x_i}}q = x_t^T\nabla_pi_t$
	- $\nabla_{W_{h_i}}{i_t}=\nabla_pi_t \nabla_vp \nabla_{W_{h_i}}v = h_{t-1}^T\nabla_pi_t$
	- $\nabla_{b_i}i_t = \nabla_pi_t \nabla_{b_i}p = \nabla_pi_t$
- $f_t$
	- $\nabla_{W_{x_f}}{f_t}=\nabla_pf_t \nabla_qp \nabla_{W_{x_f}}q = x_t^T\nabla_pf_t$
	- $\nabla_{W_{h_f}}{f_t}=\nabla_pf_t \nabla_vp \nabla_{W_{h_f}}v = h_{t-1}^T\nabla_pf_t$
	- $\nabla_{b_f}f_t = \nabla_pf_t \nabla_{b_f}p = \nabla_pf_t$
- $o_t$
	- $\nabla_{W_{x_o}}{o_t}=\nabla_po_t \nabla_qp \nabla_{W_{x_o}}q = x_t^T\nabla_po_t$
	- $\nabla_{W_{h_o}}{o_t}=\nabla_po_t \nabla_vp \nabla_{W_{h_o}}v = h_{t-1}^T\nabla_po_t$
	- $\nabla_{b_o}o_t = \nabla_po_t \nabla_{b_o}p = \nabla_po_t$
- $g_t$
	- $\nabla_{W_{x_g}}{g_t}=\nabla_pg_t \nabla_qp \nabla_{W_{x_g}}q = x_t^T\nabla_pg_t$
	- $\nabla_{W_{h_g}}{g_t}=\nabla_pg_t \nabla_vp \nabla_{W_{h_g}}v = h_{t-1}^T\nabla_pg_t$
	- $\nabla_{b_g}g_t = \nabla_pg_t \nabla_{b_g}p = \nabla_pg_t$
- $c_t$
	- $\nabla_{f_t}c_t=\nabla_qc_t\nabla_{f_t}q=c_{t-1}$
	- $\nabla_{i_t}c_t= \nabla_{p}c_t\nabla_{i_t}p=g_t$
	- $\nabla_{g_t}c_t= \nabla_{p}c_t\nabla_{g_t}p=i_t$

- $h_t$
	- $\nabla_{o_t}h_t=q$
	- $\nabla_{c_t}h_t=\nabla_{q}h_t\nabla_{c_t}q=o_t\nabla_{c_t}q$

## Backpropagation





# Ripped LSTM Docs

# LSTM (Long Short-Term Memory)

## Overview

Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.

For each element in the input sequence, each layer computes the following function:

$$
\begin{array}{ll} \\
i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
h_t = o_t \odot \tanh(c_t) \\
\end{array}
$$

where $h_t$ is the hidden state at time `t`, $c_t$ is the cell state at time `t`, $x_t$ is the input at time `t`, $h_{t-1}$ is the hidden state of the layer at time `t-1` or the initial hidden state at time `0`, and $i_t$, $f_t$, $g_t$, $o_t$ are the input, forget, cell, and output gates, respectively. $\sigma$ is the sigmoid function, and $\odot$ is the Hadamard product[1].

In a multilayer LSTM, the input $x^{(l)}_t$ of the $l$-th layer ($l \ge 2$) is the hidden state $h^{(l-1)}_t$ of the previous layer multiplied by dropout $\delta^{(l-1)}_t$ where each $\delta^{(l-1)}_t$ is a Bernoulli random variable which is $0$ with probability `dropout`[1].

If `proj_size > 0` is specified, LSTM with projections will be used. This changes the LSTM cell: the dimension of $h_t$ will be changed from `hidden_size` to `proj_size`, and the output hidden state of each layer will be multiplied by a learnable projection matrix: $h_t = W_{hr}h_t$. More details can be found in https://arxiv.org/abs/1402.1128[1].

## Arguments

- **input_size**: The number of expected features in the input `x`
- **hidden_size**: The number of features in the hidden state `h`
- **num_layers**: Number of recurrent layers. Default: 1
- **bias**: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`
- **batch_first**: If `True`, then the input and output tensors are provided as `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: `False`
- **dropout**: If non-zero, introduces a `Dropout` layer on the outputs of each LSTM layer except the last layer. Default: 0
- **bidirectional**: If `True`, becomes a bidirectional LSTM. Default: `False`
- **proj_size**: If `> 0`, will use LSTM with projections of corresponding size. Default: 0[1]

## Inputs

- **input**: tensor of shape $(L, H_{in})$ for unbatched input, $(L, N, H_{in})$ when `batch_first=False` or $(N, L, H_{in})$ when `batch_first=True` containing the features of the input sequence.
- **h_0**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the initial hidden state.
- **c_0**: tensor of shape $(D * \text{num\_layers}, H_{cell})$ for unbatched input or $(D * \text{num\_layers}, N, H_{cell})$ containing the initial cell state[1].

Where:

$$
\begin{aligned}
N &= \text{batch size} \\
L &= \text{sequence length} \\
D &= 2 \text{ if bidirectional=True otherwise } 1 \\
H_{in} &= \text{input\_size} \\
H_{cell} &= \text{hidden\_size} \\
H_{out} &= \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
\end{aligned}
$$

## Outputs

- **output**: tensor of shape $(L, D * H_{out})$ for unbatched input, $(L, N, D * H_{out})$ when `batch_first=False` or $(N, L, D * H_{out})$ when `batch_first=True` containing the output features $(h_t)$ from the last layer of the LSTM.
- **h_n**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the final hidden state.
- **c_n**: tensor of shape $(D * \text{num\_layers}, H_{cell})$ for unbatched input or $(D * \text{num\_layers}, N, H_{cell})$ containing the final cell state[1].

## Attributes

- **weight_ih_l[k]**: the learnable input-hidden weights of the $\text{k}^{th}$ layer
- **weight_hh_l[k]**: the learnable hidden-hidden weights of the $\text{k}^{th}$ layer
- **bias_ih_l[k]**: the learnable input-hidden bias of the $\text{k}^{th}$ layer
- **bias_hh_l[k]**: the learnable hidden-hidden bias of the $\text{k}^{th}$ layer
- **weight_hr_l[k]**: the learnable projection weights of the $\text{k}^{th}$ layer (only when `proj_size > 0`)[1]

Bidirectional LSTMs also have corresponding reverse direction weights and biases.

## Notes

- All weights and biases are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = \frac{1}{\text{hidden\_size}}$
- For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively
- For bidirectional LSTMs, `h_n` is not equivalent to the last element of `output`
- `batch_first` argument is ignored for unbatched inputs
- `proj_size` should be smaller than `hidden_size`[1]

## Example

```python
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
```
