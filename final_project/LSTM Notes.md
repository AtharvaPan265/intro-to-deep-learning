**Input Gate($i$)**$$i_t = \sigma(W_{x_i}x_t+W_{h_i}h_{t-1}+b_i))$$
**Forget gate($f$)**$$f_t=\sigma({W_{x_f}x_t}+{W_{h_f}h_{t-1}}+{b_f})$$
**Output gate($o$)**$$o_t=\sigma({W_{x_o}x_t}+{W_{h_o}h_{t-1}}+{b_o})$$
**Gate gate($g$)**$$g_t=\tanh({W_{x_g}x_t}+{W_{h_g}h_{t-1}}+{b_g})$$
**Cell State Update**
1. Element-wise multiplication of forget and prev cell state$$f_t \odot c_{t-1}$$
2. Element-wise multiplication of input gate and gate gate$$i_t \odot g_t$$
3. sum them$$c_t = {f_t \odot c_{t-1}}+{i_t \odot g_t}$$

**Hidden State Update**
1. apply $\tanh$ to cell state$$\tanh(c_t)$$
2. Element-wise multiplication with output state$$h_t =o_t \odot \tanh(c_t)$$

