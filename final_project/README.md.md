#### Input Gate($i$)
$$i_t = \sigma(W_{x_i}x_t+W_{h_i}h_{t-1}+b_i))$$
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

#### Forget gate($f$)
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
#### Output gate($o$)
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
#### Gate gate($g$)
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
#### Cell State Update
1. Element-wise multiplication of forget and prev cell state
   $$f_t \odot c_{t-1}$$
2. Element-wise multiplication of input gate and gate gate
   $$i_t \odot g_t$$
3. sum them
   $$c_t = {f_t \odot c_{t-1}}+{i_t \odot g_t}$$

#### Hidden State Update
1. apply $\tanh$ to cell state
   $$\tanh(c_t)$$
2. Element-wise multiplication with output state
   $$h_t =o_t \odot \tanh(c_t)$$

