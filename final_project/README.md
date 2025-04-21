# LSTM Cell
## Input Gate($i$)

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

## Forget gate($f$)
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
## Output gate($o$)
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
## Gate gate($g$)
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
## Cell State Update($c_t$)
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

## Hidden State Update($h_t$)
$$h_t =o_t \odot \tanh(c_t)$$

- Partials
	- $h_t = o_t \odot q$
		- $\nabla_{o_t}h_t=q$
		- $\nabla_{q}h_t=o_t$
	- $q=\tanh(c_t)$
		- $\nabla_{c_t}q=1-\tanh^2(c_t)$
- $\nabla_{o_t}h_t=q$
- $\nabla_{c_t}h_t=\nabla_{q}h_t\nabla_{c_t}q=o_t\nabla_{c_t}q$

