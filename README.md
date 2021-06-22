# DCWH
A PyTorch Implementation of [Deep Class-Wise Hashing: Semantics-Preserving Hashing via Class-Wise Loss](https://ieeexplore.ieee.org/document/8759067) (DCWH)
# Demo
## Training
 ```
python dcwh.py  --network google  --dataset cifar100 --len 32 --path cifar100_google_32.pth
```
This will train a model using pre-trained GoogLeNet architecture with 32-bit codes on CIFAR-100 dataset. The model will be saved as "cifar100_google_32.pth" under the directory "./checkpoint".

## Evaluation

Simply add "-e" to turn into evaluation mode:

 ```
python dcwh.py  -e --network google  --dataset cifar100 --len 32 --path cifar100_google_32.pth
```

# Results

The experimental results of our implementation are shown as follows:
<table>
	<tbody>
		<tr>
			<td rowspan="2">Net Structure</td>
			<td rowspan="2">Dataset</td>
			<td colspan="4">mAP (%)</td>
		</tr>
		<tr>
			<td>16-bit</td>
			<td>24-bit</td>
			<td>32-bit</td>
			<td>48-bit</td>
		</tr>
		<tr>
			<td rowspan="2">GoogLeNet</td>
			<td>CIFAR-10</td>
			<td>95.74</td>
			<td>96.03</td>
			<td>96.05</td>
			<td>96.04</td>
		</tr>
		<tr>
			<td>CIFAR-100</td>
			<td>63.93</td>
			<td>74.33</td>
			<td>74.84</td>
			<td>76.37</td>
		</tr>
	</tbody>
</table>
Our results on CIFAR-10 are better than the original paper, which were 94.0%, 95.0%, 95.4%, and 95.2% under 16-bit, 24-bit, 32-bit, and 48-bit, respectively. 
<br/><br/>
For CIFAR-100 dataset, our results are comparable to the original paper since DCWH achieved 75.70% and 76.90% under 32-bit and 48-bit codes, respectively.
 
