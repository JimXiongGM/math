# Note

原始的prompt有问题，
1. output序列过长，需要设置1024才够

改了一点
1. prompt参考了 https://github.com/FranxYao/chain-of-thought-hub/blob/main/MATH/lib_prompt/algebra/prompt_grad_5.txt gpt4能给出结果，但是评价脚本有问题，使用的是字符相等，这有问题。


两个矩阵
$$\begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ -\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \end{bmatrix}$$
和
$$\begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ -1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}$$
是相等的，怎么使用Python判断？

```python
import numpy as np

# 定义两个矩阵
matrix1 = np.array([
    [np.sqrt(2)/2, np.sqrt(2)/2],
    [-np.sqrt(2)/2, np.sqrt(2)/2]
])

matrix2 = np.array([
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)]
])

# 比较两个矩阵
are_equal = np.allclose(matrix1, matrix2)

print("两个矩阵是否相等:", are_equal)
```

没有找到合适的将latex转换为numpy的库。