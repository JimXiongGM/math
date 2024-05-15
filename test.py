import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex

def latex_to_numpy(latex_str):
    """
    将LaTeX矩阵表达式转换为NumPy数组。
    """
    # 解析LaTeX字符串为SymPy表达式
    sympy_mat = parse_latex(latex_str)

    # 将SymPy矩阵转换为NumPy数组
    numpy_array = np.array(sympy_mat.tolist()).astype(float)
    return numpy_array

def compare_matrices(latex_mat1, latex_mat2):
    """
    解析两个LaTeX矩阵表达式，并比较它们是否相等。
    """
    # 转换LaTeX到NumPy数组
    mat1 = latex_to_numpy(latex_mat1)
    mat2 = latex_to_numpy(latex_mat2)

    # 使用NumPy的 allclose 方法比较两个矩阵
    return np.allclose(mat1, mat2)

# LaTeX矩阵表达式
latex_matrix1 = r"\begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ -\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \end{bmatrix}"
latex_matrix2 = r"\begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}"

# 比较两个矩阵
result = compare_matrices(latex_matrix1, latex_matrix2)
print("两个矩阵是否相等:", result)