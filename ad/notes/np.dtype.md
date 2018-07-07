# np.dtype

> 在阅读Snorkel源码，numbskull库时，发现了很多数据类型诡异，不能直接解释，比如这样的`factor[factor_id]["factorFunction"]`。
> 
> 在观察`factor`的过程中，发现`dtpye`的结果并不能看懂，
> 
> `dtype([('factorFunction', '<i2'), ('weightId', '<i8'), ('featureValue', '<f8'), ('arity', '<i8'), ('ftv_offset', '<i8')])`
> 
> 
> 故查看了numpy的说明文档，下面用例子来说明`dtype`的作用。

## 1. 定义dtype
```python
dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
```
下面来查看刚刚定义的结构。
```python
>>> dt['name']
dtype('|U16')
>>> dt['grades']
dtype(('float64',(2,)))
```
## 2. 在np.array中应用

定义array
```python
 x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
```

查看`dt`如何影响的x
```python
>>>x[1]
('John', [6.0, 7.0])
>>> x[1]['grades']
array([ 6.,  7.])
>>> type(x[1])
<type 'numpy.void'>
>>> type(x[1]['grades'])
<type 'numpy.ndarray'>
```

## 结论
dt制定了array中每一个元素应该具有的数据结构。

仿佛是一个列标签，x被分为`name`&`grades`两个标签。