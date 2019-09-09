import torch
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
out = z.mean()
print(z,out)

a= torch.randn(2,2)
a = ((a*3)/(a-1))
print('a.requires_grad: ',a.requires_grad)
b = (a * a).sum()
print('b.requires_grad: ',b.requires_grad)
print(b.grad_fn)
a.requires_grad_(True)
print('a.requires_grad: ',a.requires_grad)
print('b.requires_grad: ',b.requires_grad)
print(b.grad_fn)
b = (a * a).sum()
print('b.requires_grad: ',b.requires_grad)
print(b.grad_fn)

out.backward()
print(x.grad)
