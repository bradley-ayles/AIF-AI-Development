import torch




def main():
    var1 = torch.ones(size=(2, 4))
    print(var1.shape)
    var3 = torch.tensor([[1., 2., 3., 4.], 
                         [4., 3., 2., 1.]])
    print(var3.shape)
    var2 = torch.normal(mean=var3, std=var1)

    var1 = torch.ones(size=(8, 2))

    print(var1)
    print(var2)

    print(torch.matmul(var1, var2))

main()