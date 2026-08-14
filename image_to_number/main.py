import torch




def main():
    var1 = torch.ones(size=(3, 2))
    var2 = torch.rand(size=(2, 6))

    print(var1)
    print(var2)

    print(torch.matmul(var1, var2))

main()