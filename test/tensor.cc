#include "tensor.h"

int main()
{
    std::vector<int> dim_a{2,3,5};
    std::vector<float> a(2*3*5, 1.0);
    for(int i=0;i<2*3*5;i++)
    {
        a[i] = i+1;
    }
    tensor::Tensor<float, 3> A(dim_a, a);


    A.pretty_print_3d();

    // A.permute({0,2,1});
    A.permute({2,1,0});

    std::cout << "permute" << std::endl;

    A.pretty_print_3d();



    // std::vector<int> res = tensor::default_stride(dim_a);
    // for(auto it : res)
    // {
    //     std::cout << it << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
