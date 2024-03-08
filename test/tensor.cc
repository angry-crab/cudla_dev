#include "tensor.h"

int main()
{
    std::vector<int> dim_a{2,3};
    std::vector<float> a(2*3, 1.0);
    for(int i=0;i<3;i++)
    {
        a[i] = 2.0;
    }
    Tensor<float> A(dim_a, a);

    return 0;
}
