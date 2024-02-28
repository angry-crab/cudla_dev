#include <cstring>
#include <vector>
#include <iostream>
#include <cassert>

static void transpose2d(std::vector<float>& input, std::vector<float>& output, std::vector<int>& dim)
{
    assert(dim.size() == 2);
    int la = dim[0], lb = dim[1];
    for(int i=0; i<dim[0]; ++i)
    {
        for(int j=0; j<dim[1]; ++j)
        {
            int idx_from = i*lb + j;
            int idx_to = j*la + i;
            output[idx_to] = input[idx_from];
        }
    }
}

// 1 2 3 
// 4 5 6
// 7 8 9

// 1 4 7
// 2 5 8
// 3 6 9

// 1 2 3 4
// 5 6 7 8

// 1 2 3 4 5 6 7 8

// 1 5
// 2 6
// 3 7
// 4 8

// 1 5 2 6 3 7 4 8
int main()
{

    std::vector<float> a{1,2,3,4,5,6,7,8};
    std::vector<float> b(8, 0);
    std::vector<int> dim_from{2,4};
    std::vector<int> dim_to{4,2};

    transpose2d(a, b, dim_from);

    for(int i=0; i<a.size(); ++i)
    {
        std::cout << b[i] << " ";
        if((i+1) % dim_to.back() == 0 )
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
}