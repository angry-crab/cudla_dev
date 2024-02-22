#include <cstring>
#include <vector>
#include <iostream>

static bool concat(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, 
                    std::vector<int>& dim_A, std::vector<int>& dim_B, std::vector<int>& dim_C, 
                    std::vector<float>& output)
{
    std::size_t pa = 0, pb = 0, pc = 0;
    std::size_t da = dim_A.back(), db = dim_B.back(), dc = dim_C.back();
    for(std::size_t i=0; i<output.size(); i+= da+db+dc, pa+=da, pb+=db, pc+=dc)
    {
        if(i+da >= output.size() || i+da+db >= output.size() || pa >= A.size() || pb >= B.size() || pc >= C.size())
        {
            return false;
        }
        std::memcpy((void*)&output[i], (void*)&A[pa], da*sizeof(float));
        std::memcpy((void*)&output[i+da], (void*)&B[pa], db*sizeof(float));
        std::memcpy((void*)&output[i+da+db], (void*)&C[pc], dc*sizeof(float));
    }

    return true;
}

int main()
{
    std::vector<float> a(6, 1.0);
    std::vector<float> b(8, 2.0);
    std::vector<float> c(10, 3.0);
    
    std::vector<int> da{2,3};
    std::vector<int> db{2,4};
    std::vector<int> dc{2,5};

    std::vector<float> output(24, 0.0);
    std::vector<int> output_dims{2, 12};

    auto res = concat(a, b, c, da, db, dc, output);
    if(res == false)
    {
        std::cout << "invalid ? " << std::endl;
    }

    for(int i=0; i<output.size(); ++i)
    {
        std::cout << output[i] << " ";
        if((i+1) % output_dims.back() == 0 )
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
}