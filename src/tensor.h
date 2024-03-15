#include <iostream>
#include <vector>
#include <type_traits>
#include <cassert>

namespace tensor
{

inline std::vector<int> default_stride(std::vector<int>& dim)
{
    std::vector<int> res(dim.size(), 1);
    int s = 1;
    for(int i=dim.size()-1; i>0; --i)
    {
        s *= dim[i];
        res[i-1] = s;
    }

    return res;
}

template <typename T, int RANK,
            typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
struct Tensor
{

    Tensor() = delete;
    Tensor(Tensor&& other) = delete;
    Tensor& operator=(Tensor&& other) = delete;
    explicit Tensor(std::vector<int>& dim) : dim_(dim)
    {
        assert(dim.size() == RANK);
        // might overflow
        int total = 1;
        for(auto it : dim)
        {
            total *= it;
        }
        data_.reserve(total);
        stride_ = default_stride(dim);
    }

    explicit Tensor(std::vector<int>& dim, std::vector<T>& data) : dim_(dim), data_(data)
    {
        assert(dim.size() == RANK);
        stride_ = default_stride(dim);
    }

    explicit Tensor(const Tensor& other)
    {
        assert(other.size().size() == RANK);
        dim_ = other.size();
        data_ = other.data();
        stride_ = other.stride();
    }

    inline std::vector<int> size()
    {
        return dim_;
    }

    inline std::vector<T> data()
    {
        return data_;
    }

    inline std::vector<int> stride()
    {
        return stride_;
    }

    Tensor& operator=(const Tensor& other)
    {
        assert(other.size().size() == RANK);
        dim_ = other.size();
        data_ = other.data();
        stride_ = other.stride();
    }

    void permute(const std::vector<int>& axis)
    {
        assert(axis.size() == dim_.size());
        std::vector<int> new_dim(dim_.size());
        std::vector<int> new_stride(stride_.size());

        for(int i=0; i<axis.size(); ++i)
        {
            new_dim[i] = dim_[axis[i]];
            new_stride[i] = stride_[axis[i]];
        }
        dim_ = new_dim;
        stride_ = new_stride;
    }

    template <typename ... Ts, typename = std::enable_if_t<(std::is_convertible_v<Ts, int> && ...)>>
    T operator()(Ts... args)
    {
        int index = get_index({args...});
        return data_[index];
    }

    // void continuous()
    // {
    //     std::vector<T> new_data(data_.size());

    // }

    void pretty_print_3d()
    {
        for(int i=0; i<dim_[0]; ++i)
        {
            for(int j=0; j<dim_[1]; ++j)
            {
                for(int k=0; k<dim_[2]; k++)
                {
                    std::cout << (*this)(i,j,k) << " ";
                }
                std::cout << std::endl;

            }
            std::cout << std::endl;
        }
    }

private:
    int get_index(const std::vector<int>& coords)
    {
        int res = 0;
        for(int i=0; i<stride_.size(); ++i)
        {
            res += stride_[i] * coords[i];
        }

        return res;
    }

    std::vector<int> dim_;
    std::vector<int> stride_;
    std::vector<T> data_;

};
// dim :    {2,4} -> {4,2}
// stride : {4,1} -> {1,4}

// 1 2 3 4
// 5 6 7 8

// 1 5
// 2 6
// 3 7
// 4 8


}
