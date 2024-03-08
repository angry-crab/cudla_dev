#include <iostream>
#include <vector>
#include <type_traits>

template <typename T,
            typename = std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>>
struct Tensor
{
    Tensor() = delete;
    Tensor(Tensor&& other) = delete;
    Tensor& operator=(Tensor&& other) = delete;
    explicit Tensor(std::vector<int>& dim) : dim_(dim)
    {
        // might overflow
        int total = 1;
        for(auto it : dim)
        {
            total *= it;
        }
        data_.reserve(total);
    }

    explicit Tensor(std::vector<int>& dim, std::vector<T>& data) : dim_(dim), data_(data)
    {}

    explicit Tensor(const Tensor& other)
    {
        dim_ = other.size();
        data_ = other.data();
    }

    inline std::vector<int> size()
    {
        return dim_;
    }

    inline std::vector<T> data()
    {
        return data_;
    }

    void setData(std::vector<T>& data)
    {
        data_ = data;
    }

    void setDim(std::vector<int>& dim)
    {
        dim_ = dim;
    }

    Tensor& operator=(const Tensor& other)
    {
        dim_ = other.size();
        data_ = other.data();
    }

    std::vector<int> dim_;
    std::vector<T> data_;
};

