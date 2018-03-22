#module TestComp
#export main
using DataFrames

function main()
    x = rand(100)
    #mean(x,2)
    mean(x)
    y = DataFrames.DataFrame(x=x)
    show(y)
end

@time for i ∈ 1:10000000
    u = rand()
end

@time rand(1000000)

#show(x)
