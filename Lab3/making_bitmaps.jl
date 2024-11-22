using CSV
using DataFrames
using Plots
using LinearAlgebra
using LaTeXStrings
using StatsPlots
using TSVD
using Images
using ImageIO
using ColorTypes
using ImageView;

function main()
    abstract1 = load("bitmaps/abstract1.bmp")

    abstract1_rows, abstract1_cols = size(abstract1)
    println(size(abstract1))
    abstract1R, abstract1G, abstract1B = zeros(abstract1_rows, abstract1_cols), zeros(abstract1_rows, abstract1_cols), zeros(abstract1_rows, abstract1_cols)

    for i in 1:1:abstract1_rows
        for j in 1:1:abstract1_cols
            pixel_ij = abstract1[i,j]
            abstract1R[i,j] = red(pixel_ij)
            abstract1G[i,j] = green(pixel_ij)
            abstract1B[i,j] = blue(pixel_ij)
        end
    end

    built_image = colorview(RGB, abstract1R, abstract1G, abstract1B)
    save("bitmaps/built_abstract1.bmp", built_image)

end

main()