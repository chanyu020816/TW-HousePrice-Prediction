using CSV
using DataFrames
using Pipe
using Statistics
using LinearAlgebra
using Plots
using RCall

using MLJ
using StatsBase
using LinRegOutliers
using UMAP
using Makie
using MultivariateStats
using ScikitLearn
using MLBase
using Distances
using DecisionTree
using ScikitLearn.CrossValidation: cross_val_score
# set up environment
ENV["LINES"], ENV["COLUMNS"] = 15, 200
# set up location for Atom
ls_loc = "/home/chanyu/Desktop/school/DataMining/project/dataset/dataset/last_year/"
# variable not needed
exc = ["Column1", "土地位置建物門牌", "編號", "非都市土地使用分區", "非都市土地使用編定",
    "都市土地使用分區", "交易筆棟數", "交易年月日", "建築完成年月", "備註", "移轉編號", "period"]
# import files
begin
    ls_tp = @pipe CSV.File(ls_loc*"LastYear_tp.csv") |>
        DataFrame |>
        DataFrames.select(_, Not(exc)) |>
        rename(_,
        :"建物現況格局.房" => :建物現況格局_房,
        :"建物現況格局.廳" => :建物現況格局_廳,
        :"建物現況格局.衛" => :建物現況格局_衛,
        :"建物現況格局.隔間" => :建物現況格局_隔間,
        :"車位移轉總面積.平方公尺." => :車位移轉總面積_平方公尺
        )
    ls_ks = @pipe CSV.File(ls_loc*"LastYear_ks.csv") |>
        DataFrame |>
        DataFrames.select(_, Not(exc)) |>
        rename(_,
        :"建物現況格局.房" => :建物現況格局_房,
        :"建物現況格局.廳" => :建物現況格局_廳,
        :"建物現況格局.衛" => :建物現況格局_衛,
        :"建物現況格局.隔間" => :建物現況格局_隔間,
        :"車位移轉總面積.平方公尺." => :車位移轉總面積_平方公尺
        )
    ls_nt = @pipe CSV.File(ls_loc*"LastYear_nt.csv")  |>
        DataFrame |>
        DataFrames.select(_, Not(exc)) |>
        rename(_,
        :"建物現況格局.房" => :建物現況格局_房,
        :"建物現況格局.廳" => :建物現況格局_廳,
        :"建物現況格局.衛" => :建物現況格局_衛,
        :"建物現況格局.隔間" => :建物現況格局_隔間,
        :"車位移轉總面積.平方公尺." => :車位移轉總面積_平方公尺
        )
    ls_tc = @pipe CSV.File(ls_loc*"LastYear_tc.csv") |>
        DataFrame |>
        DataFrames.select(_, Not(exc)) |>
        rename(_,
        :"建物現況格局.房" => :建物現況格局_房,
        :"建物現況格局.廳" => :建物現況格局_廳,
        :"建物現況格局.衛" => :建物現況格局_衛,
        :"建物現況格局.隔間" => :建物現況格局_隔間,
        :"車位移轉總面積.平方公尺." => :車位移轉總面積_平方公尺
        )
    ls_tn = @pipe CSV.File(ls_loc*"LastYear_tn.csv")  |>
        DataFrame |>
        DataFrames.select(_, Not(exc)) |>
        rename(_,
        :"建物現況格局.房" => :建物現況格局_房,
        :"建物現況格局.廳" => :建物現況格局_廳,
        :"建物現況格局.衛" => :建物現況格局_衛,
        :"建物現況格局.隔間" => :建物現況格局_隔間,
        :"車位移轉總面積.平方公尺." => :車位移轉總面積_平方公尺
        )
    ls_ty = @pipe CSV.File(ls_loc*"LastYear_ty.csv") |>
        DataFrame |>
        DataFrames.select(_, Not(exc)) |>
        rename(_,
        :"建物現況格局.房" => :建物現況格局_房,
        :"建物現況格局.廳" => :建物現況格局_廳,
        :"建物現況格局.衛" => :建物現況格局_衛,
        :"建物現況格局.隔間" => :建物現況格局_隔間,
        :"車位移轉總面積.平方公尺." => :車位移轉總面積_平方公尺
        )
end

function prob(x, df::DataFrame)
    nrow(x) / sum(nrow(df))
end

function cate_table(df::DataFrame)
    # find cate var
    int =  [eltype.(eachcol(df)) .== Int] +
        [eltype.(eachcol(df)) .== Union{Missing, Int}]
    flo = [eltype.(eachcol(df)) .== Float64]
    num_var = Bool.((int + flo)[1])
    cat_df = df[:, .!num_var]
    for i in 1:length(names(cat_df))
        var = names(cat_df)[i]
        tdf = @pipe groupby(cat_df, var) |>
            combine(_, nrow)
        total = sum(tdf[:, 2])
        @pipe tdf |>
            transform!(_, :nrow => x -> x / total * 100) |>
            rename!(_, [:nrow, :nrow_function] .=> [:n, :prob])
        @show tdf
    end
    println("-----------------------")
end

function splitdf(df::DataFrame, pct::Float64)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    MLJ.shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    train = df[sel, :]
    test = df[.!sel, :]
    return train, test
end

function type_split(df::DataFrame)
    int =  [eltype.(eachcol(df)) .== Int] +
           [eltype.(eachcol(df)) .== Union{Missing, Int}]
    flo = [eltype.(eachcol(df)) .== Float64]
    num_var = Bool.((int + flo)[1])
    return df[:, num_var], df[:, .!num_var]
end

begin
    cont_tp, cate_tp = type_split(ls_tp)
    cont_ks, cate_ks = type_split(ls_ks)
    cont_nt, cate_nt = type_split(ls_nt)
    cont_tc, cate_tc = type_split(ls_tc)
    cont_tn, cate_tn = type_split(ls_tn)
    cont_ty, cate_ty = type_split(ls_ty)
end

# using LinRegOutliers to detect outliets
cont_tp_womv = dropmissing(cont_tp)
formula = @formula(總價元 ~ 土地移轉總面積平方公尺 + 建物移轉總面積平方公尺 +
    建物現況格局_房 + 建物現況格局_廳 + 建物現況格局_衛 + 建物現況格局_隔間 +
    單價元平方公尺 + 車位移轉總面積_平方公尺 + 車位總價元 + 主建物面積 +
    附屬建物面積 + 陽台面積 + 電梯)
reg_tp = createRegressionSetting(formula, cont_tp_womv)
outsmr = smr98(reg_tp)
# outpy = py95(reg_tp) don't run!!!!


# PCA
labels = ls_tp[!, :鄉鎮市區]
label_map = labelmap(label)
uniqueids = labelencode(label_map, label)
data = Matrix(cont_tp_womv)
# standarlized
data = (data .- mean(data, dims = 1)) ./ std(data, dims = 1)
# generate a 2-dims PCA
p = fit(PCA, data', maxoutdim = 2)
P = projection(p)
P'*(data[1,:]-mean(p))
Yte = MultivariateStats.transform(p, data')
Plots.scatter(filte red[1,:], filtered[2,:])
Plots.xlabel!("pca component1")
Plots.ylabel!("pca component2")


# regression tree (ls_tp)
cont_tp_train, cont_tp_test = splitdf(cont_tp, 0.7)
dropmissing!(cont_tp_train)
dropmissing!(cont_tp_test)
labels = cont_tp_train[:, :總價元]
features = Matrix(cont_tp_train[!, Not("總價元")])
model = build_tree(labels, features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   rng = seed)
print_tree(model)
n_subfeatures = 0; max_depth = 5; min_samples_leaf = 5
min_samples_split = 2; min_purity_increase = 0.0; pruning_purity = 1.0 ; seed=3


using EvoTrees
tree_model = DecisionTreeClassifier(max_depth=5)
tree = fit!(tree_model, features, labels)
using ScikitLearn: fit!
print_tree(tree)


#########
using MLJ
using OutlierDetection
using MLJBase
KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors
LOF = @iload LOFDetector pkg=OutlierDetectionNeighbors


#Learning networks
cont_tp_train, cont_tp_test = splitdf(cont_tp, 0.7)
test = dropmissing(cont_tp_train)
Xs = source(test)
Xstd = MLJ.transform(machine(Standardizer(), Xs), Xs)
lof_mach = MLJ.transform(machine(LOF(), Xstd), Xstd)
lof = LOF()
model = fit!(lof_mach)
train_scores, test_scores = score(lof, model, cont_tp_test)
score
fit! # from MLJ
lof_mach(fake_dataframe)
machine
X = rand(100, 10);
lof = machine(LOF(k=5, algorithm=:balltree, leafsize=30, parallel=true), X) |> fit!
transform(lof, X)
predict_proba(lof, X)


using OutlierDetection: fit, transform, scale_minmax, classify_quantile, outlier_fraction
using OutlierDetectionNeighbors: KNNDetector # explicitly import detector
using OutlierDetectionData: ODDS

# outlier detection for taipei
cont_tp_train, cont_tp_test = splitdf(cont_tp, 0.7)
cont_tp_trainF = dropmissing(cont_tp_train)
cont_tp_testF = dropmissing(cont_tp_test)
trainMat = Matrix(cont_tp_trainF)'
testMat = Matrix(cont_tp_testF)'

KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors
model, scores_train = fit(knn, trainMat; verbosity = 0)
scores_test = transform(knn, model, testMat)
proba_train, proba_test = scale_minmax((scores_train, scores_test))
label_train, label_test = classify_quantile(0.95)((scores_train, scores_test))
cont_tp_trainF[label_train .== "normal", :]
cont_tp_testF[label_test .== "normal", :]
mis_df = cont_tp[sum.(eachrow(ismissing.(cont_tp))) .== 1, :]


### building model-based regression tree
a = 10
a = cont_tp.總價元 ./ 1
@rput cont_tp.車位移轉總面積_平方公尺
@rimport party as rparty
.sum(cont_tp.總價元 ./ 1)
R"dim($cont_t)"
sum(cont_tp.總價元)
R"df <- $a"
println("$cont_tp.總價元")
a = round(cont_tp.總價元)
using RCall
R""
a = ls_loc*"LastYear_tp.csv"
R"""
df = read.csv($a)
"""
