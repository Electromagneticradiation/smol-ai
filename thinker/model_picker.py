# takes relevant info about data & returns verdict for models

'''
input :
    task = regression | classification
    ftype = {"n_num": int, "n_cat": int, "sparsity": list[float]}
            sparsity = ratio of rarest class in each categorical feature (0-1)
    missing = ratio of missing values in data (0-1)
    outlier = ratio of outliers in data (0-1)
'''

from dataclasses import dataclass, field

@dataclass
class Model:
    name: str
    score: int = 0
    why: list[str] = field(default_factory=list)
    why_not: list[str] = field(default_factory=list)

def model_picker(task: str, n_rows: int, n_feat: int, ftype: dict, missing: float, outlier: float, extrapol: bool) -> list[Model]:
    
    # CONFIGS
    R_UPPER = 100_000
    R_MEDIUM = 10_000
    R_LOWER = 1_000
    OUT_THRESHOLD = 0.05
    ENCODING_LIMIT = 10   # max categories before CatBoost preferred

    # dict{name:object} > list[object] for access & understandability of unordered items
    models = {
        "logistic_regression": Model("logistic_regression"),
        "ridge":               Model("ridge"),
        "random_forest":       Model("random_forest"),
        "catboost":            Model("catboost"),
        "xgboost":             Model("xgboost"),
        "lightgbm":            Model("lightgbm"),
    }

    m = models #alias

    # n_rows
    # >= upper_threshold boost score++++, <= lower_threshold linear & rf score++++, else rf+ (extra + if <medium_threshold) cat++ xg++ lgbm+ (extra + if >medium_threshold)
    if n_rows >= R_UPPER:
        m["lightgbm"].score += 4
        m["lightgbm"].why.append("best for large datasets")
        m["catboost"].score += 3
        m["catboost"].why.append("scales well to large data")
        m["xgboost"].score += 3
        m["xgboost"].why.append("scales well to large data")

    elif n_rows <= R_LOWER:
        m["logistic_regression"].score += 4
        m["logistic_regression"].why.append("reliable on small datasets")
        m["random_forest"].score += 4
        m["random_forest"].why.append("robust on small datasets")
        m["ridge"].score += 4
        m["ridge"].why.append("stable on small datasets")
            
    else:
        m["random_forest"].score += 3
        m["random_forest"].why.append("good default for medium datasets")
        m["catboost"].score += 2
        m["xgboost"].score += 2
        m["lightgbm"].score += 2
        if n_rows < R_MEDIUM:
            m["random_forest"].score += 1
            m["random_forest"].why.append("preferred under medium threshold")
        else:
            m["lightgbm"].score += 1
            m["lightgbm"].why.append("preferred over medium threshold")

    # n_feat
    # too many = rf alright, linear regularize=stabliize but multicollinearity, boost okok 
    if n_feat > 50:
        m["random_forest"].score += 1
        m["random_forest"].why.append("handles high dimensionality well")
        m["logistic_regression"].why_not.append("multicollinearity risk with many features")
        m["ridge"].why.append("regularization stabilizes high-dimensional input")
        m["ridge"].score += 1

    # ftype
    n_num = ftype["n_num"]
    n_cat = ftype["n_cat"]
    sparsity = ftype["sparsity"]
    # many categorical features = catboost++ why = native handling, xg++ lgbm+ why = can handle with encoding, LR and Ridge score-- whynot = require heavy encoding for categorical data
    if n_cat > n_num:
        m["catboost"].score += 3
        m["catboost"].why.append("handles categorical features natively")

        for name in ["xgboost", "lightgbm"]:
            m[name].score += 1
            m[name].why.append("can handle categorical with encoding")

        for name in ["logistic_regression", "ridge"]:
            if name in m:
                m[name].score -= 2
                m[name].why_not.append("requires heavy encoding for categorical data")
    # high cardinality (rare categories)
    if sparsity and min(sparsity) < 0.01:
        m["catboost"].score += 2
        m["catboost"].why.append("handles rare categories well")

        for name in ["xgboost", "lightgbm"]:
            m[name].score += 1
            m[name].why.append("can handle high cardinality with encoding")

        for name in ["logistic_regression", "ridge"]:
            if name in m:
                m[name].score -= 2
                m[name].why_not.append("struggles with sparse high-cardinality features")

    # missing
    # too much = boost score++ why = native handling, RF score+ why = can handle with simple strategies, LR and Ridge score-- whynot = require imputation, sensitive to missing data
    if missing > 0.1:
        for name in ["xgboost", "lightgbm", "catboost"]:
            m[name].score += 2
            m[name].why.append("handles missing values natively")

        m["random_forest"].score += 1
        m["random_forest"].why.append("can handle missing with simple strategies")

        for name in ["logistic_regression", "ridge"]:
            if name in m:
                m[name].score -= 2
                m[name].why_not.append("requires imputation, sensitive to missing data")

    # outliers
    # too many = RF score++ why = immune to outliers, LR and Boost score-- whynot = outliers affect model,  ridge score---- whynot = overfit to extreme outliers  
    if outlier > OUT_THRESHOLD:
        m["random_forest"].score += 2
        m["random_forest"].why.append("immune to outliers")

        for name in ["xgboost", "lightgbm", "catboost"]:
            if name in m:
                m[name].score -= 1
                m[name].why_not.append("somewhat sensitive to outliers")

        m["logistic_regression"].score -= 2
        m["logistic_regression"].why_not.append("outliers distort decision boundary")
        
        m["ridge"].score -= 4
        m["ridge"].why_not.append("prone to overfitting extreme outliers")

    # extrapolation
    if extrapol:
        m["ridge"].score = 999
        m["ridge"].why.append("only model in set capable of extrapolation")
        for name in ["random_forest", "catboost", "xgboost", "lightgbm"]:
            if name in m:
                m[name].why_not.append("tree-based models cannot extrapolate beyond training range")
                m[name].score -= 3

    # task : last to avoid deletion before scoring (error handling overhead)
    if task == "regression":
        del m["logistic_regression"]
    elif task == "classification":
        del m["ridge"]

    # sort and return
    result = sorted(m.values(), key=lambda x: x.score, reverse=True)
    return result