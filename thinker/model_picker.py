# takes relevant info about data & returns verdict for models

'''
input :
    task = regression | classification
    ftype = {}
    outlier = 
    extrapol =
'''

    # if n_rown >= upper_threshold boost score++++ why = best for larger databases, elif <= lower_threshold linear, rf score++++ why = good for small datasets, else rf+ (extra + if <medium_threshold) cat++ xg++ lgbm+ (extra + if >medium_threshold)
    # feats : more = rf alright, linear regularize=stabliize but multicollinearity, boost okok 
    # category ( if exceeds encoding feasibility) = catboost++
    # if outlier > threshold RF score++ why = immune to outliers, LR and Boost score-- whynot = outliers affect model,  ridge score---- whynot = overfit to extreme outliers  
    # if extrapol == True ridge score = MAX_INT why = capable of extrapolation unlike the other models in the platter

from dataclasses import dataclass, field

@dataclass
class Model:
    name: str
    score: int = 0
    why: list[str] = field(default_factory=list)
    why_not: list[str] = field(default_factory=list)

def model_picker(task: str, n_rows: int, n_feat: int, ftype: dict, outlier: float, extrapol: bool) -> list[Model]:
    
    # CONFIGS
    R_UPPER = 100_000
    R_MEDIUM = 10_000
    R_LOWER = 1_000
    OUT_THRESHOLD = 0.05  # tweak this
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
    if n_rows >= R_UPPER:
        m["lightgbm"].score += 4
        m["lightgbm"].why.append("best for large datasets")
        m["catboost"].score += 3
        m["catboost"].why.append("scales well to large data")
        m["xgboost"].score += 3
        m["xgboost"].why.append("scales well to large data")

    elif n_rows <= R_LOWER:
        if "logistic_regression" in m:
            m["logistic_regression"].score += 4
            m["logistic_regression"].why.append("reliable on small datasets")
        m["random_forest"].score += 4
        m["random_forest"].why.append("robust on small datasets")
        if "ridge" in m:
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

    # --- feature count ---
    if n_feat > 50:
        m["random_forest"].score += 1
        m["random_forest"].why.append("handles high dimensionality well")
        if "logistic_regression" in m:
            m["logistic_regression"].why_not.append("multicollinearity risk with many features")
        if "ridge" in m:
            m["ridge"].why.append("regularization stabilizes high-dimensional input")
            m["ridge"].score += 1

    # --- categorical features ---
    high_card_cats = ftype.get("high_cardinality_cat", 0)
    if high_card_cats > ENCODING_LIMIT:
        m["catboost"].score += 3
        m["catboost"].why.append("native handling of high cardinality categoricals")

    # --- outliers ---
    if outlier > OUT_THRESHOLD:
        m["random_forest"].score += 2
        m["random_forest"].why.append("immune to outliers")
        for name in ["xgboost", "lightgbm", "catboost"]:
            if name in m:
                m[name].score -= 1
                m[name].why_not.append("somewhat sensitive to outliers")
        if "logistic_regression" in m:
            m["logistic_regression"].score -= 2
            m["logistic_regression"].why_not.append("outliers distort decision boundary")
        if "ridge" in m:
            m["ridge"].score -= 4
            m["ridge"].why_not.append("prone to overfitting extreme outliers")

    # --- extrapolation ---
    if extrapol:
        if "ridge" in m:
            m["ridge"].score = 999
            m["ridge"].why.append("only model in set capable of extrapolation")
        for name in ["random_forest", "catboost", "xgboost", "lightgbm", "tabnet"]:
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