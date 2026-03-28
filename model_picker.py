# takes relevant info and returns suitable model(s)

def model_picker(task: str, n_rows: int, n_feat: int, ftype: dict, out, extrapol: bool) :
    '''
    task = reg | class
    ftype = {}
    noise =
    style =
    '''
    
    models = {'log_reg:{'score';'sd'} }

    # data : large 50k+ => boost perfect. small <1k => linear, rf. med => rf = safe but not best, cat, xg, lgbm = kinda risky.. see to it and make subparts 

    # feats : more = rf alright, linear regularize=stabliize but multicollinearity, boost okok 

    # category ( if exceeds encoding feasibility) = catboost++

    # outlier : RF immune, effects boost n LR,  kills ridge  

    # need extrapolation : only ridge can do

    # logreg (no regression), ridge (no classification), ranfor, xgb, lgbm, catb

    if task=="reg" 
    return()

# if (__name__=="__main__") :

# models, why, whynot
# tabnet | decision tree | svm | knn | naive bayes rejected bcz : 