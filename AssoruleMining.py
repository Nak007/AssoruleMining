'''
Available methods are the followings:
[1] AssoRuleMining
[2] evaluate_rules
[3] define_dtype
[4] discretize

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-05-2022

'''
import pandas as pd, numpy as np, time, os
import collections
from IPython.display import display
import ipywidgets as widgets
import inspect
from itertools import combinations

__all__  = ["AssoRuleMining",
            "evaluate_rules", 
            "define_dtype",
            "discretize"]

def AssoRule_base(X, y, start_with=None, metric="entropy", operator="or",
                  min_lift=1, class_weights=None):
    
    '''
    Using the similar principle as "Association rule", but instead of 
    measuring on "confidence" or "support", it focuses on class 
    attribute i.e. "1" and finds the best RHS (a consequent rule) that
    maximizes selected metric e.g. precision, or decile-lift given LHS
    (antecedent rules).
    
    Parameters
    ----------
    X : pd.DataFrame, shape of (n_samples , n_features)
        Binary variables
        
    y : array-like of shape (n_samples,)
        Target values (binary)
        
    start_with : list of str, optional, default: None
        List of starting features. If None, the first variable that 
        has the highest score from specified `metric` will be 
        seleceted.
    
    operator : {"or", "and"}, default="or"
        If "or", "or" operator is assigned as a relationship between 
        antecedent and consequent rules (n_operands > 0). If "and", 
        "and" operator is assigned (n_operands > 1).
        
    metric : str, default="entropy"
        The function to evaluate the quality of rule (RHS). Supported 
        criteria are "lift" for the decile lift, "recall" for the 
        recall, "precision" for the precision, "f1" for the balanced 
        F-score, and "entropy" for the information gain. 
       
    min_lift : float, default=1
        The minimum per-decile lift required to continue. This is 
        relevant when metric is "lift".
    
    class_weights : "balanced" or dict, default=None
        Weights associated with classes in the form {class_label: 
        weight}. If not given, all classes are supposed to have weight 
        one. The "balanced" mode uses the values of y to automatically 
        adjust weights inversely proportional to class frequencies in 
        the input data as n_samples / (n_classes * np.bincount(y)).
        This is relevant when metric is "entropy".
    
    Returns
    -------
    Results : collections.namedtuple
        A dictionary subclass that contains fields as follows:
        
        Field        Description
        -----        -----------
        metric       Evaluating metric
        start_with   List of starting features
        features     Selected features
        cum_target   Cumulative number of targets
        cum_sample   Cumulative number of samples
        cum_lift     Cumulative lift
        dec_lift     Decile lift
        p_target     % target
        p_sample     % sample
        recall       Recall score
        precision    Precision score
        f1_score     F1 score
        entropy      Entropy
        
        Note: all outputs are arranged according to `features`
        
    '''
    # Initialize parameters
    values = {"metric":metric, "start_with":start_with}
    r_cum_lifts  , r_dec_lifts      = [], []
    r_cum_targets, r_cum_samples    = [], [] 
    recall_scores, precision_scores = [], []
    f1_scores    , r_entropies      = [], []
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"`X` must be DataFrame, "
                         f"Got {type(X)} instead.")
    features  = np.array(X.columns)
    X, target = np.array(X), np.r_[y].reshape(-1,1)
    
    # Remaining, and current indices 
    r_indices, c_indices = np.arange(len(features)), []
    
    # Convert `start_with` to an array of column indices.
    if start_with is None:start_with = np.array([], dtype="int32")
    else: start_with = r_indices[np.isin(features, start_with)]
        
    # Convert `class_weights` to np.ndarray.
    is_entropy = True if metric=="entropy" else False
    class_weights = get_classweights(class_weights, y, is_entropy)
        
    while len(c_indices)<=len(r_indices):
        
        # Remaining features (sorted automatically)
        r_indices = list(set(r_indices).difference(c_indices))

        # Aantecedent and Consequent rules (n_samples,)
        n_operands = 1 if operator=="or" else max(len(c_indices),1)
        antecedent = X[:,c_indices].sum(axis=1,keepdims=True)>=n_operands
        antecedent = antecedent.astype(int)
        consequent = X[:,r_indices]
            
        # New rules
        if (len(c_indices)==0) | (operator=="or"):
            new_rules = (consequent + antecedent) >= 1
        else: new_rules = (consequent + antecedent) >= 2
        args = (target, new_rules.astype(int))
        
        # Lift components
        (cum_targets, cum_samples, 
         cum_lifts, dec_lifts) = lift_base(*(args + (antecedent,)))
        
        # Confusion matrix, recall, precision, f1, and Entropy
        (tp, fp, fn, tn, new_recalls, 
         new_precisions, new_f1s) = cfm_base(*args)
        new_entropies = entropy_base(*(args + (class_weights,)))
        
        # Check metric crieria
        if metric=="lift":
            
            pass_criteria = sum(dec_lifts>=min_lift) > 0
            n = np.argmax(dec_lifts)
            
        elif metric=="precision":
            
            if len(precision_scores)==0: improve = new_precisions
            else: improve = (new_precisions - precision_scores[-1])   
            pass_criteria = sum(improve>0) > 0
            n = np.argmax(improve)
            
        elif metric=="recall":
            
            if len(recall_scores)==0: improve = new_recalls
            else: improve = (new_recalls - recall_scores[-1])
            pass_criteria = sum(improve>0) > 0
            n = np.argmax(improve)
            
        elif metric=="f1":
            
            if len(f1_scores)==0: improve = new_f1s
            else: improve = (new_f1s - f1_scores[-1])
            pass_criteria = sum(improve>0) > 0
            n = np.argmax(improve)
            
        elif metric=="entropy":
        
            if len(r_entropies)==0: 
                p = np.mean(target.ravel())
                p0 = (1-p)*np.log2(1-p) if p<1 else 0
                p1 = p*np.log2(p) if p>0 else 0
                ent = -sum(np.r_[p0, p1] * class_weights)
                improve = ent - new_entropies
            else: improve = (r_entropies[-1] - new_entropies)
            pass_criteria = sum(improve>0)>0
            n = np.argmax(improve)
      
        else: pass
        
        # Need to pass metric criteria
        if pass_criteria:
            
            # Select next `n` from `start_with`.
            if len(start_with)>0: 
                n = np.argmax(r_indices==start_with[0])
                start_with = start_with[1:]
            
            # Cumulative and Per-decile lifts.
            r_cum_lifts += [cum_lifts[n]]
            r_dec_lifts += [dec_lifts[n]]
            
            # Cumulative targets and samples.
            r_cum_targets += [cum_targets[n]]
            r_cum_samples += [cum_samples[n]]
            
            # Recall, Precisions,  F1s, and Entropy
            recall_scores += [new_recalls[n]]
            precision_scores += [new_precisions[n]]
            f1_scores += [new_f1s[n]]
            r_entropies += [new_entropies[n]]
            
            # Adding index to `c_indices`
            c_indices = np.r_[c_indices, r_indices[n]].astype(int)
            
        else: break

    # Returning results
    values.update({"features"   : features[c_indices], 
                   "cum_target" : np.array(r_cum_targets), 
                   "cum_sample" : np.array(r_cum_samples), 
                   "cum_lift"   : np.array(r_cum_lifts), 
                   "dec_lift"   : np.array(r_dec_lifts), 
                   "p_target"   : np.array(r_cum_targets) / sum(y), 
                   "p_sample"   : np.array(r_cum_samples) / len(y), 
                   "recall"     : np.array(recall_scores), 
                   "precision"  : np.array(precision_scores), 
                   "f1_score"   : np.array(f1_scores), 
                   "entropy"    : np.array(r_entropies)})
    
    return collections.namedtuple("Results", values.keys())(**values)

def get_classweights(class_weights, y, is_entropy):
    
    '''
    `class_weights`.
    
    Parameters
    ----------
    class_weights : str or dict {class_label: weight}
    y : np.ndarray, of shape (n_samples,)
    is_entropy : bool
    
    Returns
    -------
    class_weights : np.ndarray of shape (n_classes,)
    
    '''
    n_classes = len(np.unique(y))
    if (class_weights=="balanced") & is_entropy:
        return len(y)/(n_classes*np.bincount(y)) 
    elif isinstance(class_weights, dict) & is_entropy:
        return np.array([class_wights[c] for c in np.unique(y)])
    else: return np.ones(n_classes)

def lift_base(y_true, y_pred, y_prev):
    
    '''
    Cumulative and decile lifts.
    
    Parameters
    ----------
    y_true : np.ndarray, of shape (n_samples, 1)
    y_pred : np.ndarray, of shape (n_samples, n_features)
    y_prev : np.ndarray, of shape (n_samples, 1)
    
    Returns
    -------
    cum_targets, cum_samples, cum_lifts, dec_lifts
    
    '''
    # Number of targets and samples
    n_targets, n_samples  = sum(y_true), len(y_true)
    
    # Cumulative number of targets and samples by varibale
    # of shape (n_features,)
    cum_targets = ((y_pred + y_true)==2).sum(axis=0).astype(int)
    cum_samples = (y_pred==1).sum(axis=0)

    # Cumulative number of existing targets,and samples.
    ext_targets = sum((y_prev + y_true)==2)
    ext_samples = sum(y_prev)

    # Calculate change of targets and samples (n_features,)
    delta_t = (cum_targets - ext_targets) / n_targets 
    delta_s = (cum_samples - ext_samples) / n_samples 

    # Cumulative, and Per-decile lifts
    denom = np.fmax(cum_samples, 0.5) / n_samples
    cum_lifts = (cum_targets/n_targets) / denom
    dec_lifts = delta_t / np.where(delta_s==0, 1, delta_s)
        
    return cum_targets, cum_samples, cum_lifts, dec_lifts

def cfm_base(y_true, y_pred):
    
    '''
    Confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray, of shape (n_samples, 1)
    y_pred : np.ndarray, of shape (n_samples, n_features)
    
    Returns
    -------
    tp, fp, fn, tn, recall, precision, f1
    
    '''
    tp = ((y_pred==1) & (y_true==1)).sum(0)
    fp = ((y_pred==1) & (y_true==0)).sum(0)
    fn = ((y_pred==0) & (y_true==1)).sum(0)
    tn = ((y_pred==0) & (y_true==0)).sum(0)
    recall = tp / np.fmax(tp + fn, 1)
    precision = tp / np.fmax(tp + fp, 1)
    denom = np.fmax(recall + precision, 1)
    f1 = (2 * recall * precision) / denom
    return tp, fp, fn, tn, recall, precision, f1

def entropy_base(y_true, y_pred, class_weights):
    
    '''
    Entropy (Information Gain).
    
    Parameters
    ----------
    y_true : np.ndarray, of shape (n_samples, 1)
    y_pred : np.ndarray, of shape (n_samples, n_features)
    class_weights : np.ndarray, of shape (n_classes,)
    
    Returns
    -------
    Entropies : np.ndarray, of shape (n_features,)
    
    '''
    n_features = y_pred.shape[1]
    Entropies = np.zeros(n_features)
    for c in range(n_features):
        ent, new_x = 0, y_pred[:,c]
        unq, cnt = np.unique(new_x, return_counts=True)
        w = cnt/sum(cnt)
        for n,k in enumerate(unq):
            p = np.mean(y_true[new_x==k,:])
            p0 = (1-p)*np.log2(1-p) if p<1 else 0
            p1 = p*np.log2(p) if p>0 else 0
            ent += -w[n] * sum(np.r_[p0, p1] * class_weights)
        Entropies[c] = ent
    return Entropies

class AssoRuleMining:
    
    '''
    Using the similar principle as "Association rule", but instead of 
    measuring on "confidence" or "support", it focuses on class 
    attribute i.e. "1" and finds the best RHS (a consequent rule) that
    maximizes selected metric e.g. precision, or decile-lift given LHS
    (antecedent rules).
    
    Parameters
    ----------
    metric : str, default="entropy"
        The function to evaluate the quality of rule (RHS). Supported 
        criteria are "lift" for the decile lift, "recall" for the 
        recall, "precision" for the precision, "f1" for the balanced 
        F-score, and "entropy" for the information gain. 
    
    operator : {"or", "and"}, default="or"
        If "or", "or" operator is assigned as a relationship between 
        antecedent and consequent rules (n_operands > 0). If "and", 
        "and" operator is assigned (n_operands > 1).
       
    min_lift : float, default=1
        The minimum per-decile lift required to continue. This is 
        relevant when metric is "lift".
    
    class_weights : "balanced" or dict, default=None
        Weights associated with classes in the form {class_label: 
        weight}. If not given, all classes are supposed to have weight 
        one. The "balanced" mode uses the values of y to automatically 
        adjust weights inversely proportional to class frequencies in 
        the input data as n_samples / (n_classes * np.bincount(y)).
        This is relevant when metric is "entropy".
    
    Attributes
    ----------
    asso_results_ : dict of collections.namedtuple
        A dict with keys as antecedent variables (`start_with` 
        excluded), and namedtuple ("Results") as values, whose fields
        are as follows:
    
        Field        Description
        -----        -----------
        metric       Evaluating metric
        start_with   List of starting features
        features     Selected features
        cum_target   Cumulative number of targets
        cum_sample   Cumulative number of samples
        cum_lift     Cumulative lift
        dec_lift     Decile lift
        p_target     % target
        p_sample     % sample
        recall       Recall score
        precision    Precision score
        f1_score     F1 score
        entropy      Entropy
        
        Note: all outputs are arranged according to `features`
        
    info : dict of numpy (masked) ndarrays
        A dict with keys as column headers. It can be imported into a 
        pandas DataFrame, whose fields are as follows:

        Field       Description
        -----       -----------
        start_with  List of starting features
        variable    First consequent rule (RHS)
        n_features  Number of features
        p_target    % target
        p_sample    % sample
        recall      Recall score
        precision   Precision score
        f1_score    F1 score
        entropy     Entropy
        
    '''
    def __init__(self, metric="entropy", operator="or", 
                 min_lift=1, class_weights=None):
        
        # Keyword arguments for `AssoRule_base`.
        self.kwargs = {"metric" : metric, 
                       "operator" : operator, 
                       "min_lift" : min_lift, 
                       "class_weights" : class_weights}
     
    def fit(self, X, y, start_with=None):
        
        '''
        Fit model.

        Parameters
        ----------
        X : pd.DataFrame, shape of (n_samples , n_features)
            Binary variables

        y : array-like of shape (n_samples,)
            Target values (binary)

        start_with : list of str, optional, default: None
            List of starting features. If None, the first variable that 
            has the highest score from specified `metric` will be 
            seleceted.
 
        '''
        # Initialize widgets
        w1 = widgets.HTMLMath(value='Calculating . . .')
        w2 = widgets.HTMLMath(value='')
        w = widgets.HBox([w1, w2])
        display(w); time.sleep(1)
        
        # Initialize parameters.
        start = time.time()
        self.asso_results_ = dict()
        kwds = {**self.kwargs, **dict(start_with=start_with)}
        if start_with is None: start_with = []
        features = np.r_[list(X)]
        features = features[~np.isin(features,start_with)]
        n_features = len(features)
        
        t = "Start with: {:} ({:,.0%})".format
        for n,var in enumerate(features,1):
            if len(start_with)==0: s = var
            else: s = ", ".join(start_with) + " >>> " + var
            w1.value =  t(s, n/n_features)
            kwds.update({"start_with": start_with + [var]})
            self.asso_results_[var] = AssoRule_base(X, y, **kwds)
        
        w1.value = 'Number of features : %d' % n_features
        r_time = time.gmtime(time.time() - start)
        r_time = time.strftime("%H:%M:%S", r_time)
        w2.value = ', Total running time: {}'.format(r_time)
        self.start_with = start_with
        
        # Create attribute `info`.
        self.__CreateInfo__()
        
        return self
        
    def __CreateInfo__(self):
        
        '''
        Summary of all combinations.
        
        Attributes
        ----------
        info : dict of numpy (masked) ndarrays
            A dict with keys as column headers. It can be imported into a 
            pandas DataFrame, whose fields are as follows:

            Field       Description
            -----       -----------
            start_with  List of starting features
            variable    First consequent rule (RHS)
            n_features  Number of features
            p_targets   % targets
            p_samples   % samples
            recalls     Recall scores
            precisions  Precision scores
            f1_scores   F1 scores
            entropies   Entropies
            
        '''
        data = []
        if len(self.start_with)==0: start = None
        else: start = ", ".join(self.start_with) 
        for var in self.asso_results_.keys():
            a = self.asso_results_[var]
            data += [{"start_with" : start, 
                      "variable"   : var,
                      "n_features" : len(a.features), 
                      "p_target"   : a.p_target[-1], 
                      "p_sample"   : a.p_sample[-1], 
                      "f1_score"   : a.f1_score[-1],
                      "recall"     : a.recall[-1], 
                      "precision"  : a.precision[-1],
                      "entropy"    : a.entropy[-1]}]
        self.info = pd.DataFrame(data).to_dict(orient="list")

def evaluate_rules(eval_set):
    
    '''
    Evaluate set of rules (binary).
    
    Parameters
    ----------  
    eval_set : list of tuples
        A list of tuples, where first item is X of shape (n_samples, 
        n_features) and the seccond item is y of shape (n_samples,) 
        e.g. [(X0, y0), (X1, y1)]. 
        
    Returns
    -------
    EvalResults : collections.namedtuple
        A dictionary subclass that contains fields as follows:
    
        Field      Description
        -----      -----------
        tp         True positive
        fp         False positive
        fn         False negative
        tn         True negative
        recall     Recall scores
        precision  Precision scores
        f1         F1 scores
        
        Note: all outputs are arranged according to `eval_set`
    
    '''
    if not isinstance(eval_set, list):
        raise ValueError(f"`eval_set` has to be list. "
                         f"Got {type(eval_set)} instead.")
    
    data = {"tp":[], "fp":[], "fn":[], "tn":[], 
            "recall":[], "precision":[], "f1":[]}

    for X,y in eval_set:
        y_pred = (np.array(X).sum(1)>0).astype(int)
        y_true = np.array(y)
        tp, fp, fn, tn, recall, precision, f1 = cfm_base(y_true, y_pred)
        data["tp"] += [tp]
        data["fp"] += [fp]
        data["fn"] += [fn]
        data["tn"] += [tn]
        data["precision"] += [precision]
        data["recall"] += [recall]
        data["f1"] += [f1]
    return collections.namedtuple("EvalResults", 
                                  data.keys())(**data)

def define_dtype(X, max_category=100):
    
    '''
    This function converts columns to best possible dtypes which are 
    "float32", "int32" (boolean), "category", and "object". However, 
    it ignores columns, whose dtype is either np.datetime64 or 
    np.timedelta64.
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Input array.
    
    max_category : int, default=100
        If number of unique elements from column with "object" dtype, 
        is less than or equal to max_category, its dtype will be 
        converted to "category". max_category must be greater than or 
        equal to 2.
    
    Returns
    -------
    Converted_X : pd.DataFrame
    
    '''
    # Select columns, whose dtype is neither datetimes, nor timedeltas.
    exclude = [np.datetime64, np.timedelta64] 
    columns = list(X.select_dtypes(exclude=exclude))
    
    if isinstance(max_category, int): 
        max_category = max(2, max_category)
    else: max_category = 100
    
    # Replace pd.isnull() with np.nan
    Converted_X = X.copy()
    Converted_X.iloc[:,:] = np.where(X.isnull(), np.nan, X)
    
    for var in columns:
        x = Converted_X[var].copy()
        try:
            float32 = x.astype("float32")
            if np.isnan(float32).sum()==0:
                int32 = x.astype("int32")
                if (int32-float32).sum()==0: Converted_X[var] = int32
                else: Converted_X[var] = float32
            else: Converted_X[var] = float32 
        except:
            objtype = x.astype("object")
            n_unq = len(objtype.unique())
            if n_unq<=max_category:
                Converted_X[var] = x.astype(str).astype("category") 
            else: Converted_X[var] = objtype
    return Converted_X

def to_DataFrame(X) -> pd.DataFrame:
    
    '''
    If `X` is not `pd.DataFrame`, column(s) will be automatically 
    created with "Unnamed" format.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
    
    '''
    if not (hasattr(X,'shape') or hasattr(X,'__array__')):
        raise TypeError(f'Data must be array-like. '
                        f'Got {type(X)} instead.')
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        try:
            z = int(np.log(X.shape[1])/np.log(10)+1)
            columns = ['Unnamed_{}'.format(str(n).zfill(z)) 
                       for n in range(1,X.shape[1]+1)]
        except: columns = ['Unnamed']
        return pd.DataFrame(X, columns=columns)
    return X

def discretize(X, n_cutoffs=10, decimal=4, equal_width=False, 
               start_index=0):

    '''
    Discretization is the process through which continuous variables 
    can be transformed into a discrete form through use of intervals 
    (or bins). 
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Input data.

    n_cutoffs : int, default=10
        Number of cutoffs. This number could be less than `n_cutoffs` 
        due to rounding of decimals and uniqueness of values.
    
    decimal : int, default=None
        Rounding decimals of bin-edges. If None, rounding is ignored.
    
    equal_width : bool, default=False
        If True, it uses equal-width binning, otherwise equal-sample 
        binning is used instead.
        
    start_index : int, default=0
        Starting index of columns.
        
    Returns
    -------
    discr_X : pd.DataFrame of (n_samples, n_discretized)
        Discretized variables.
        
    conditions : dict
        A dict with keys as column headers or indices in `discr_X`, 
        and values as interval e.g. ("feature", ">", 10)

    '''
    # Initialize parameters
    features, n_samples = list(X), len(X)
    arr, conditions, index = [],  {}, int(start_index)-1
    num_dtypes = ["float32", "float64", "int32", "int64"]
    for var in features:
        x = X[[var]].values
        
        if str(X[var].dtype) in num_dtypes:
            
            bins = cal_bins(x, n_cutoffs, equal_width)
            if decimal is not None:
                bins = np.unique(np.round(bins, decimal)) 
            shape= (n_samples, len(bins))

            # Create varialbes
            for attr, sign in [("less","<"), ("greater_equal",">=")]:
                arr += [getattr(np,attr)(np.full(shape, x)-bins,0)]
                conditions.update(dict([(n,(var, sign, v)) for n,v in 
                                        enumerate(bins, index + 1)]))
                index += len(bins)
        else:
            categories = np.unique(x)
            arr += [np.hstack([x==c for c in categories])]
            conditions.update(dict([(n,(var, "==", v)) for n,v in 
                                    enumerate(categories, index + 1)]))
            index += len(categories)
            
    discr_X = pd.DataFrame(np.hstack(arr).astype(int))
    return discr_X, conditions

def cal_bins(x, bins, equal_width=True):
        
    '''
    According to binning method (equal-width or equal-sample),this 
    function generates 1-dimensional and monotonic array of bins. The 
    last bin edge is the maximum value in x plus np.finfo("float32").
    eps.

    '''
    bins = np.fmax(bins, 2) + 1
    if equal_width: 
        args = (np.nanmin(x), np.nanmax(x), bins)
        bins = np.linspace(*args)
    elif equal_width==False:
        q = np.linspace(0, 100, bins)
        bins = np.unique(np.nanpercentile(x, q))
    bins[-1] = bins[-1] + np.finfo("float32").eps
    return bins