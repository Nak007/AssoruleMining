'''
Available methods are the followings:
[1] AssoRuleMining
[2] define_dtype
[3] discretize
[4] RuleToFeature
[5] print_stats
[6] print_rule
[7] evaluate_rules
[8] create_rule

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-05-2022

'''
import pandas as pd, numpy as np, time, os
from collections import namedtuple
from IPython.display import display
import ipywidgets as widgets
import multiprocessing, numbers
from joblib import Parallel, delayed
from functools import partial
from prettytable import PrettyTable
from sklearn.metrics import (confusion_matrix,f1_score)

__all__  = ["AssoRuleMining", 
            "define_dtype",
            "discretize",
            "RuleToFeature",
            "print_stats",
            "print_rule", 
            "evaluate_rules",
            "create_rule"]

def AssoRule_base(X, y, start_with=None, metric="entropy", operator="and",
                  min_lift=1, class_weights=None, rules=None, 
                  max_features="log2", tol=1e-4):
    
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
    
    operator : {"or", "and"}, default="and"
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
    
    rules : dict, default=None
        A dict with keys as column headers in `X`, and values as 
        interval e.g. {"0": ("feature", ">", 10)}. If provided, `rule` 
        will be added to `asso_results_`. It contains a list of 
        subrules, which defines the specific intervals of a broader
        rule.
     
    max_features : {"sqrt", "log2"}, int or float, default="log2"
        The number of features for stopping criteria.
        - If int, then consider max_features features.
        - If float, then max_features is a fraction and 
          round(max_features * n_features) features.
        - If "auto", then max_features = log2(n_features).
        - If "sqrt", then max_features = sqrt(n_features).
        - If "log2", then max_features = log2(n_features)
    
    tol : float, default=1e-4
        Tolerance for stopping criteria. This is relevant for all
        metric except "lift".
    
    Returns
    -------
    Results : collections.namedtuple
        A dictionary subclass that contains fields as follows:
        
        Field        Description
        -----        -----------
        metric       Evaluating metric
        operator     Relationship between rules
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
        rule         A list of subrules
        
        Note: all outputs are arranged according to `features`
        
    '''
    # Initialize parameters
    values = {"metric": metric,
              "operator": operator, 
              "start_with": start_with}
    r_cum_lifts  , r_dec_lifts      = [], []
    r_cum_targets, r_cum_samples    = [], [] 
    recall_scores, precision_scores = [], []
    f1_scores    , r_entropies      = [], []
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"`X` must be DataFrame, "
                         f"Got {type(X)} instead.")
    features  = np.array(X.columns)
    X, target = np.array(X), np.r_[y].reshape(-1,1)
    max_features_ = get_max_features(max_features, X.shape[1])
    
    if not isinstance(tol, numbers.Number) or (tol < 0):
        raise ValueError("Tolerance for stopping criteria must "
                         "be positive; got (tol=%r)" % tol)

    # Remaining, and current indices 
    r_indices, c_indices = np.arange(len(features)), []
    
    # Convert `start_with` to an array of column indices.
    if start_with is None:start_with = np.array([], dtype="int32")
    else: start_with = r_indices[np.isin(features, start_with)]
        
    # Convert `class_weights` to np.ndarray.
    is_entropy = True if metric=="entropy" else False
    class_weights = get_classweights(class_weights, y, is_entropy)
        
    while len(c_indices)<=max_features_:
        
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
            pass_criteria = sum(improve > tol) > 0
            n = np.argmax(improve)
            
        elif metric=="recall":
            
            if len(recall_scores)==0: improve = new_recalls
            else: improve = (new_recalls - recall_scores[-1])
            pass_criteria = sum(improve > tol) > 0
            n = np.argmax(improve)
            
        elif metric=="f1":
            
            if len(f1_scores)==0: improve = new_f1s
            else: improve = (new_f1s - f1_scores[-1])
            pass_criteria = sum(improve > tol) > 0
            n = np.argmax(improve)
            
        elif metric=="entropy":
        
            if len(r_entropies)==0: 
                p = np.mean(target.ravel())
                p0 = (1-p)*np.log2(1-p) if p<1 else 0
                p1 = p*np.log2(p) if p>0 else 0
                ent = -sum(np.r_[p0, p1] * class_weights)
                improve = ent - new_entropies
            else: improve = (r_entropies[-1] - new_entropies)
            pass_criteria = sum(improve > tol)>0
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
    if rules is None: rule = None
    else: rule = [rules[f] for f in features[c_indices]]
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
                   "entropy"    : np.array(r_entropies), 
                   "rule"       : rule})
    
    return namedtuple("Results", values.keys())(**values)

def get_max_features(max_features, n_features):
    '''Private function: find max number of features'''
    if isinstance(max_features, str):     
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else: raise ValueError("Invalid value for max_features. "
                               "Allowed string values are 'sqrt'"
                               " or 'log2'.")
    elif isinstance(max_features, int):
        return min(max(max_features,1), n_features)
    elif isinstance(max_features, float):
        p = min(1, max_features)
        return max(1, int(p * n_features))
    else: return n_features

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
        weight}. If not given, all classes are supposed to have 
        weight one. The "balanced" mode uses the values of y to 
        automatically adjust weights inversely proportional to class 
        frequencies in the input data as n_samples / (n_classes * 
        np.bincount(y)). This is relevant when metric is "entropy".
        
    max_features : {"sqrt", "log2"}, int or float, default="log2"
        The number of features for stopping criteria.
        - If int, then consider max_features features.
        - If float, then max_features is a fraction and 
          round(max_features * n_features) features.
        - If "auto", then max_features = log2(n_features).
        - If "sqrt", then max_features = sqrt(n_features).
        - If "log2", then max_features = log2(n_features)
    
    tol : float, default=1e-4
        Tolerance for stopping criteria. This is relevant for all
        metric except "lift".
    
    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over combination
        of subrules. None means 1 and -1 means using all processors.
    
    n_batches : int, default=2
        Number of batches (iterations) to process `n_features` 
        combinations.
        
    min_support : int or float, default=0.01
        The minimum support determines how often the rule is 
        applicable to a given `y` or targets, and is used to eliminate 
        some of the k-th features using the support-based pruning 
        strategy.
        - If int, then consider min_support targets.
        - If float, then min_support is a fraction and 
          round(min_support * sum(y)) targets.

    Attributes
    ----------
    asso_results_ : dict of collections.namedtuple
        A dict with keys as antecedent variables (`start_with` 
        excluded), and namedtuple ("Results") as values, whose fields
        are as follows:
    
        Field        Description
        -----        -----------
        metric       Evaluating metric
        operator     Relationship between rules
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
        rule         A list of subrules
        
        Note: all outputs are arranged according to `features`
        
    info : dict of numpy (masked) ndarrays
        A summary table that comes in a form of dict with keys as 
        column headers. It can be imported into a pandas DataFrame, 
        whose fields are as follows:

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
    def __init__(self, metric="entropy", operator="or", min_lift=1, 
                 class_weights=None, max_features="log2", tol=1e-4, 
                 n_jobs=None, n_batches=2, min_support=0.01):
        
        # Keyword arguments for `AssoRule_base`.
        self.kwargs = {"metric" : metric, 
                       "operator" : operator, 
                       "min_lift" : min_lift, 
                       "class_weights" : class_weights, 
                       "max_features" : max_features, 
                       "tol" : tol}
        
        # Number of processors required
        max_CPU = multiprocessing.cpu_count()
        if isinstance(n_jobs, int):
            if n_jobs == -1: self.n_jobs = max_CPU 
            else: self.n_jobs = min(max(1,n_jobs), max_CPU)
        else: self.n_jobs = 1
        
        # Number of batches
        self.n_batches = (max(n_batches, 1) if 
                          isinstance(n_batches,int) else 1)
        
        if not isinstance(min_support, (int,float)):
             raise ValueError(f"Invalid dtype for min_support. "
                              f"Allowed dtype values are int or "
                              f"float. Got {type(min_support)} instead.")
        else: self.min_support = min_support
     
    def fit(self, X, y, start_with=None, rules=None):
        
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
        
        rules : dict, default=None
            A dict with keys as column headers in `X`, and values as 
            interval e.g. {"0": ("feature", ">", 10)}. If provided, `rule` 
            will be added to `asso_results_`. It contains a list of 
            subrules, which defines the specific intervals of a broader
            rule.
 
        '''
        # Initialize widgets
        w1 = widgets.HTMLMath(value='Calculating . . .')
        w2 = widgets.HTMLMath(value='')
        w = widgets.HBox([w1, w2])
        display(w); time.sleep(1)
        
        # Initialize parameters.
        start = time.time()
        self.asso_results_ = dict()
        kwds = {**self.kwargs, 
                **dict(start_with=start_with, rules=rules)}
        if start_with is None: start_with = []
        
        # Support-based pruning
        n_targets = sum(y)
        if isinstance(self.min_support, float):
            min_support = int(min(1, self.min_support)*n_targets)
        min_support = max(min(n_targets, min_support),0)
        
        # Select features (n_supports>=min_support).
        n_supports = ((np.array(X) + y.reshape(-1,1))==2).sum(0)
        features = np.r_[list(X)][n_supports>=min_support]
        features = features[~np.isin(features, start_with)]
        
        # Create batches
        n_features = len(features)
        batch_size = int(np.ceil(n_features/self.n_batches))
        sections = np.arange(0, n_features, batch_size)
        batches  = [b for b in np.split(features,sections) if len(b) > 0]
        n_batches = len(batches)
        
        # Set partial functions.
        assorule = partial(AssoRule_base, **kwds)
        asso_job = Parallel(n_jobs=self.n_jobs)
        
        # Run batch
        t = 'Calculating . . . Batch : ({:,d}/{:,d})'
        for n,batch in enumerate(batches,1):
            results = asso_job([delayed(assorule)\
                                (X, y, start_with=start_with+[var]) 
                                for var in batch])
            for var,res in zip(batch, results):
                self.asso_results_[var] = res
            w1.value = t.format(n, n_batches)
        
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

def define_dtype(X, max_category=100):
    
    '''
    This function converts columns to possible dtypes which are 
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
    ** Private Function **
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
        
    rules : dict
        A dict with keys as column headers (indices) in `discr_X`, 
        and values as interval e.g. ("feature", ">", 10)

    '''
    # Initialize parameters
    features, n_samples = list(X), len(X)
    arr, rules, index = [],  {}, int(start_index)-1
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
                rules.update(dict([(n,(var, sign, v)) for n,v in 
                                   enumerate(bins, index + 1)]))
                index += len(bins)
        else:
            categories = np.unique(x)
            arr += [np.hstack([x==c for c in categories])]
            rules.update(dict([(n,(var, "==", v)) for n,v in 
                               enumerate(categories, index + 1)]))
            index += len(categories)
            
    discr_X = pd.DataFrame(np.hstack(arr).astype(int))
    discr_X.columns = range(start_index, index+1)
    return discr_X, rules

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

def RuleToFeature(X, asso_results_, which_rules=None):
    
    '''
    Convert rules into features array.
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Input data.
    
    asso_results_ : dict of collections.namedtuple
        An attribute from fitted `AssoRuleMining`. The conversion
        requires that every key must contain a list of subrules i.e.
        a field, namely "rule" must not be None.
    
    which_rules : list of keys, default=None
        A list of selectd keys in `asso_results_`. If None, all keys
        will be converted.
    
    Returns
    -------
    Converted_X : pd.DataFrame of shape (n_samples, n_rules)
        A converted X.
    
    rules : dict
        A dict with keys as column headers (indices) in `Converted_X`, 
        and values as tuple subclass named "Rule" e.g. 
        Rule(rule=[('feature', '==', '0')]).
    
    '''
    # Initialize parameters
    Converted_X, rules = [], dict()
    Rule = namedtuple("Rule",["rule","operator"])
    func = {"==": np.equal, "<": np.less, 
            ">=": np.greater_equal}
    
    # Determine number of rules to be converted.
    keys = np.r_[list(asso_results_.keys())]
    which_rules = (keys[np.isin(keys, which_rules)] 
                   if which_rules is not None else keys)

    for key in which_rules:
        # Store an array of converted rule and rule details.
        result = asso_results_[key]
        Converted_X += [rule_alert(X.copy(), result).reshape(-1,1)]
        rules[key] = Rule(*(result.rule, result.operator))
        
    Converted_X = pd.DataFrame(np.hstack(Converted_X),
                               columns=which_rules)        
    return Converted_X, rules

def rule_alert(X:pd.DataFrame, rule:namedtuple) -> np.ndarray:
    '''
    ** Private Function **
    It return True or False given rule (A tuple subclass named 
    "Results")
    '''
    func = {"==": np.equal  , "!=": np.not_equal,
            "<" : np.less   , "<=": np.less_equal,
            ">" : np.greater, ">=": np.greater_equal,}
    bools = np.hstack([func[sign](X[var].values, value).reshape(-1,1) 
                       for var, sign, value in rule.rule]).sum(1)
    if rule.operator=="or": return bools > 0
    else: return bools == len(rule.rule)

def print_stats(y_true, y_pred, decimal=1):
    
    '''
    Prints the summary table.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array
        Estimated targets.
        
    decimal : int, default=1
        Decimal place for %.

    Example
    -------
    >>> print_stats(y_true, y_pred)
    
    +----------------+-------+-------+
    | Statistics     | Value |     % |
    +----------------+-------+-------+
    | N              | 7,000 |       |
    | Target         |   581 |  8.3% |
    | True Positive  |   513 |  7.3% |
    | True Negative  | 6,412 | 91.6% |
    | False Positive |     7 |  0.1% |
    | False Negative |    68 |  1.0% |
    | Precision      |       | 98.7% |
    | Recall         |       | 88.3% |
    | Accuracy       |       | 98.9% |
    | F1-Score       |       | 93.2% |
    +----------------+-------+-------+
        
    '''
    # Calculate all parameters.
    tn0, fp0, fn0, tp0 = confusion_matrix(y_true, y_pred).ravel()
    tn1, fp1, fn1, tp1 = np.r_[tn0, fp0, fn0, tp0]/len(y_true)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    N0, N1 = int(sum(y_true)), sum(y_true)/len(y_true)
    decimal = max(int(decimal), 0)
    fmt1, fmt2 = "{:,d}".format, ("{:." + f"{decimal}" + "%}").format
    
    # Set up summary table
    t = PrettyTable(['Statistics', 'Value', "%"])
    t.align["Statistics"] = "l"
    t.align["Value"] = "r"
    t.align["%"] = "r"
    t.add_row(['N' , fmt1(len(y_true)), ""])
    t.add_row(['Target' , fmt1(N0), fmt2(N1)])
    t.add_row(['True Positive' , fmt1(tp0), fmt2(tp1)])
    t.add_row(['True Negative' , fmt1(tn0), fmt2(tn1)])
    t.add_row(['False Positive', fmt1(fp0), fmt2(fp1)])
    t.add_row(['False Negative', fmt1(fn0), fmt2(fn1)])
    t.add_row(["Precision", "", fmt2(tp0/np.fmax(fp0+tp0,1))])
    t.add_row(["Recall"   , "", fmt2(tp0/np.fmax(fn0+tp0,1))])
    t.add_row(["Accuracy" , "", fmt2((tp0+tn0)/len(y_true))])
    t.add_row(["F1-Score" , "", fmt2(f1)])
    print(t)

def print_rule(rule, decimal=2):
    
    '''
    Prints subrule(s).
    
    Parameters
    ----------
    rules : collections.namedtuple
        A tuple subclass named "Rule" (typename).
    
    decimal : int, default=1
        Decimal place for %.

    Example
    -------
    >>> print_rule(rule, decimal=4)
    
    +------+-------------+------+--------+
    | Item | Varibale    | Sign |  Value |
    +------+-------------+------+--------+
    |  1   | Variable_01 |  ==  |    xxx |
    |  2   | Variable_02 |  >=  | 1.0000 |
    |  3   | Variable_03 |  ==  |     xx |
    +------+-------------+------+--------+
    
    '''
    # Set up summary table
    t = PrettyTable(["Item", 'Varibale', 'Sign', "Value"])
    t.align["Item"] = "c"
    t.align["Varibale"] = "l"
    t.align["Sign"] = "c"
    t.align["Value"] = "r"
    decimal = max(int(decimal),0)
    fmt = ("{:,." +  f"{decimal}" + "f}").format
    for n,h in enumerate(rule.rule,1):
        var, sign, value = h
        if isinstance(value, (int,float)): value = fmt(value)
        t.add_row([n, var, sign, value])
    print("Operator: ",rule.operator)
    print(t)

def evaluate_rules(eval_set, rules, operator="or"):
    
    '''
    Evaluate set of rules.
    
    Parameters
    ----------  
    eval_set : list of tuples
        A list of tuples, where first item is X of shape (n_samples, 
        n_features) and the seccond item is y of shape (n_samples,) 
        e.g. [(X0, y0), (X1, y1)]. 
        
    rules : list of namedtuple
        A list of tuple subclass named "Rule" (typename).
        
    operator : {"or", "and"}, default="or"
        If "or", "or" operator is assigned as a relationship between 
        antecedent and consequent rules (n_operands > 0). If "and", 
        "and" operator is assigned (n_operands > 1).    
        
    Returns
    -------
    EvalResults : collections.namedtuple
        A dictionary subclass that contains fields as follows:
    
        Field      Description
        -----      -----------
        sample     Numer of samples
        target     Number of targets
        tp         True positive
        fp         False positive
        fn         False negative
        tn         True negative
        recall     Recall scores
        precision  Precision scores
        f1         F1 scores
        accuray    Accuracy
        
        Note: all outputs are arranged according to `eval_set`
    
    '''
    if not isinstance(eval_set, list):
        raise ValueError(f"`eval_set` has to be list. "
                         f"Got {type(eval_set)} instead.")
    
    keys = ["sample","target","tp", "fp", "fn", "tn", 
            "recall", "precision","f1", "accuracy"]
    data = dict([(key,[]) for key in keys])

    for X,y in eval_set:
        
        # Determine `y_pred` given rule(s).
        alert = np.vstack([rule_alert(X, rule) for rule in rules]).T
        if operator=="or": y_pred = (alert.sum(1)>0).astype(int)
        else: y_pred = (alert.sum(1)==len(rules)).astype(int)
        y_true = np.array(y).astype(int)
    
        # Calculate all metrics.
        tp, fp, fn, tn, recall, precision, f1 = cfm_base(y_true, y_pred)
        n_samples, n_targets = len(y_true), sum(y_true)
        accuracy = (tp+tn)/len(y_true)
        
        data["sample"] += [n_samples]
        data["target"] += [n_targets]
        data["tp"] += [tp]
        data["fp"] += [fp]
        data["fn"] += [fn]
        data["tn"] += [tn]
        data["precision"] += [precision]
        data["recall"] += [recall]
        data["f1"] += [f1]
        data["accuracy"] += [accuracy]
    return namedtuple("EvalResults", keys)(**data)

def create_rule(subrules, operator="and"):
    '''
    Create a rule.
    
    Parameters
    ----------
    subrules : list of tuples
        A list of subrules in the following format [("variable",
        "sign","value")], e.g. [("var","<=",1000)].
    
    operator : {"or", "and"}, default="or"
        If "or", "or" operator is assigned as a relationship between 
        antecedent and consequent rules. If "and", "and" operator is 
        assigned.
        
    Returns
    -------
    Rule : list of namedtuple
        A list of tuple subclass named "Rule" (typename).
        
    '''
    return namedtuple("Rule", ["rule","operator"])(subrules,operator)