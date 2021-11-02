import numpy as np
import nilearn.decoding
from nilearn.image import index_img, new_img_like
import nibabel as nib
from scipy.io import loadmat
from scipy.special import logit
from scipy.stats import norm
import os
import pdb
import sklearn
import sklearn.pipeline
import nilearn.decoding
from sklearn.base import BaseEstimator, TransformerMixin


def sensitivity_index(y_true, y_pred, extreme_correction='unbiased'):
    """
    Inputs:
        y_true: true binary label vector
        y_pred: predicted binary label vector
        extreme_correction: (see Hautus, 1995)
            None: do no correction
            unbiased: add 0.5 to each cell of confusion matrix (also called LogLinear rule)
            powerful: add 1/2N to 0s in confusion matrix, subtract 1/2N to 1s in confusion matrix (also called 1/2N rule)
    """
    if len(np.unique(y_true))!=2:
        raise ValueError('output variable must be binary')
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    if extreme_correction == 'unbiased':
        tn+=0.5
        fp+=0.5
        fn+=0.5
        tp+=0.5
    elif extreme_correction == 'powerful':
        N = len(y_true)
        for metric in [tn, fp, fn, tp]:
            if metric == 0:
                metric += 1/(2*N)
            elif metric == 0:
                metric -= 1/(2*N)
        
    dprime = norm.ppf(tp/(tp+fn)) - norm.ppf(fp/(fp+tp))
    return dprime


def get_classifier(classifier, 
                   prior='tv-l1',
                   n_alphas=10, 
                   cv=2,
                   n_jobs=12,
                   n_spacenet_alphas=10,
                   ridge_alpha_range=np.logspace(-3,6,num=100),
                   svm_c_range=np.logspace(-3,6,num=10),
                   svm_kernel='linear',
                   k_percent_feats=5,
                  ):
    manual_grid_search = False
    selector = None
    if classifier == 'spacenet':
        clf_args = {'penalty': prior, 'n_alphas': n_spacenet_alphas, 'cv': cv, 'n_jobs': n_jobs, 'verbose': 0}
        clf = nilearn.decoding.SpaceNetClassifier(**clf_args)
    elif classifier == 'ridge':
        clf_args = {'alphas': ridge_alpha_range, 'cv': None}
        standardizer = sklearn.preprocessing.RobustScaler()
        estimator = sklearn.linear_model.RidgeClassifierCV(**clf_args)
        if k_percent_feats == 100:
            selector = IdentityTransformer()
        else:
            selector = sklearn.feature_selection.SelectPercentile(percentile=k_percent_feats)
        clf = sklearn.pipeline.Pipeline([
                    ('standardizer', standardizer),
                    ('selector', selector),
                    ('ridge', estimator),
                    ])
    elif classifier == 'svm':
        estimator = sklearn.svm.SVC(kernel=svm_kernel)
        if k_percent_feats == 100:
            selector = IdentityTransformer()
        else:
            selector = sklearn.feature_selection.SelectPercentile(percentile=k_percent_feats)
        pipeline = sklearn.pipeline.Pipeline([('selector', selector),
                    ('svc', estimator)]) # to match spacenet default
        param_grid = {'svc__C': svm_c_range}
        clf = sklearn.model_selection.GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=n_jobs)
        manual_grid_search = True
    else:
        raise ValueError(f'Not configured for classifier {classifier}')
    return clf, manual_grid_search


def do_mvpa(data, targets,
            kfolds=5,
            classifier='ridge',
            compute_dprime=False,
            **classifier_kwargs,
           ):
    """
    decode data based on target vector, split into chunks (i.e., fMRI runs)
    
    if classifier == 'spacenet', data should be a NiftiImage, else, it should be a numpy array
    """
    
    clf_, manual_grid_search = get_classifier(classifier, **classifier_kwargs)
    splitter = sklearn.model_selection.StratifiedKFold(kfolds, shuffle=True, random_state=1)

    results = {'classifiers':[], 'accuracy':[], 'coefs':[], 'dprime':[]}
    for train_i, test_i in splitter.split(data, targets):
        if classifier == 'spacenet':
            X_train = index_img(data, train_i)
            X_test = index_img(data, test_i)
        else:
            X_train = data[train_i, :]
            X_test = data[test_i, :]
        clf = clf_.fit(X_train, targets[train_i])
        if manual_grid_search:
            clf = clf.best_estimator_
        preds = clf.predict(X_test)
        accuracy = np.mean(np.equal(preds.flatten(),targets[test_i].flatten()))
        results['classifiers'].append(classifier)
        results['accuracy'].append(accuracy)
        if compute_dprime:
            dprime = sensitivity_index(targets[test_i].flatten(), preds, extreme_correction='unbiased')
            results['dprime'].append(dprime)
        if classifier == 'spacenet':
            results['coefs'].append(clf.coef_img_.get_data())
        elif classifier == 'ridge':
            results['coefs'].append(clf[-1].coef_)
            
    return results

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1