import sys
sys.path.append('./model')
import  mlm_lout_flaten
import  MyModel_fnc

def Generator(model):
    if model == 'mlm_lout_flaten':

        return mlm_lout_flaten.Feature()


def Classifier(model):
    if model == 'mlm_lout_flaten':

        return mlm_lout_flaten.Predictor()

    if model == 'fnc_model':

        return  MyModel_fnc.Predictor()