from easyEval.Evaluation.QAEvaluation import QAEvaluation
from easyEval.Evaluation.MutiChoiceEvaluation import MutiChoiceEvaluation

str2evaluation = { "qa": QAEvaluation, "mutichoice": MutiChoiceEvaluation}



__all__ = ["str2datasets"]
